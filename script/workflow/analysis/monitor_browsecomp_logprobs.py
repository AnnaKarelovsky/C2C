#!/usr/bin/env python3
"""
Monitor neg log prob for tool use with BrowseComp questions.

Uses actual browse_searcher tools (search, get_document) with questions
from the BrowseComp evaluation dataset.

Usage:
    python script/workflow/analysis/monitor_browsecomp_logprobs.py \
        --input local/evaluation/gpt_oss_120b/singletool/browsecomp/full_full_full_0/results.jsonl \
        --runs 3 --questions 2
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TokenLogprob:
    """Single token with logprob info."""
    token: str
    logprob: float

    @property
    def neg_log_prob(self) -> float:
        return -self.logprob


@dataclass
class SectionMetrics:
    """Metrics for a section."""
    name: str
    tokens: List[TokenLogprob]

    @property
    def avg_nlp(self) -> float:
        if not self.tokens:
            return 0.0
        return sum(t.neg_log_prob for t in self.tokens) / len(self.tokens)

    @property
    def count(self) -> int:
        return len(self.tokens)


def parse_logprobs_by_section(
    logprobs_content: List[Dict[str, Any]],
) -> Dict[str, SectionMetrics]:
    """Parse logprobs and categorize by section.

    GPT-OSS format:
    <|channel|>analysis<|message|>reasoning<|end|>
    <|start|>assistant<|channel|>commentary to=functions.xxx <|constrain|>json<|message|>tool_args<|call|>
    <|channel|>final<|message|>text<|return|>
    """
    sections = {
        "reasoning": SectionMetrics("reasoning", []),
        "tool_call": SectionMetrics("tool_call", []),
        "text": SectionMetrics("text", []),
        "all": SectionMetrics("all", []),
    }

    current_section = None
    saw_constrain_json = False

    for item in logprobs_content:
        token = item.get("token", "")
        logprob = item.get("logprob", 0.0)

        token_obj = TokenLogprob(token=token, logprob=logprob)
        sections["all"].tokens.append(token_obj)

        # Detect section transitions
        if token == "<|channel|>":
            current_section = "detecting_channel"
        elif current_section == "detecting_channel":
            if "analysis" in token:
                current_section = "pre_reasoning"
            elif "comment" in token:
                current_section = "pre_tool_header"
            elif "final" in token:
                current_section = "pre_text"
            else:
                current_section = None
        elif token == "<|constrain|>":
            saw_constrain_json = True
        elif token == "<|message|>":
            if current_section == "pre_reasoning":
                current_section = "reasoning"
            elif saw_constrain_json:
                current_section = "tool_call"
                saw_constrain_json = False
            elif current_section == "pre_text":
                current_section = "text"
        elif token in ("<|end|>", "<|call|>", "<|return|>"):
            current_section = None
        elif token == "<|start|>":
            current_section = None
            saw_constrain_json = False
        else:
            if current_section in sections:
                sections[current_section].tokens.append(token_obj)

    return sections


def run_generation_with_tools(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[Dict[str, Any]]:
    """Run a single generation and collect logprobs."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract info
        result = {
            "content": message.content or "",
            "reasoning": getattr(message, "reasoning_content", "") or "",
            "tool_calls": [],
            "logprobs_content": [],
            "finish_reason": choice.finish_reason,
        }

        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        if choice.logprobs and choice.logprobs.content:
            result["logprobs_content"] = [
                {"token": item.token, "logprob": item.logprob}
                for item in choice.logprobs.content
            ]

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def execute_tool_call(tool_call: Dict[str, Any], tools_impl: Dict) -> str:
    """Execute a tool call and return the result."""
    name = tool_call["name"]
    args = json.loads(tool_call["arguments"])

    if name in tools_impl:
        try:
            result = tools_impl[name](**args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown tool: {name}"})


def get_tool_response_logprobs(
    api_key: str,
    model: str,
    tokenizer,
    messages: List[Dict[str, Any]],
    tools_schema: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Get logprobs for tool response tokens using Completions API with echo=True.

    Returns dict with metrics for tool response sections.
    """
    import requests

    # Render and tokenize the conversation
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools_schema,
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
    except Exception as e:
        print(f"    [Tool response tokenization error: {e}]")
        return None

    # Limit tokens for API call
    max_tokens_for_api = 8000
    if len(tokens) > max_tokens_for_api:
        tokens = tokens[:max_tokens_for_api]

    # Call Completions API with echo=True
    url = "https://api.fireworks.ai/inference/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": tokens,
        "max_tokens": 1,
        "echo": True,
        "logprobs": True,
        "top_logprobs": 5,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result = resp.json()

        if "error" in result:
            print(f"    [API error: {result['error']}]")
            return None

        content = result["choices"][0]["logprobs"]["content"]
        prompt_content = content[:len(tokens)]

    except Exception as e:
        print(f"    [API call error: {e}]")
        return None

    # Find tool response sections
    # GPT-OSS format: <|start|>functions.xxx to=assistant<|channel|>commentary<|message|>content<|end|>
    # Note: "functions" and ".xxx" are separate tokens
    tool_response_logprobs = []

    in_tool_response = False
    saw_functions_start = False

    for i, item in enumerate(prompt_content):
        token = item.get("token", "")
        logprob = item.get("logprob")

        # Detect <|start|>functions pattern (tool response header)
        if token == "<|start|>":
            saw_functions_start = False
        elif token == "functions" and i > 0:
            # Check if previous token was <|start|> (tool response)
            # vs "to=functions" (tool call)
            if i > 0 and prompt_content[i-1].get("token") == "<|start|>":
                saw_functions_start = True
        elif token == "<|message|>" and saw_functions_start:
            in_tool_response = True
            saw_functions_start = False
        elif token in ("<|end|}>" , "<|end|>", "<|call|>", "<|return|>"):
            in_tool_response = False
        elif in_tool_response and logprob is not None:
            tool_response_logprobs.append(-logprob)

    if tool_response_logprobs:
        return {
            "tool_response_avg_nlp": sum(tool_response_logprobs) / len(tool_response_logprobs),
            "tool_response_tokens": len(tool_response_logprobs),
            "tool_response_min_nlp": min(tool_response_logprobs),
            "tool_response_max_nlp": max(tool_response_logprobs),
        }

    return None


def run_multi_turn(
    client,
    model: str,
    question: str,
    tools_schema: List[Dict[str, Any]],
    tools_impl: Dict,
    max_turns: int = 5,
    temperature: float = 0.7,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run multi-turn conversation with tool use.

    Returns:
        (results, messages) - results for each turn and final messages list
    """
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Use tools to find information."},
        {"role": "user", "content": question},
    ]

    results = []

    for turn in range(max_turns):
        result = run_generation_with_tools(
            client, model, messages, tools_schema, temperature
        )

        if result is None:
            break

        result["turn"] = turn
        results.append(result)

        # Check if we got tool calls
        if result["tool_calls"]:
            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": result["content"] or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                    }
                    for tc in result["tool_calls"]
                ]
            }
            if result["reasoning"]:
                assistant_msg["reasoning_content"] = result["reasoning"]
            messages.append(assistant_msg)

            # Execute tools and add responses
            for tc in result["tool_calls"]:
                tool_response = execute_tool_call(tc, tools_impl)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_response,
                })

                # Store tool response length for reference
                result["tool_response_length"] = len(tool_response)
        else:
            # No tool calls, generation complete
            break

    return results, messages


def print_turn_summary(result: Dict[str, Any], verbose: bool = False):
    """Print summary for a single turn."""
    turn = result["turn"]
    sections = parse_logprobs_by_section(result["logprobs_content"])

    tool_names = [tc["name"] for tc in result["tool_calls"]] if result["tool_calls"] else ["(answer)"]

    reasoning_nlp = sections["reasoning"].avg_nlp if sections["reasoning"].count > 0 else None
    tool_nlp = sections["tool_call"].avg_nlp if sections["tool_call"].count > 0 else None
    text_nlp = sections["text"].avg_nlp if sections["text"].count > 0 else None

    print(f"  Turn {turn}: {', '.join(tool_names)}")
    print(f"    Reasoning: {sections['reasoning'].count} tokens, avg_nlp={reasoning_nlp:.4f}" if reasoning_nlp else f"    Reasoning: 0 tokens")
    print(f"    Tool Call: {sections['tool_call'].count} tokens, avg_nlp={tool_nlp:.4f}" if tool_nlp else f"    Tool Call: 0 tokens")
    if text_nlp:
        print(f"    Text:      {sections['text'].count} tokens, avg_nlp={text_nlp:.4f}")

    if verbose and sections["tool_call"].count > 0:
        # Show tool call tokens
        print(f"    Tool call tokens:")
        sorted_tokens = sorted(sections["tool_call"].tokens, key=lambda t: t.neg_log_prob, reverse=True)
        for t in sorted_tokens[:10]:
            print(f"      {t.token!r:25} nlp={t.neg_log_prob:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Monitor BrowseComp tool use logprobs")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--questions", "-q", type=int, default=2, help="Number of questions to test")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Runs per question")
    parser.add_argument("--max-turns", type=int, default=3, help="Max turns per run")
    parser.add_argument("--model", type=str, default="accounts/fireworks/models/gpt-oss-20b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed token info")
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("ERROR: FIREWORKS_API_KEY not set")
        sys.exit(1)

    # Load questions
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    questions = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            questions.append({
                "id": record.get("example_id", record.get("idx")),
                "question": record["question"],
                "gold_answer": record.get("gold_answer", ""),
            })
            if len(questions) >= args.questions:
                break

    print(f"Loaded {len(questions)} questions from {input_path.name}")

    # Setup tools
    from rosetta.workflow.browse_searcher import search, get_document
    from camel.toolkits import FunctionTool

    tools_schema = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    tools_impl = {
        "search": search,
        "get_document": get_document,
    }

    # Setup client
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    print("\n" + "=" * 70)
    print("BROWSECOMP TOOL USE LOGPROB MONITOR")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Questions: {args.questions}, Runs per question: {args.runs}")
    print(f"Max turns: {args.max_turns}, Temperature: {args.temperature}")

    # Aggregate stats
    all_sections = {
        "reasoning": [],
        "tool_call": [],
        "text": [],
        "tool_response": [],
    }

    # Load tokenizer for tool response analysis
    from transformers import AutoTokenizer
    print("\nLoading tokenizer for tool response analysis...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

    for q_idx, q in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {q_idx + 1}/{len(questions)}: {q['id']}")
        print(f"Q: {q['question'][:100]}...")
        print(f"Gold: {q['gold_answer']}")
        print("=" * 70)

        for run_idx in range(args.runs):
            print(f"\n>>> Run {run_idx + 1}/{args.runs}")

            results, final_messages = run_multi_turn(
                client,
                args.model,
                q["question"],
                tools_schema,
                tools_impl,
                max_turns=args.max_turns,
                temperature=args.temperature,
            )

            for result in results:
                print_turn_summary(result, verbose=args.verbose)

                # Aggregate generation metrics
                sections = parse_logprobs_by_section(result["logprobs_content"])
                for name in ["reasoning", "tool_call", "text"]:
                    if sections[name].count > 0:
                        all_sections[name].append(sections[name].avg_nlp)

            # Get tool response logprobs for the final conversation
            if any(r["tool_calls"] for r in results):
                print("  Computing tool response logprobs...")
                tool_resp_metrics = get_tool_response_logprobs(
                    api_key,
                    args.model,
                    tokenizer,
                    final_messages,
                    tools_schema,
                )
                if tool_resp_metrics:
                    print(f"    Tool Response: {tool_resp_metrics['tool_response_tokens']} tokens, "
                          f"avg_nlp={tool_resp_metrics['tool_response_avg_nlp']:.4f}, "
                          f"range=[{tool_resp_metrics['tool_response_min_nlp']:.2f}, "
                          f"{tool_resp_metrics['tool_response_max_nlp']:.2f}]")
                    all_sections["tool_response"].append(tool_resp_metrics["tool_response_avg_nlp"])

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"\n{'Section':<15} {'Samples':<10} {'Mean NLP':<12} {'Min':<10} {'Max':<10}")
    print("-" * 57)

    for name in ["reasoning", "tool_call", "tool_response", "text"]:
        nlps = all_sections[name]
        if nlps:
            print(f"{name:<15} {len(nlps):<10} {sum(nlps)/len(nlps):<12.4f} "
                  f"{min(nlps):<10.4f} {max(nlps):<10.4f}")
        else:
            print(f"{name:<15} 0")

    print("\n" + "-" * 57)
    print("Note: tool_call = model generating JSON args (low NLP)")
    print("      tool_response = external search results (high NLP)")


if __name__ == "__main__":
    main()
