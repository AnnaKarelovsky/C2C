#!/usr/bin/env python3
"""
Monitor neg log prob for tool use during GPT-OSS generation.

This script:
1. Calls GPT-OSS Chat Completions API with tools
2. Collects logprobs from generated tokens
3. Tracks neg log prob for different parts (reasoning, tool_call, text)
4. Runs multiple iterations to analyze patterns

Usage:
    python script/workflow/analysis/monitor_tool_logprobs.py --runs 5
    python script/workflow/analysis/monitor_tool_logprobs.py --runs 10 --query "What's the weather in Tokyo?"
    python script/workflow/analysis/monitor_tool_logprobs.py --runs 3 --max-turns 4 --detailed

Findings:
---------
Tool CALL neg_log_prob is typically very low (~0.001-0.01) because:
1. Arguments often come directly from user input (near-zero NLP)
2. JSON syntax is highly constrained (structure tokens like '{"', '":"' have low NLP)
3. Model knows which tool to call based on context

Exceptions (higher NLP in tool calls):
- Model-generated query text (e.g., adding "research papers" to search): NLP ~0.07-0.14
- Creative argument choices: NLP ~0.2-2.0

This contrasts sharply with tool RESPONSE neg_log_prob (~10-15) where:
- External data (names, numbers, URLs) is unpredictable
- Model cannot anticipate what tools return

Section NLP Summary (typical):
- Reasoning: ~0.3-0.7 (model's free-form thinking)
- Tool Call: ~0.001-0.01 (constrained JSON with known arguments)
- Tool Response: ~10-15 (external unpredictable data)
- Text Output: ~0.1-0.5 (model's response based on context)
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
    token_id: int
    logprob: float

    @property
    def neg_log_prob(self) -> float:
        return -self.logprob


@dataclass
class GenerationResult:
    """Result from a single generation."""
    run_id: int
    content: str
    reasoning: str
    tool_calls: List[Dict[str, Any]]
    logprobs: List[TokenLogprob]

    # Metrics by section
    reasoning_tokens: List[TokenLogprob] = field(default_factory=list)
    tool_call_tokens: List[TokenLogprob] = field(default_factory=list)
    text_tokens: List[TokenLogprob] = field(default_factory=list)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute average neg log prob for each section."""
        metrics = {}

        if self.reasoning_tokens:
            nlps = [t.neg_log_prob for t in self.reasoning_tokens]
            metrics["reasoning_avg_nlp"] = sum(nlps) / len(nlps)
            metrics["reasoning_tokens"] = len(nlps)

        if self.tool_call_tokens:
            nlps = [t.neg_log_prob for t in self.tool_call_tokens]
            metrics["tool_call_avg_nlp"] = sum(nlps) / len(nlps)
            metrics["tool_call_tokens"] = len(nlps)

        if self.text_tokens:
            nlps = [t.neg_log_prob for t in self.text_tokens]
            metrics["text_avg_nlp"] = sum(nlps) / len(nlps)
            metrics["text_tokens"] = len(nlps)

        if self.logprobs:
            nlps = [t.neg_log_prob for t in self.logprobs]
            metrics["overall_avg_nlp"] = sum(nlps) / len(nlps)
            metrics["total_tokens"] = len(nlps)

        return metrics


def parse_logprobs_by_section(
    logprobs_content: List[Dict[str, Any]],
) -> tuple[List[TokenLogprob], List[TokenLogprob], List[TokenLogprob], List[TokenLogprob]]:
    """Parse logprobs and categorize by section based on special tokens.

    GPT-OSS generation format:
    <|channel|>analysis<|message|>reasoning<|end|>
    <|start|>assistant<|channel|>commentary to=functions.xxx <|constrain|>json<|message|>tool_args<|call|>
    OR
    <|channel|>final<|message|>text<|return|>

    Returns:
        (all_tokens, reasoning_tokens, tool_call_tokens, text_tokens)
    """
    all_tokens = []
    reasoning_tokens = []
    tool_call_tokens = []
    text_tokens = []

    current_section = None  # "reasoning", "tool_call", "text", or None
    saw_constrain_json = False

    for item in logprobs_content:
        token = item.get("token", "")
        token_bytes = item.get("bytes", [])
        logprob = item.get("logprob", 0.0)

        # Try to get token_id from bytes or estimate
        if token_bytes and len(token_bytes) > 0:
            # bytes field contains UTF-8 bytes, not token ID
            token_id = -1  # Not available directly from Chat API
        else:
            token_id = -1

        token_obj = TokenLogprob(token=token, token_id=token_id, logprob=logprob)
        all_tokens.append(token_obj)

        # Detect section transitions based on special tokens
        if token == "<|channel|>":
            current_section = "detecting_channel"
        elif current_section == "detecting_channel":
            if "analysis" in token:
                current_section = "pre_reasoning"
            elif "comment" in token:  # "commentary" is split as "comment" + "ary"
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
        elif token in ("<|end|}>" , "<|end|>"):
            current_section = None
        elif token == "<|call|>":
            current_section = None
        elif token in ("<|return|>",):
            current_section = None
        elif token == "<|start|>":
            # New message starts, reset state
            current_section = None
            saw_constrain_json = False
        else:
            # Assign token to current section
            if current_section == "reasoning":
                reasoning_tokens.append(token_obj)
            elif current_section == "tool_call":
                tool_call_tokens.append(token_obj)
            elif current_section == "text":
                text_tokens.append(token_obj)

    return all_tokens, reasoning_tokens, tool_call_tokens, text_tokens


def run_generation(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    run_id: int,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Optional[GenerationResult]:
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

        # Extract content
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", "") or ""

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        # Parse logprobs
        logprobs_content = []
        if choice.logprobs and choice.logprobs.content:
            logprobs_content = [
                {
                    "token": item.token,
                    "bytes": item.bytes,
                    "logprob": item.logprob,
                }
                for item in choice.logprobs.content
            ]

        all_tokens, reasoning_tokens, tool_call_tokens, text_tokens = \
            parse_logprobs_by_section(logprobs_content)

        result = GenerationResult(
            run_id=run_id,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            logprobs=all_tokens,
            reasoning_tokens=reasoning_tokens,
            tool_call_tokens=tool_call_tokens,
            text_tokens=text_tokens,
        )

        return result

    except Exception as e:
        print(f"  Run {run_id} ERROR: {e}")
        return None


def simulate_tool_response(tool_call: Dict[str, Any]) -> str:
    """Generate a simulated tool response."""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", "{}")

    try:
        args_dict = json.loads(args)
    except json.JSONDecodeError:
        args_dict = {}

    if name == "get_weather":
        city = args_dict.get("city", "Unknown")
        return json.dumps({
            "city": city,
            "temperature": 22,
            "condition": "sunny",
            "humidity": 65,
        })
    elif name == "search":
        query = args_dict.get("query", "")
        return json.dumps({
            "results": [
                {"title": f"Result 1 for {query}", "snippet": "This is a sample result..."},
                {"title": f"Result 2 for {query}", "snippet": "Another sample result..."},
            ],
            "total": 2,
        })
    else:
        return json.dumps({"status": "ok", "result": "simulated response"})


def run_multi_turn_generation(
    client,
    model: str,
    initial_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    run_id: int,
    max_turns: int = 3,
    temperature: float = 0.7,
) -> List[GenerationResult]:
    """Run multi-turn generation with tool use."""
    results = []
    messages = list(initial_messages)

    for turn in range(max_turns):
        result = run_generation(
            client, model, messages, tools,
            run_id=run_id * 100 + turn,
            temperature=temperature,
        )

        if result is None:
            break

        results.append(result)

        # Check if we got tool calls
        if result.tool_calls:
            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                    }
                    for tc in result.tool_calls
                ]
            }
            if result.reasoning:
                assistant_msg["reasoning_content"] = result.reasoning
            messages.append(assistant_msg)

            # Add tool responses
            for tc in result.tool_calls:
                tool_response = simulate_tool_response(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_response,
                })
        else:
            # No tool calls, generation complete
            break

    return results


def print_results_summary(all_results: List[List[GenerationResult]]):
    """Print summary of all results."""
    print("\n" + "=" * 70)
    print("SUMMARY: Neg Log Prob by Section")
    print("=" * 70)

    # Aggregate metrics
    reasoning_nlps = []
    tool_call_nlps = []
    text_nlps = []
    overall_nlps = []

    for run_results in all_results:
        for result in run_results:
            metrics = result.compute_metrics()
            if "reasoning_avg_nlp" in metrics:
                reasoning_nlps.append(metrics["reasoning_avg_nlp"])
            if "tool_call_avg_nlp" in metrics:
                tool_call_nlps.append(metrics["tool_call_avg_nlp"])
            if "text_avg_nlp" in metrics:
                text_nlps.append(metrics["text_avg_nlp"])
            if "overall_avg_nlp" in metrics:
                overall_nlps.append(metrics["overall_avg_nlp"])

    print(f"\n{'Section':<20} {'Count':<10} {'Mean NLP':<12} {'Min':<12} {'Max':<12}")
    print("-" * 66)

    if reasoning_nlps:
        print(f"{'Reasoning':<20} {len(reasoning_nlps):<10} "
              f"{sum(reasoning_nlps)/len(reasoning_nlps):<12.4f} "
              f"{min(reasoning_nlps):<12.4f} {max(reasoning_nlps):<12.4f}")

    if tool_call_nlps:
        print(f"{'Tool Call':<20} {len(tool_call_nlps):<10} "
              f"{sum(tool_call_nlps)/len(tool_call_nlps):<12.4f} "
              f"{min(tool_call_nlps):<12.4f} {max(tool_call_nlps):<12.4f}")

    if text_nlps:
        print(f"{'Text':<20} {len(text_nlps):<10} "
              f"{sum(text_nlps)/len(text_nlps):<12.4f} "
              f"{min(text_nlps):<12.4f} {max(text_nlps):<12.4f}")

    if overall_nlps:
        print(f"{'Overall':<20} {len(overall_nlps):<10} "
              f"{sum(overall_nlps)/len(overall_nlps):<12.4f} "
              f"{min(overall_nlps):<12.4f} {max(overall_nlps):<12.4f}")


def print_detailed_run(result: GenerationResult):
    """Print detailed info for a single run."""
    print(f"\n--- Run {result.run_id} ---")

    if result.reasoning:
        print(f"Reasoning ({len(result.reasoning)} chars): {result.reasoning[:100]}...")

    if result.tool_calls:
        print(f"Tool calls: {len(result.tool_calls)}")
        for tc in result.tool_calls:
            print(f"  - {tc['name']}({tc['arguments'][:50]}...)")

    if result.content:
        print(f"Content: {result.content[:100]}...")

    metrics = result.compute_metrics()
    print(f"Metrics:")
    for key, value in metrics.items():
        if "nlp" in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Show high neg log prob tokens in tool calls
    if result.tool_call_tokens:
        high_nlp_tokens = sorted(result.tool_call_tokens, key=lambda t: t.neg_log_prob, reverse=True)[:10]
        print(f"\nTop 10 highest NLP tokens in tool_call:")
        for t in high_nlp_tokens:
            print(f"  {t.token!r:20} nlp={t.neg_log_prob:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Monitor tool use neg log prob")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--query", type=str, default="What's the weather in Paris?",
                        help="User query")
    parser.add_argument("--model", type=str,
                        default="accounts/fireworks/models/gpt-oss-20b",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-turns", type=int, default=2,
                        help="Max turns per run")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed output for each run")
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("ERROR: FIREWORKS_API_KEY not set")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Initial messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": args.query},
    ]

    print("=" * 70)
    print("TOOL USE NEG LOG PROB MONITOR")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print(f"Runs: {args.runs}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")

    all_results = []

    for run_id in range(args.runs):
        print(f"\n>>> Run {run_id + 1}/{args.runs}")

        results = run_multi_turn_generation(
            client,
            args.model,
            messages,
            tools,
            run_id=run_id,
            max_turns=args.max_turns,
            temperature=args.temperature,
        )

        all_results.append(results)

        if args.detailed:
            for result in results:
                print_detailed_run(result)
        else:
            # Brief summary
            for result in results:
                metrics = result.compute_metrics()
                tool_nlp = metrics.get("tool_call_avg_nlp", "N/A")
                reasoning_nlp = metrics.get("reasoning_avg_nlp", "N/A")
                tool_nlp_str = f"{tool_nlp:.4f}" if isinstance(tool_nlp, float) else tool_nlp
                reasoning_nlp_str = f"{reasoning_nlp:.4f}" if isinstance(reasoning_nlp, float) else reasoning_nlp

                tool_names = [tc["name"] for tc in result.tool_calls] if result.tool_calls else ["(no tools)"]
                print(f"  Turn {result.run_id % 100}: tools={tool_names}, "
                      f"reasoning_nlp={reasoning_nlp_str}, tool_call_nlp={tool_nlp_str}")

    # Print summary
    print_results_summary(all_results)

    # Show token-level analysis for tool calls
    print("\n" + "=" * 70)
    print("TOKEN-LEVEL ANALYSIS: Tool Call Tokens")
    print("=" * 70)

    all_tool_tokens = []
    for run_results in all_results:
        for result in run_results:
            all_tool_tokens.extend(result.tool_call_tokens)

    if all_tool_tokens:
        # Group by token
        token_stats = {}
        for t in all_tool_tokens:
            if t.token not in token_stats:
                token_stats[t.token] = []
            token_stats[t.token].append(t.neg_log_prob)

        # Sort by average NLP
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )

        print(f"\nTop 20 tokens by average NLP in tool calls:")
        print(f"{'Token':<25} {'Count':<8} {'Avg NLP':<12} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        for token, nlps in sorted_tokens[:20]:
            print(f"{token!r:<25} {len(nlps):<8} "
                  f"{sum(nlps)/len(nlps):<12.4f} "
                  f"{min(nlps):<10.4f} {max(nlps):<10.4f}")


if __name__ == "__main__":
    main()
