"""Compare minisglang rollout generation vs HF generation for CacheOptimize.

Loads a trained CacheOptimize checkpoint, then compares raw text output
between minisglang and HF (both using the same optimized KV).

Usage:
    CUDA_VISIBLE_DEVICES=5 python script/optimize/test_minisglang_vs_hf.py \
        --rollout-url http://localhost:30001 \
        --dataset local/datasets/countdown/prompt_tool \
        --opt-checkpoint local/checkpoints/opd_cacheopt_countdown_10step/step_10
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.train_utils import (
    RolloutEngine,
    register_tools,
    seed_everything,
)
from rosetta.optimize.wrapper import CacheOptimizeModel
from rosetta.optimize.interface.countdown import _extract_task_info

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, return only the answer part."""
    return THINK_RE.sub("", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="local/datasets/countdown/prompt_tool")
    parser.add_argument("--opt-checkpoint",
                        default="local/checkpoints/opd_cacheopt_countdown_10step/step_10")
    parser.add_argument("--rollout-url", default="http://localhost:30001")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    hf_dataset = load_from_disk(args.dataset)
    hf_dataset = hf_dataset.filter(lambda x: bool(json.loads(x["tools"])))
    split = hf_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Load model + checkpoint
    print(f"Loading student: {args.student} ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )
    opt_model = CacheOptimizeModel(student_model)

    register_tools(opt_model, tokenizer, train_dataset, **tmpl_kwargs)
    register_tools(opt_model, tokenizer, eval_dataset, **tmpl_kwargs)

    print(f"Loading checkpoint: {args.opt_checkpoint} ...")
    opt_model.load_pretrained(args.opt_checkpoint)

    engine = RolloutEngine(args.rollout_url, args.student)
    engine.update_opt_kv(opt_model.get_opt_kv())
    print("Pushed trained KV params to rollout server")

    device = next(student_model.parameters()).device

    # Pick test prompts
    test_items = []
    for i in range(min(args.n_test, len(eval_dataset))):
        item = eval_dataset[i]
        msgs = json.loads(item["messages"])
        tools = json.loads(item["tools"]) or None
        target, nums = _extract_task_info(msgs)
        prompt = [m for m in msgs if m["role"] in ("system", "user")]
        test_items.append({
            "prompt": prompt, "tools": tools,
            "target": target, "nums": nums,
        })

    # === minisglang generation ===
    print("\n--- Generating with minisglang ---")
    extra = {"chat_template_kwargs": tmpl_kwargs} if tmpl_kwargs else {}
    mini_completions = engine.generate(
        [t["prompt"] for t in test_items],
        max_tokens=2048, temperature=0,
        tools_list=[t["tools"] for t in test_items],
        **extra,
    )

    # === HF generation (with optimized KV cache) ===
    print("--- Generating with HF + opt KV ---")
    hf_completions = []
    for item in test_items:
        result = opt_model.prepare_chat(
            tokenizer, item["prompt"], item["tools"],
            for_generate=True,
            template_kwargs=tmpl_kwargs or None,
        )
        prompt_len = result["input_ids"].shape[1]

        with torch.no_grad():
            output = student_model.generate(
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in result.items()},
                max_new_tokens=2048, do_sample=False,
            )

        generated = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        hf_completions.append({"role": "assistant", "content": generated})

    # === Compare raw outputs ===
    print(f"\n{'='*70}")
    print("RAW OUTPUT COMPARISON (minisglang vs HF+optKV)")
    print(f"{'='*70}\n")

    for i, item in enumerate(test_items):
        mini_full = mini_completions[i].get("content", "")
        hf_full = hf_completions[i].get("content", "")

        # Also check reasoning_content field
        mini_reasoning = mini_completions[i].get("reasoning_content", "")

        mini_answer = _strip_think(mini_full)
        hf_answer = _strip_think(hf_full)

        print(f"Problem {i+1}: {item['nums']} -> {item['target']}")
        print(f"  minisglang content len:    {len(mini_full)}")
        print(f"  HF+optKV content len:      {len(hf_full)}")
        if mini_reasoning:
            print(f"  minisglang reasoning_content: {mini_reasoning[:100]}...")
        print(f"  minisglang after </think>: {repr(mini_answer[:200])}")
        print(f"  HF+optKV after </think>:   {repr(hf_answer[:200])}")
        match = "MATCH" if mini_answer == hf_answer else "DIFFER"
        print(f"  [{match}]")

        # Also show first 100 chars of thinking for comparison
        mini_think = mini_full[:200].replace("\n", " ")
        hf_think = hf_full[:200].replace("\n", " ")
        print(f"  minisglang think start: {mini_think}")
        print(f"  HF+optKV think start:   {hf_think}")
        print()

    # Summary
    n_match = sum(
        1 for i in range(len(test_items))
        if _strip_think(mini_completions[i].get("content", "")) ==
           _strip_think(hf_completions[i].get("content", ""))
    )
    print(f"{'='*70}")
    print(f"Answer match: {n_match}/{len(test_items)}")
    print(f"Avg content len - minisglang: "
          f"{sum(len(c.get('content','')) for c in mini_completions)/len(test_items):.0f}, "
          f"HF: {sum(len(c.get('content','')) for c in hf_completions)/len(test_items):.0f}")


if __name__ == "__main__":
    main()
