"""Quick accuracy test for Countdown with current prompt tool.

Tests models via their rollout engines.

Usage:
    python script/optimize/test_countdown_acc.py \
        --url http://localhost:30001 \
        --model Qwen/Qwen3-1.7B \
        --n-problems 20
"""

from __future__ import annotations

import argparse
import json

from rosetta.optimize.interface.countdown import (
    PROMPT_TOOL,
    SYSTEM_PROMPT,
    _countdown_reward,
    _extract_answer_text,
    _make_user_prompt,
)
from rosetta.optimize.train_utils import RolloutEngine


def test_model(engine: RolloutEngine, label: str, problems: list, n_samples: int = 1,
               verbose: bool = False):
    """Run countdown problems through a model and report accuracy."""
    prompts = []
    tools_list = []
    for target, nums in problems:
        for _ in range(n_samples):
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _make_user_prompt(target, nums)},
            ])
            tools_list.append(PROMPT_TOOL)

    completions = engine.generate(
        prompts, max_tokens=2048, temperature=0.9,
        tools_list=tools_list,
    )

    total_correct = 0
    for i, (target, nums) in enumerate(problems):
        sample_comps = completions[i * n_samples : (i + 1) * n_samples]
        scores = [_countdown_reward(c, target, nums) for c in sample_comps]
        n_correct = sum(scores)
        total_correct += n_correct

        c = sample_comps[0]
        content = c.get("content", "")
        answer = _extract_answer_text(content) if content else None
        tool_calls = c.get("tool_calls", [])

        print(f"  Problem {i+1}: {nums} -> {target}")
        print(f"    <answer>: {repr(answer)}")
        print(f"    content len: {len(content)}")
        if tool_calls:
            print(f"    tool_calls: {len(tool_calls)}")
            for tc in tool_calls:
                fn = tc.get("function", {})
                print(f"      {fn.get('name')}: {fn.get('arguments', '')[:200]}")
        print(f"    reward: {scores[0]}")

    acc = total_correct / (len(problems) * n_samples)
    print(f"\n  {label}: {total_correct}/{len(problems)*n_samples} = {acc:.1%}\n")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", nargs="+", required=True,
                        help="Server URLs (e.g. http://localhost:30001)")
    parser.add_argument("--model", nargs="+", required=True,
                        help="Model names (same count as --url)")
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    assert len(args.url) == len(args.model), "--url and --model must have same count"

    # Load test problems
    from rosetta.optimize.interface.countdown import _load_countdown_test
    problems = _load_countdown_test(n=args.n_problems)
    print(f"Loaded {len(problems)} test problems\n")

    # Show current tool schema
    print("Current tool schema:")
    print(json.dumps(PROMPT_TOOL[0]["function"], indent=2))
    print()

    results = {}
    for url, model in zip(args.url, args.model):
        print(f"=== Testing {model} ({url}) ===")
        engine = RolloutEngine(url, model)
        acc = test_model(engine, model, problems, verbose=args.verbose)
        results[model] = acc

    print("=" * 60)
    print("Summary:")
    for model, acc in results.items():
        print(f"  {model}: {acc:.1%}")


if __name__ == "__main__":
    main()
