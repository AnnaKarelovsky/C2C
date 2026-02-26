"""Convert tau-bench data into C2C cache-optimize training format.

Reads from either APIGen-MT-5k or tau-bench trajectory JSONL, produces
an HF dataset with columns [messages, tools, round, total_rounds].

Usage::

    # APIGen with full domain tools (auto-detected per sample)
    python script/optimize/data/dataset_tau.py \
        --source apigen --full-tool \
        --output local/datasets/full/apigen

    # APIGen with per-trajectory filtered tools (dynamic)
    python script/optimize/data/dataset_tau.py \
        --source apigen \
        --output local/datasets/dynamic/apigen

    # Tau trajectory
    python script/optimize/data/dataset_tau.py \
        --source trajectory \
        --trajectory local/trajectory/tau/test/kimi_k2p5/airline_trajectories.jsonl \
        --output local/datasets/tau_kimi_k2p5_airline
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from datasets import Dataset

from rosetta.benchmark.tau.convert_apigen import convert_dataset as convert_apigen_dataset
from rosetta.benchmark.tau.convert_trajectories import convert_trajectories


def _load_full_tools(domain: str) -> List[Dict[str, Any]]:
    """Load the full tool set for a tau-bench domain."""
    from rosetta.benchmark.tau.interface import get_tools_info
    return get_tools_info(domain)


def _replace_tools_auto(ds: Dataset, replace_prompt: bool = False) -> Dataset:
    """Replace tools (and optionally system prompt) with the eval-exact versions."""
    from rosetta.benchmark.tau.convert_apigen import detect_domain
    from rosetta.benchmark.tau.interface import get_system_prompt

    # Pre-load both domain tool sets and system prompts
    tool_sets = {
        domain: json.dumps(_load_full_tools(domain))
        for domain in ["airline", "retail"]
    }
    prompts = {
        domain: get_system_prompt(domain)
        for domain in ["airline", "retail"]
    } if replace_prompt else {}

    def _replace(example):
        messages = json.loads(example["messages"])
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        domain = detect_domain(system_msg["content"]) if system_msg else "unknown"
        if domain in tool_sets:
            example["tools"] = tool_sets[domain]
        if domain in prompts and system_msg is not None:
            system_msg["content"] = prompts[domain]
            example["messages"] = json.dumps(messages)
        return example

    return ds.map(_replace, desc="Replacing tools/prompt with eval-exact versions")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tau-bench data to C2C training format."
    )
    parser.add_argument("--source", required=True, choices=["apigen", "trajectory"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-tool", action="store_true",
                        help="Replace tools with full domain tool set (auto-detected per sample).")
    parser.add_argument("--full-prompt", action="store_true",
                        help="Replace system prompt with eval-exact wiki prompt (requires --full-tool).")

    # apigen options
    parser.add_argument("--domain", default="all", choices=["airline", "retail", "all"])
    parser.add_argument("--limit", type=int, default=None)

    # trajectory options
    parser.add_argument("--trajectory", default=None, help="Path to trajectory JSONL.")
    parser.add_argument("--records", default=None, help="Records JSONL for reward filtering.")
    parser.add_argument("--min-reward", type=float, default=None)

    args = parser.parse_args()

    if args.source == "apigen":
        hf_ds, stats = convert_apigen_dataset(domain=args.domain, limit=args.limit)
        summary = (
            f"Saved {stats['total_rows']} rows ({stats['total_converted']} examples) "
            f"to {args.output}\n"
            f"Domain: {args.domain} | "
            f"Avg tools: {stats['avg_tools']:.1f} | "
            f"Avg messages: {stats['avg_messages']:.1f}"
        )
    elif args.source == "trajectory":
        if not args.trajectory:
            parser.error("--trajectory is required for source=trajectory")
        hf_ds, stats = convert_trajectories(
            args.trajectory, args.records, args.min_reward,
        )
        summary = (
            f"Saved {stats['total_rows']} rows from {stats['total_trajectories']} "
            f"trajectories to {args.output}\n"
            f"Tools: {stats['num_tools']} | "
            f"Avg messages: {stats['avg_messages']:.1f}"
        )

    if args.full_tool:
        hf_ds = _replace_tools_auto(hf_ds, replace_prompt=args.full_prompt)
        summary += "\nReplaced tools with full domain set (auto-detected per sample)"
        if args.full_prompt:
            summary += "\nReplaced system prompt with eval-exact wiki prompt"

    hf_ds.save_to_disk(args.output)
    print(summary)
    print(f"Columns: {hf_ds.column_names}")


if __name__ == "__main__":
    main()
