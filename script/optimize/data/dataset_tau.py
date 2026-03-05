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

from rosetta.benchmark.tau.convert_apigen import (
    convert_dataset as convert_apigen_dataset,
    detect_domain,
)
from rosetta.benchmark.tau.convert_trajectories import convert_trajectories
from rosetta.benchmark.tau.interface import (
    get_system_prompt as tau1_system_prompt,
    get_tools_info as tau1_tools_info,
)
from rosetta.benchmark.tau2.interface import (
    get_environment as tau2_get_environment,
    get_system_prompt as tau2_system_prompt,
    get_tools_info as tau2_tools_info,
)


def _load_full_tools(domain: str) -> List[Dict[str, Any]]:
    """Load the full tool set for a tau-bench domain."""
    return tau1_tools_info(domain)


def _replace_tools_auto(
    ds: Dataset,
    full_set: bool = False,
    tool_source: str = "tau",
) -> Dataset:
    """Replace tool descriptions and system prompt with eval-exact versions.

    Args:
        ds: Dataset with ``tools`` and ``messages`` JSON-string columns.
        full_set: If True, expand to the full domain tool set (all tools).
            If False, keep only the per-trajectory filtered subset but with
            matched descriptions.
        tool_source: Which tool schemas to use:
            - ``"apigen"``: skip replacement entirely (keep original APIGen descriptions)
            - ``"tau"``: replace with tau1 eval-exact schemas (default, original behavior)
            - ``"tau2"``: replace with tau2 environment schemas
    """
    if tool_source == "apigen":
        return ds

    if tool_source == "tau2":
        domains = ["airline", "retail"]
        full_tools = {}
        prompts = {}
        for domain in domains:
            env = tau2_get_environment(domain)
            full_tools[domain] = tau2_tools_info(env)
            prompts[domain] = tau2_system_prompt(env)
    else:
        domains = ["airline", "retail"]
        full_tools = {domain: _load_full_tools(domain) for domain in domains}
        prompts = {domain: tau1_system_prompt(domain) for domain in domains}

    full_tools_json = {
        domain: json.dumps(tools)
        for domain, tools in full_tools.items()
    }
    schema_by_name: Dict[str, Dict[str, Any]] = {}
    for tools in full_tools.values():
        for t in tools:
            schema_by_name[t["function"]["name"]] = t

    desc_label = "tau2" if tool_source == "tau2" else "tau v1"

    def _replace(example):
        messages = json.loads(example["messages"])
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        domain = detect_domain(system_msg["content"]) if system_msg else "unknown"

        if domain in full_tools_json:
            if full_set:
                example["tools"] = full_tools_json[domain]
            else:
                # Keep filtered subset but swap each schema
                row_tools = json.loads(example["tools"])
                replaced = []
                for t in row_tools:
                    name = t["function"]["name"]
                    if name in schema_by_name:
                        replaced.append(schema_by_name[name])
                    else:
                        replaced.append(t)
                example["tools"] = json.dumps(replaced)

        if system_msg is not None and domain in prompts:
            system_msg["content"] = prompts[domain]
            example["messages"] = json.dumps(messages)
        return example

    return ds.map(_replace, desc=f"Replacing tool schemas with {desc_label} eval-exact versions")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tau-bench data to C2C training format."
    )
    parser.add_argument("--source", required=True, choices=["apigen", "trajectory"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-tool", action="store_true",
                        help="Replace tools with full domain tool set (auto-detected per sample).")
    parser.add_argument("--tool-source", default="tau", choices=["apigen", "tau", "tau2"],
                        help="Which tool schemas to use: apigen (original), tau (v1 eval-exact), tau2 (v2 environment).")

    # apigen options
    parser.add_argument("--domain", default="all", choices=["airline", "retail", "telecom", "all"])
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

    # Replace tool descriptions and system prompt unless --tool-source apigen.
    # --full-tool additionally expands to the full domain tool set.
    hf_ds = _replace_tools_auto(hf_ds, full_set=args.full_tool, tool_source=args.tool_source)
    if args.tool_source == "apigen":
        summary += "\nKept original APIGen tool descriptions (no replacement)"
    elif args.full_tool:
        summary += f"\nReplaced tools with full domain set ({args.tool_source} descriptions)"
    else:
        summary += f"\nReplaced tool descriptions with {args.tool_source} eval-exact versions"
    if args.tool_source != "apigen":
        summary += f"\nReplaced system prompt with {args.tool_source} wiki prompt"

    hf_ds.save_to_disk(args.output)
    print(summary)
    print(f"Columns: {hf_ds.column_names}")


if __name__ == "__main__":
    main()
