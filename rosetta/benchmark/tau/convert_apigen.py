"""Convert Salesforce/APIGen-MT-5k into the C2C training format.

The training pipeline (rosetta.optimize.dataset) expects an HF Dataset with two
JSON-string columns:
  - ``messages``: list of {"role", "content", "tool_calls"?, "tool_call_id"?}
  - ``tools``: list of OpenAI function-calling schemas

APIGen-MT-5k uses a different convention:
  - ``conversations``: list of {"from": role, "value": text}
    Roles: human, gpt, function_call, observation
  - ``tools``: JSON string of tool definitions (name/description/parameters, no
    wrapping "type"/"function" envelope)
  - ``system``: system prompt string

This script bridges the two formats and optionally filters by domain.

Usage::

    python -m rosetta.benchmark.tau.convert_apigen --output local/datasets/apigen_mt_5k
    python -m rosetta.benchmark.tau.convert_apigen --domain airline --limit 100
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from rosetta.optimize.dataset import split_multi_turn


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def detect_domain(system_prompt: str) -> str:
    """Return 'airline' or 'retail' based on system prompt content."""
    lower = system_prompt.lower()
    has_airline = "airline" in lower
    has_retail = "retail" in lower
    if has_airline and has_retail:
        return "unknown"
    if has_airline:
        return "airline"
    if has_retail:
        return "retail"
    return "unknown"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def convert_tools(raw_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wrap bare tool dicts into OpenAI function-calling schema."""
    wrapped = []
    for t in raw_tools:
        wrapped.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        })
    return wrapped


def convert_example(
    conversations: List[Dict[str, str]],
    raw_tools: List[Dict[str, Any]],
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert one APIGen trajectory into our message/tools format.

    Returns (messages, tools) where tools are filtered to only those actually
    called in the trajectory.
    """
    messages: List[Dict[str, Any]] = []
    called_tool_names: set = set()

    # System message first
    messages.append({"role": "system", "content": system_prompt})

    i = 0
    while i < len(conversations):
        msg = conversations[i]
        role = msg["from"]
        value = msg["value"]

        if role == "human":
            messages.append({"role": "user", "content": value})
            i += 1

        elif role == "gpt":
            messages.append({"role": "assistant", "content": value})
            i += 1

        elif role == "function_call":
            # Parse the function call
            call_data = json.loads(value)
            func_name = call_data["name"]
            func_args = call_data.get("arguments", {})
            called_tool_names.add(func_name)

            tc_id = _make_tool_call_id()

            # Build assistant message with tool_calls
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args),
                    },
                }],
            }
            messages.append(assistant_msg)

            # The next message should be the observation
            i += 1
            if i < len(conversations) and conversations[i]["from"] == "observation":
                obs_value = conversations[i]["value"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": obs_value,
                })
                i += 1
            # else: skip (shouldn't happen based on data inspection)

        elif role == "observation":
            # Orphan observation (shouldn't happen), skip
            i += 1
        else:
            i += 1

    # Wrap tools in OpenAI format, filtered to those actually called
    all_tools = convert_tools(raw_tools)
    filtered_tools = [t for t in all_tools if t["function"]["name"] in called_tool_names]

    return messages, filtered_tools


# ---------------------------------------------------------------------------
# Dataset-level conversion
# ---------------------------------------------------------------------------

def convert_dataset(
    domain: str = "all",
    limit: Optional[int] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Load and convert APIGen-MT-5k, returning (hf_dataset, stats)."""
    ds = load_dataset("Salesforce/APIGen-MT-5k", split="train")

    results = []
    n_converted = 0
    errors = 0
    domain_counts: Counter = Counter()
    tool_counts: List[int] = []
    msg_counts: List[int] = []

    for idx, row in enumerate(ds):
        system_prompt = row["system"]
        row_domain = detect_domain(system_prompt)
        domain_counts[row_domain] += 1

        # Domain filter
        if domain != "all" and row_domain != domain:
            continue

        raw_tools = json.loads(row["tools"])
        try:
            messages, tools = convert_example(
                row["conversations"], raw_tools, system_prompt,
            )
        except Exception as e:
            errors += 1
            continue

        tool_counts.append(len(tools))
        msg_counts.append(len(messages))
        n_converted += 1

        for split in split_multi_turn(messages, tools):
            results.append({
                "messages": json.dumps(split["messages"]),
                "tools": json.dumps(split["tools"]),
                "round": split["round"],
                "total_rounds": split["total_rounds"],
                "domain": row_domain,
            })

        if limit is not None and n_converted >= limit:
            break

    hf_ds = Dataset.from_list(results)

    stats = {
        "total_converted": n_converted,
        "total_rows": len(results),
        "errors": errors,
        "domain_counts": dict(domain_counts),
        "avg_tools": sum(tool_counts) / len(tool_counts) if tool_counts else 0,
        "avg_messages": sum(msg_counts) / len(msg_counts) if msg_counts else 0,
    }
    return hf_ds, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Salesforce/APIGen-MT-5k to C2C training format."
    )
    parser.add_argument("--output", default="local/datasets/apigen_mt_5k")
    parser.add_argument(
        "--domain", default="all", choices=["airline", "retail", "all"],
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    hf_ds, stats = convert_dataset(domain=args.domain, limit=args.limit)

    hf_ds.save_to_disk(args.output)

    print(f"Saved {stats['total_rows']} rows ({stats['total_converted']} source examples) to {args.output}")
    print(f"Columns: {hf_ds.column_names}")
    print(f"\nDomain counts (full dataset): {stats['domain_counts']}")
    print(f"Conversion errors: {stats['errors']}")
    print(f"Avg tools per trajectory: {stats['avg_tools']:.1f}")
    print(f"Avg messages per trajectory: {stats['avg_messages']:.1f}")

    # Show a sample
    if len(hf_ds) > 0:
        sample = json.loads(hf_ds[0]["messages"])
        print(f"\nSample message roles: {[m['role'] for m in sample]}")


if __name__ == "__main__":
    main()
