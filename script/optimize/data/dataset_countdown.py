"""Convert Countdown-Task-GOLD dataset into C2C training format.

Reads from ``HuggingFaceTB/Countdown-Task-GOLD`` on Hugging Face (verified
subset with correct model answers), produces an HF dataset with columns
[messages, tools].

Usage::

    python script/optimize/data/dataset_countdown.py \
        --output local/datasets/countdown/prompt_tool
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset, load_dataset

from rosetta.optimize.interface.countdown import PROMPT_TOOL


def main():
    parser = argparse.ArgumentParser(
        description="Convert Countdown data to C2C training format."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--subset",
        default="verified_Qwen2.5-7B-Instruct",
        help="Dataset config name (default: verified_Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src = load_dataset(
        "HuggingFaceTB/Countdown-Task-GOLD", args.subset, split="train",
    )
    if args.limit:
        src = src.select(range(min(args.limit, len(src))))

    tools_json = json.dumps(PROMPT_TOOL)
    rows = []
    for row in src:
        messages = row["messages"]  # already a list of dicts
        rows.append({"messages": json.dumps(messages), "tools": tools_json})

    hf_ds = Dataset.from_list(rows)
    hf_ds.save_to_disk(args.output)
    print(
        f"Saved {len(hf_ds)} rows to {args.output}\n"
        f"Columns: {hf_ds.column_names}"
    )


if __name__ == "__main__":
    main()
