"""Convert AIME dataset into C2C cache-optimize training format.

Reads from ``gneubig/aime-1983-2024`` on Hugging Face, produces an HF dataset
with columns [messages, tools].

Usage::

    python script/optimize/data/dataset_aime.py \
        --output local/datasets/aime/prompt_tool
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset, load_dataset

PROMPT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "prompt_tool",
            "description": (
                "This is NOT a tool to be called. It is a placeholder for "
                "instructions.\n\nDo NOT call this tool. Instead, follow the "
                "instructions below to guide your behavior when responding to "
                "user queries.\n\nGeneral guidelines:\n- Think step by step "
                "before answering.\n- Be concise and accurate in your "
                "responses.\n- Use available tools when you need external "
                "information."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

SYSTEM_PROMPT = "You are a helpful math assistant. Put your final answer in \\boxed{}."


def main():
    parser = argparse.ArgumentParser(
        description="Convert AIME data to C2C training format."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src = load_dataset("gneubig/aime-1983-2024", split="train")
    if args.limit:
        src = src.select(range(min(args.limit, len(src))))

    tools_json = json.dumps(PROMPT_TOOL)
    rows = []
    for row in src:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Question"]},
            {"role": "assistant", "content": f"\\boxed{{{row['Answer']}}}"},
        ]
        rows.append({"messages": json.dumps(messages), "tools": tools_json})

    hf_ds = Dataset.from_list(rows)
    hf_ds.save_to_disk(args.output)
    print(
        f"Saved {len(hf_ds)} rows to {args.output}\n"
        f"Columns: {hf_ds.column_names}"
    )


if __name__ == "__main__":
    main()
