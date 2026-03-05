"""Convert DeepMath-103K into C2C training format.

Reads from ``zwhe99/DeepMath-103K`` on Hugging Face, produces an HF dataset
with columns [messages, tools].  Each row stores the question as a user turn
and the ground-truth answer in a short assistant turn (for reward extraction).
No tools are included (pure math reasoning).

Usage::

    python script/optimize/data/dataset_deepmath.py \
        --output local/datasets/deepmath/train
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = "You are a helpful math assistant. Put your final answer in \\boxed{}."


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepMath-103K to C2C training format."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src = load_dataset("zwhe99/DeepMath-103K", split="train")
    if args.limit:
        src = src.select(range(min(args.limit, len(src))))

    rows = []
    for row in src:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": f"\\boxed{{{row['final_answer']}}}"},
        ]
        rows.append({"messages": json.dumps(messages), "tools": "null"})

    hf_ds = Dataset.from_list(rows)
    hf_ds.save_to_disk(args.output)
    print(
        f"Saved {len(hf_ds)} rows to {args.output}\n"
        f"Columns: {hf_ds.column_names}"
    )


if __name__ == "__main__":
    main()
