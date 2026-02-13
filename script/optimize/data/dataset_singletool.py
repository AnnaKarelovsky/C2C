"""
Convert eval_singletool.py output to a HuggingFace Dataset.

Reads the records JSONL and trajectories JSONL, merges them by example_id,
and saves as a HuggingFace dataset in the same directory.

Usage:
    # From default output path
    python script/optimize/data/dataset_singletool.py \
        --output local/trajectory/singletool/browsecomp.jsonl

    # Load later:
    #   from datasets import load_from_disk
    #   ds = load_from_disk("local/trajectory/singletool/browsecomp_dataset")
    #   for item in ds:
    #       for msg in item["messages"]:
    #           print(msg["role"], msg["content"][:80])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from rosetta.optimize.dataset import split_multi_turn
from rosetta.workflow.camel_utils import read_jsonl


def build_dataset(output_path: Path) -> Dataset:
    """Merge records + trajectories into a single HuggingFace Dataset."""
    traj_path = output_path.parent / (output_path.stem + "_trajectories.jsonl")

    if not output_path.exists():
        raise FileNotFoundError(f"Records not found: {output_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectories not found: {traj_path}")

    records = {str(r["example_id"]): r for r in read_jsonl(output_path)}
    trajs = {str(t["example_id"]): t for t in read_jsonl(traj_path)}

    rows = []
    for eid, rec in records.items():
        traj = trajs.get(eid, {})
        messages = traj.get("messages") or []
        tools = traj.get("tools") or []
        base = {
            "example_id": eid,
            "idx": rec.get("idx"),
            "question": rec.get("question", ""),
            "gold_answer": rec.get("gold_answer", ""),
            "pred_answer": rec.get("pred_answer", ""),
            "correct_em": rec.get("correct_em", False),
            "correct_llm": rec.get("correct_llm"),
            "error_category": rec.get("error_category"),
            "seconds": rec.get("seconds"),
            "rounds": rec.get("rounds"),
            "error": rec.get("error"),
            "model_identity": traj.get("model_identity"),
            "usage": json.dumps(rec.get("usage")),
            "usage_per_interaction": json.dumps(
                traj.get("usage_per_interaction") or []
            ),
            "logprobs": json.dumps(traj.get("logprobs") or []),
        }
        for split in split_multi_turn(messages, tools):
            rows.append({
                **base,
                "messages": json.dumps(split["messages"]),
                "tools": json.dumps(split["tools"]),
                "round": split["round"],
                "total_rounds": split["total_rounds"],
            })

    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert eval_singletool output to HuggingFace Dataset"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the records JSONL (same as --output in eval_singletool.py)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Where to save the dataset (default: <output_stem>_dataset in same dir)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = output_path.parent / (output_path.stem + "_dataset")

    ds = build_dataset(output_path)
    ds.save_to_disk(str(dataset_dir))
    print(f"Saved {len(ds)} examples to {dataset_dir}")
    print(f"\nLoad with:\n  from datasets import load_from_disk")
    print(f"  ds = load_from_disk(\"{dataset_dir}\")")
    print(f"\nColumns: {ds.column_names}")


if __name__ == "__main__":
    main()
