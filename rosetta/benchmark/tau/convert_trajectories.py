"""Convert tau-bench trajectories into the C2C cache-optimize training format.

Reads trajectory JSONL files (with columns ``messages``, ``tools``,
``task_id``, ``trial``, ``actions``) and produces an HF Dataset with
columns ``[messages, tools, round, total_rounds]`` — the format expected
by ``cache_optimize_training.py``.

Unlike APIGen conversion, no format bridging is needed: tau trajectories
already use OpenAI message/tool schemas.

Usage::

    python -m rosetta.benchmark.tau.convert_trajectories \
        --trajectory local/trajectory/tau/test/kimi_k2p5/airline_trajectories.jsonl \
        --output local/datasets/tau_kimi_k2p5_airline
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from rosetta.optimize.dataset import split_multi_turn
from pathlib import Path

from rosetta.workflow.camel_utils import read_jsonl


def convert_trajectories(
    trajectory_path: str,
    records_path: Optional[str] = None,
    min_reward: Optional[float] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Load and convert tau-bench trajectories.

    Args:
        trajectory_path: Path to trajectory JSONL file.
        records_path: Optional path to records JSONL for reward filtering.
        min_reward: Minimum reward to keep a trajectory (requires records_path).

    Returns:
        (hf_dataset, stats) tuple.
    """
    trajectories = read_jsonl(Path(trajectory_path))

    # Optional reward filtering
    if records_path and min_reward is not None:
        records = read_jsonl(Path(records_path))
        reward_map = {(r["task_id"], r["trial"]): r.get("reward", 0) for r in records}
        trajectories = [
            t for t in trajectories
            if reward_map.get((t["task_id"], t["trial"]), 0) >= min_reward
        ]

    results: List[Dict[str, Any]] = []
    msg_counts: List[int] = []

    for traj in trajectories:
        messages = traj["messages"]
        tools = traj["tools"]
        msg_counts.append(len(messages))

        for split in split_multi_turn(messages, tools):
            results.append({
                "messages": json.dumps(split["messages"]),
                "tools": json.dumps(split["tools"]),
                "round": split["round"],
                "total_rounds": split["total_rounds"],
            })

    hf_ds = Dataset.from_list(results)

    stats = {
        "total_trajectories": len(trajectories),
        "total_rows": len(results),
        "avg_messages": sum(msg_counts) / len(msg_counts) if msg_counts else 0,
        "num_tools": len(trajectories[0]["tools"]) if trajectories else 0,
    }
    return hf_ds, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert tau-bench trajectories to C2C training format."
    )
    parser.add_argument(
        "--trajectory", required=True,
        help="Path to trajectory JSONL file.",
    )
    parser.add_argument("--output", default="local/datasets/tau_trajectories")
    parser.add_argument("--records", default=None, help="Records JSONL for reward filtering.")
    parser.add_argument("--min-reward", type=float, default=None)
    args = parser.parse_args()

    hf_ds, stats = convert_trajectories(
        args.trajectory, args.records, args.min_reward,
    )

    hf_ds.save_to_disk(args.output)

    print(f"Saved {stats['total_rows']} rows from {stats['total_trajectories']} trajectories to {args.output}")
    print(f"Columns: {hf_ds.column_names}")
    print(f"Tools per trajectory: {stats['num_tools']}")
    print(f"Avg messages per trajectory: {stats['avg_messages']:.1f}")

    if len(hf_ds) > 0:
        sample = json.loads(hf_ds[0]["messages"])
        print(f"\nSample message roles: {[m['role'] for m in sample]}")


if __name__ == "__main__":
    main()
