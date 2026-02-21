"""Analyze tool distribution in tau-bench data.

Supports two input formats:
  --dataset     HF dataset on disk (from dataset_tau.py)
  --trajectory  Raw tau-bench trajectory JSON (e.g. gpt-4o-airline.json)

Produces four plots:
  1. Tool frequency        — per-trajectory, which tools appear
  2. Tool count            — per-trajectory, how many tools
  3. Tool combinations     — per-trajectory, top 15 tool sets
  4. Tool count per round  — per-round, how many new tools called

Usage::

    python script/analysis/optimize/tool_distribution.py \
        --dataset local/datasets/tau/dynamic/apigen_airline \
        --save-dir local/analysis/apigen_airline

    python script/analysis/optimize/tool_distribution.py \
        --trajectory local/analysis/gpt-4o-airline.json \
        --save-dir local/analysis/gpt4o_airline
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _tool_calls_in(messages: list[dict]) -> set[str]:
    """Extract distinct tool names called in a message list."""
    return {
        tc["function"]["name"]
        for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
    }


def load_from_hf(path: str) -> tuple[list[set[str]], Counter, int]:
    """Load HF dataset. Returns (tool_sets_per_traj, per_round_counts, num_rounds)."""
    from datasets import load_from_disk
    ds = load_from_disk(path)

    tool_sets: list[set[str]] = []
    per_round = Counter()
    i, n = 0, len(ds)

    while i < n:
        total = ds[i]["total_rounds"]
        # Last round has all cumulative messages
        last = min(i + total - 1, n - 1)
        tools = json.loads(ds[last]["tools"])
        tool_sets.append({t["function"]["name"] for t in tools})

        # Per-round: new tool calls in each round
        prev = set()
        for j in range(i, min(i + total, n)):
            cur = _tool_calls_in(json.loads(ds[j]["messages"]))
            per_round[len(cur - prev)] += 1
            prev = cur
        i += total

    return tool_sets, per_round, len(ds)


def _load_trajectory_items(path: str) -> list[dict]:
    """Load trajectory data from JSON (list) or JSONL (one object per line)."""
    with open(path) as f:
        first = f.read(1)
    with open(path) as f:
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _get_messages(item: dict) -> list[dict]:
    """Extract message list from a trajectory item (supports 'traj' and 'messages' keys)."""
    return item.get("traj") or item.get("messages") or []


def load_from_trajectory(path: str) -> tuple[list[set[str]], Counter, int]:
    """Load trajectory JSON/JSONL. Returns (tool_sets_per_traj, per_round_counts, num_rounds).

    Tool set per trajectory = tools actually called (from messages), NOT the
    environment tool definitions, since those are the same full set for every task.
    """
    data = _load_trajectory_items(path)

    tool_sets: list[set[str]] = []
    per_round = Counter()
    total_rounds = 0

    for item in data:
        msgs = _get_messages(item)
        tool_sets.append(_tool_calls_in(msgs))

        # Split by user turns for per-round analysis
        user_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        prev = set()
        for k, ui in enumerate(user_idxs):
            end = user_idxs[k + 1] if k + 1 < len(user_idxs) else len(msgs)
            cur = _tool_calls_in(msgs[:end])
            per_round[len(cur - prev)] += 1
            prev = cur
        total_rounds += len(user_idxs) or 1

    return tool_sets, per_round, total_rounds


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    num_traj: int
    num_rounds: int
    tool_freq: Counter = field(default_factory=Counter)     # tool -> #traj
    tool_count: Counter = field(default_factory=Counter)     # #tools -> #traj
    tool_combo: Counter = field(default_factory=Counter)     # (tool,...) -> #traj
    round_tool_count: Counter = field(default_factory=Counter)  # #tools -> #rounds


def compute_stats(tool_sets: list[set[str]], per_round: Counter, num_rounds: int) -> Stats:
    s = Stats(num_traj=len(tool_sets), num_rounds=num_rounds, round_tool_count=per_round)
    for ts in tool_sets:
        for name in ts:
            s.tool_freq[name] += 1
        s.tool_count[len(ts)] += 1
        s.tool_combo[tuple(sorted(ts))] += 1
    return s


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_stats(s: Stats):
    print(f"Trajectories: {s.num_traj}\n")

    # Tool frequency
    print("=" * 60)
    print("Tool Frequency (per trajectory)")
    print(f"{'Tool':<45} {'Count':>7} {'%':>7}")
    print("-" * 60)
    for name, cnt in s.tool_freq.most_common():
        print(f"{name:<45} {cnt:>7} {cnt / s.num_traj * 100:>6.1f}%")
    print(f"Unique tools: {len(s.tool_freq)}")

    # Tool count per trajectory
    print(f"\n{'=' * 40}\nTool Count (per trajectory)")
    print(f"{'#Tools':>7} {'#Traj':>7} {'%':>7}\n{'-' * 30}")
    for k in sorted(s.tool_count):
        n = s.tool_count[k]
        print(f"{k:>7} {n:>7} {n / s.num_traj * 100:>6.1f}%")

    # Top 15 combos
    print(f"\n{'=' * 70}\nTop 15 Tool Combinations ({len(s.tool_combo)} unique)")
    print(f"{'#':>3} {'Count':>6} {'%':>6}  Tools\n{'-' * 70}")
    for rank, (combo, cnt) in enumerate(s.tool_combo.most_common(15), 1):
        label = " + ".join(combo) if combo else "(none)"
        print(f"{rank:>3} {cnt:>6} {cnt / s.num_traj * 100:>5.1f}%  {label}")

    # Per-round
    print(f"\n{'=' * 40}\nTool Count (per round, N={s.num_rounds})")
    print(f"{'#Tools':>7} {'#Rounds':>7} {'%':>7}\n{'-' * 30}")
    for k in sorted(s.round_tool_count):
        n = s.round_tool_count[k]
        print(f"{k:>7} {n:>7} {n / s.num_rounds * 100:>6.1f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_labels(ax, bars, fmt=int):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    str(fmt(h)), ha="center", va="bottom", fontsize=8)


def _hbar_pct_labels(ax, y, vals, total):
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + total * 0.01, y[i],
                    f"{v / total * 100:.1f}%", va="center", fontsize=8)


def _counter_bar(ax, counter, xlabel, ylabel, title):
    keys = sorted(counter)
    vals = [counter[k] for k in keys]
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, color="#3b82f6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _bar_labels(ax, bars)


def plot_stats(s: Stats, save_dir: str | None = None):
    COLOR = "#3b82f6"
    figs = {}

    # 1. Per-trajectory: which tools appear
    tools = sorted(s.tool_freq, key=lambda t: s.tool_freq[t])
    vals = [s.tool_freq[t] for t in tools]
    fig, ax = plt.subplots(figsize=(10, max(4, len(tools) * 0.4)))
    y = np.arange(len(tools))
    ax.barh(y, vals, color=COLOR, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(tools)
    ax.set_xlabel("Number of Trajectories")
    ax.set_title(f"Per-Trajectory Tool Frequency (N={s.num_traj})")
    _hbar_pct_labels(ax, y, vals, s.num_traj)
    fig.tight_layout()
    figs["1_tool_frequency"] = fig

    # 2. Per-trajectory: how many distinct tools
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _counter_bar(ax, s.tool_count, "# Distinct Tools", "# Trajectories",
                 f"Per-Trajectory Tool Count (N={s.num_traj})")
    fig.tight_layout()
    figs["2_tool_count_per_trajectory"] = fig

    # 3. Per-trajectory: top 15 tool set combinations
    top = s.tool_combo.most_common(15)
    labels = [" + ".join(c) if c else "(none)" for c, _ in reversed(top)]
    vals = [v for _, v in reversed(top)]
    max_label = max((len(l) for l in labels), default=0)
    fig_w = max(10, min(18, 4 + max_label * 0.08))
    fig, ax = plt.subplots(figsize=(fig_w, max(4, len(labels) * 0.45)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=COLOR, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of Trajectories")
    ax.set_title(f"Per-Trajectory Top 15 Tool Combinations (N={s.num_traj}, {len(s.tool_combo)} unique)")
    for i, v in enumerate(vals):
        ax.text(v + s.num_traj * 0.005, y[i],
                f"{v} ({v / s.num_traj * 100:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    figs["3_tool_combinations"] = fig

    # 4. Per-round: how many new tools called
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _counter_bar(ax, s.round_tool_count, "# New Tools Called", "# Rounds",
                 f"Per-Round Tool Count (N={s.num_rounds})")
    fig.tight_layout()
    figs["4_tool_count_per_round"] = fig

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150)
            plt.close(fig)
        print(f"\nPlots saved to {save_dir}/")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze tool distribution in tau-bench data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="HF dataset on disk (from dataset_tau.py).")
    group.add_argument("--trajectory", help="Raw tau-bench trajectory JSON.")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots.")
    args = parser.parse_args()

    if args.dataset:
        print(f"Dataset: {args.dataset}")
        tool_sets, per_round, num_rounds = load_from_hf(args.dataset)
    else:
        print(f"Trajectory: {args.trajectory}")
        tool_sets, per_round, num_rounds = load_from_trajectory(args.trajectory)

    print(f"Rounds: {num_rounds}")
    s = compute_stats(tool_sets, per_round, num_rounds)
    print_stats(s)
    plot_stats(s, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
