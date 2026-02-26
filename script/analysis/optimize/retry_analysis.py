"""Analyze retry behavior and trajectory quality across tau-bench experiments.

Compares multiple trajectory files on:
  1. Success rate
  2. Tool call counts and distribution by success
  3. Retry loops (same tool+args called consecutively)
  4. Action tool error rates
  5. Hallucinated (non-existent) tool calls
  6. Per-tool frequency comparison table
  7. Per-round tool call distribution (which tool at round 1, 2, 3)

Also supports an HF dataset as training-data reference (``--dataset``).
Each dataset row is one *round*; rows are grouped back into trajectories
via ``total_rounds``.

Usage::

    python script/analysis/optimize/retry_analysis.py \
        --trajectory baseline=local/trajectory/tau/mixed/baseline/retail_trajectories.jsonl \
        --records    baseline=local/trajectory/tau/mixed/baseline/retail.jsonl \
        --trajectory lora=local/trajectory/tau/mixed/lora/retail_trajectories.jsonl \
        --records    lora=local/trajectory/tau/mixed/lora/retail.jsonl \
        --trajectory sft=local/trajectory/tau/mixed/sft/retail_trajectories.jsonl \
        --records    sft=local/trajectory/tau/mixed/sft/retail.jsonl \
        --trajectory optCache_freeze=local/trajectory/tau/mixed/optCache_freeze/retail_trajectories.jsonl \
        --records    optCache_freeze=local/trajectory/tau/mixed/optCache_freeze/retail.jsonl \
        --dataset    apigen=local/datasets/tau/dynamic/apigen_all \
        --dataset-domain retail \
        --reference  glm5=local/trajectory/tau/glm-5/retail_trajectories.jsonl \
        --save-dir   local/analysis/retry_comparison
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


# -- Canonical tool sets for hallucination detection --------------------------

AIRLINE_TOOLS = {
    "book_reservation", "calculate", "cancel_reservation",
    "get_reservation_details", "get_user_details", "list_all_airports",
    "search_direct_flight", "search_onestop_flight", "send_certificate",
    "think", "transfer_to_human_agents", "update_reservation_baggages",
    "update_reservation_flights", "update_reservation_passengers",
}

RETAIL_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items", "calculate",
    "find_user_id_by_email", "find_user_id_by_name_zip",
    "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment",
    "modify_user_address", "return_delivered_order_items",
    "think", "transfer_to_human_agents",
}

VALID_TOOLS = AIRLINE_TOOLS | RETAIL_TOOLS

ACTION_TOOLS = {
    "exchange_delivered_order_items", "return_delivered_order_items",
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
    "modify_user_address", "cancel_reservation", "update_reservation_baggages",
    "update_reservation_flights", "update_reservation_passengers",
    "book_reservation", "send_certificate",
}


# -- Data loading -------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        first = f.read(1)
    with open(path) as f:
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _get_msgs(item: dict) -> list[dict]:
    return item.get("traj") or item.get("messages") or []


def load_trajectories(path: str) -> list[dict]:
    return _load_jsonl(path)


def load_records(path: str) -> dict[tuple, float]:
    """Return {(task_id, trial): reward}."""
    items = _load_jsonl(path)
    return {(r["task_id"], r.get("trial", 0)): r.get("reward", 0) for r in items}


def load_dataset_as_trajectories(
    path: str, domain: str | None = None,
) -> list[dict]:
    """Load HF dataset rows, group by trajectory, return pseudo-trajectory dicts.

    Each dataset row is a single *round*; consecutive rows with the same
    ``total_rounds`` starting index belong to the same trajectory.  We use
    the last round's messages (which contain all cumulative messages) as
    the trajectory.
    """
    from datasets import load_from_disk

    ds = load_from_disk(path)
    if domain and "domain" in ds.column_names:
        ds = ds.filter(lambda x: x["domain"] == domain)

    trajs = []
    i, n = 0, len(ds)
    while i < n:
        total = ds[i]["total_rounds"]
        last = min(i + total - 1, n - 1)
        msgs = json.loads(ds[last]["messages"])
        tools = json.loads(ds[last]["tools"])
        trajs.append({"messages": msgs, "tools": tools, "_source": "dataset"})
        i += total
    return trajs


# -- Per-trajectory analysis --------------------------------------------------

@dataclass
class TrajStats:
    tool_calls: list[str] = field(default_factory=list)
    num_tool_calls: int = 0
    num_messages: int = 0
    num_user_turns: int = 0
    retry_same_count: int = 0
    hallucinated: list[str] = field(default_factory=list)
    action_calls: int = 0
    action_errors: int = 0
    tool_error_count: int = 0
    total_tool_responses: int = 0
    reward: float = 0.0


def analyze_trajectory(msgs: list[dict]) -> TrajStats:
    s = TrajStats()
    s.num_messages = len(msgs)
    s.num_user_turns = sum(1 for m in msgs if m.get("role") == "user")

    prev_sig = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"].get("arguments", "")
                s.tool_calls.append(name)
                s.num_tool_calls += 1

                if name not in VALID_TOOLS:
                    s.hallucinated.append(name)

                if name in ACTION_TOOLS:
                    s.action_calls += 1
                    # Check next tool response for error
                    for j in range(i + 1, len(msgs)):
                        if msgs[j].get("role") == "tool":
                            content = str(msgs[j].get("content", "")).lower()
                            if "error" in content:
                                s.action_errors += 1
                            break

                sig = (name, args)
                if sig == prev_sig:
                    s.retry_same_count += 1
                prev_sig = sig
        elif m.get("role") == "tool":
            s.total_tool_responses += 1
            content = str(m.get("content", "")).lower()
            if "error" in content or "not found" in content or "invalid" in content:
                s.tool_error_count += 1
        else:
            prev_sig = None

    return s


# -- Aggregate statistics -----------------------------------------------------

@dataclass
class ExperimentStats:
    label: str
    num_traj: int = 0
    rewards: list[float] = field(default_factory=list)
    traj_stats: list[TrajStats] = field(default_factory=list)
    tool_freq: Counter = field(default_factory=Counter)

    @property
    def success_rate(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(1 for r in self.rewards if r >= 1) / len(self.rewards)

    @property
    def avg_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def tc_counts(self) -> list[int]:
        return [s.num_tool_calls for s in self.traj_stats]

    def unique_tool_counts(self) -> list[int]:
        return [len(set(s.tool_calls)) for s in self.traj_stats]

    def retries(self) -> int:
        return sum(s.retry_same_count for s in self.traj_stats)

    def trajs_with_retries(self) -> int:
        return sum(1 for s in self.traj_stats if s.retry_same_count > 0)

    def trajs_with_hallucination(self) -> int:
        return sum(1 for s in self.traj_stats if s.hallucinated)

    def hallucination_counter(self) -> Counter:
        c = Counter()
        for s in self.traj_stats:
            c.update(s.hallucinated)
        return c

    def action_error_rate(self) -> float:
        total = sum(s.action_calls for s in self.traj_stats)
        errors = sum(s.action_errors for s in self.traj_stats)
        return errors / total if total else 0.0

    def tool_error_rate(self) -> float:
        total = sum(s.total_tool_responses for s in self.traj_stats)
        errors = sum(s.tool_error_count for s in self.traj_stats)
        return errors / total if total else 0.0

    def round_tool_freq(self, n: int) -> Counter:
        """Count tool names at the *n*-th tool call (1-indexed) across trajs."""
        c: Counter = Counter()
        for s in self.traj_stats:
            if len(s.tool_calls) >= n:
                c[s.tool_calls[n - 1]] += 1
        return c

    def success_by_tc_bin(self, bins=None):
        if bins is None:
            bins = [(0, 5), (6, 10), (11, 999)]
        results = []
        for lo, hi in bins:
            in_bin = [
                (s, r)
                for s, r in zip(self.traj_stats, self.rewards)
                if lo <= s.num_tool_calls <= hi
            ]
            n = len(in_bin)
            succ = sum(1 for _, r in in_bin if r >= 1)
            results.append((lo, hi, n, succ))
        return results


def build_experiment(
    label: str,
    trajs: list[dict],
    reward_map: dict[tuple, float] | None = None,
) -> ExperimentStats:
    exp = ExperimentStats(label=label, num_traj=len(trajs))
    for t in trajs:
        msgs = _get_msgs(t)
        ts = analyze_trajectory(msgs)
        # Tools called (distinct per trajectory)
        called = set(ts.tool_calls)
        exp.tool_freq.update(called)

        if reward_map is not None:
            key = (t.get("task_id"), t.get("trial", 0))
            ts.reward = reward_map.get(key, 0.0)
            exp.rewards.append(ts.reward)

        exp.traj_stats.append(ts)
    return exp


def build_dataset_experiment(
    label: str, trajs: list[dict],
) -> ExperimentStats:
    """Build stats for training dataset (no rewards)."""
    exp = ExperimentStats(label=label, num_traj=len(trajs))
    for t in trajs:
        msgs = _get_msgs(t)
        ts = analyze_trajectory(msgs)
        called = set(ts.tool_calls)
        exp.tool_freq.update(called)
        exp.traj_stats.append(ts)
    return exp


# -- Printing -----------------------------------------------------------------

def print_summary(experiments: list[ExperimentStats]):
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Header
    labels = [e.label for e in experiments]
    header = f"{'Metric':<35}" + "".join(f"{l:>13}" for l in labels)
    print(header)
    print("-" * len(header))

    # Rows
    def row(name, values, fmt="{:>13}"):
        print(f"{name:<35}" + "".join(fmt.format(v) for v in values))

    has_rewards = any(e.rewards for e in experiments)
    if has_rewards:
        row("Trajectories", [e.num_traj for e in experiments])
        row("Success rate",
            [f"{e.success_rate*100:.1f}%" if e.rewards else "n/a" for e in experiments])
        row("Avg reward",
            [f"{e.avg_reward:.4f}" if e.rewards else "n/a" for e in experiments])

    tc = [e.tc_counts() for e in experiments]
    row("Avg tool calls/traj",
        [f"{sum(c)/len(c):.1f}" if c else "n/a" for c in tc])
    row("Median tool calls/traj",
        [f"{sorted(c)[len(c)//2]}" if c else "n/a" for c in tc])
    row("Trajs >15 tool calls",
        [f"{sum(1 for x in c if x > 15)} ({sum(1 for x in c if x > 15)/len(c)*100:.1f}%)"
         if c else "n/a" for c in tc])

    row("Same-call retries (total)",
        [e.retries() for e in experiments])
    row("Trajs with retries",
        [f"{e.trajs_with_retries()} ({e.trajs_with_retries()/e.num_traj*100:.1f}%)"
         for e in experiments])
    row("Tool error rate",
        [f"{e.tool_error_rate()*100:.1f}%" for e in experiments])
    row("Action tool error rate",
        [f"{e.action_error_rate()*100:.1f}%" for e in experiments])
    row("Trajs w/ hallucinated tools",
        [f"{e.trajs_with_hallucination()} ({e.trajs_with_hallucination()/e.num_traj*100:.1f}%)"
         for e in experiments])

    # Success by tool-call-count bin
    if has_rewards:
        print(f"\n{'Success rate by tool-call count':}")
        bins = [(0, 5), (6, 10), (11, 999)]
        for lo, hi in bins:
            hi_label = f"{hi}" if hi < 999 else "+"
            name = f"  {lo}-{hi_label} tool calls"
            vals = []
            for e in experiments:
                if not e.rewards:
                    vals.append("n/a")
                    continue
                rows = e.success_by_tc_bin([(lo, hi)])
                _, _, n, succ = rows[0]
                if n:
                    vals.append(f"{succ}/{n} ({succ/n*100:.1f}%)")
                else:
                    vals.append("0/0")
            row(name, vals)


def print_tool_frequency(experiments: list[ExperimentStats]):
    print(f"\n{'=' * 100}")
    print("TOOL FREQUENCY (% of trajectories)")
    print("=" * 100)

    # Collect all tools, sort by baseline (or first experiment with rewards)
    ref = next((e for e in experiments if e.label == "baseline"),
               next((e for e in experiments if e.rewards), experiments[0]))
    all_tools = set()
    for e in experiments:
        all_tools |= set(e.tool_freq.keys())
    all_tools = sorted(all_tools, key=lambda t: ref.tool_freq.get(t, 0), reverse=True)

    labels = [e.label for e in experiments]
    header = f"{'Tool':<35}" + "".join(f"{l:>13}" for l in labels)
    print(header)
    print("-" * len(header))
    for t in all_tools:
        vals = []
        for e in experiments:
            cnt = e.tool_freq.get(t, 0)
            vals.append(f"{cnt/e.num_traj*100:.1f}%")
        print(f"{t:<35}" + "".join(f"{v:>13}" for v in vals))


def print_hallucinations(experiments: list[ExperimentStats]):
    print(f"\n{'=' * 80}")
    print("HALLUCINATED TOOLS (top 5 per experiment)")
    print("=" * 80)
    for e in experiments:
        hc = e.hallucination_counter()
        if not hc:
            print(f"  {e.label}: none")
            continue
        top = hc.most_common(5)
        items = ", ".join(f"{name}({cnt})" for name, cnt in top)
        print(f"  {e.label}: {items}")


def print_round_tool_frequency(experiments: list[ExperimentStats], rounds: list[int] = [1, 2, 3]):
    for n in rounds:
        print(f"\n{'=' * 100}")
        print(f"ROUND {n} TOOL CALL (% of trajectories with >= {n} tool calls)")
        print("=" * 100)

        freqs = [e.round_tool_freq(n) for e in experiments]
        eligible = [sum(1 for s in e.traj_stats if len(s.tool_calls) >= n) for e in experiments]

        # Collect all tools at this round, sort by baseline
        ref_idx = next((i for i, e in enumerate(experiments) if e.label == "baseline"),
                       next((i for i, e in enumerate(experiments) if e.rewards), 0))
        all_tools = set()
        for f in freqs:
            all_tools |= set(f.keys())
        all_tools = sorted(all_tools, key=lambda t: freqs[ref_idx].get(t, 0), reverse=True)

        labels = [e.label for e in experiments]
        header = f"{'Tool':<35}" + "".join(f"{l:>13}" for l in labels)
        print(header)
        elig_row = f"{'(eligible trajs)':<35}" + "".join(f"{el:>13}" for el in eligible)
        print(elig_row)
        print("-" * len(header))
        for t in all_tools:
            vals = []
            for f, el in zip(freqs, eligible):
                cnt = f.get(t, 0)
                pct = cnt / el * 100 if el else 0
                vals.append(f"{pct:.1f}%")
            print(f"{t:<35}" + "".join(f"{v:>13}" for v in vals))


# -- Plotting -----------------------------------------------------------------

def plot_comparison(experiments: list[ExperimentStats], save_dir: str | None = None):
    figs = {}
    # Only plot experiments with rewards
    eval_exps = [e for e in experiments if e.rewards]
    all_exps = experiments
    colors = plt.cm.tab10.colors

    # 1. Success rate bar chart
    if eval_exps:
        fig, ax = plt.subplots(figsize=(max(5, len(eval_exps) * 1.2), 4))
        x = np.arange(len(eval_exps))
        vals = [e.success_rate * 100 for e in eval_exps]
        bars = ax.bar(x, vals, color=colors[:len(eval_exps)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([e.label for e in eval_exps], rotation=30, ha="right")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate Comparison")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        figs["1_success_rate"] = fig

    # 2. Tool call count distribution (box plot)
    fig, ax = plt.subplots(figsize=(max(5, len(all_exps) * 1.2), 4))
    data = [e.tc_counts() for e in all_exps]
    bp = ax.boxplot(data, labels=[e.label for e in all_exps],
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Tool Calls per Trajectory")
    ax.set_title("Tool Call Count Distribution")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    figs["2_tool_call_distribution"] = fig

    # 3. Retry comparison
    fig, ax = plt.subplots(figsize=(max(5, len(all_exps) * 1.2), 4))
    x = np.arange(len(all_exps))
    vals = [e.trajs_with_retries() / e.num_traj * 100 for e in all_exps]
    bars = ax.bar(x, vals, color=colors[:len(all_exps)], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([e.label for e in all_exps], rotation=30, ha="right")
    ax.set_ylabel("% Trajectories with Retry Loops")
    ax.set_title("Same-Call Retry Rate")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    figs["3_retry_rate"] = fig

    # 4. Success by tool-call bin (grouped bar)
    if eval_exps:
        bins = [(0, 5), (6, 10), (11, 999)]
        bin_labels = ["0-5", "6-10", "10+"]
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.8 / len(eval_exps)
        x = np.arange(len(bins))
        for idx, e in enumerate(eval_exps):
            bin_data = e.success_by_tc_bin(bins)
            rates = [succ / n * 100 if n else 0 for _, _, n, succ in bin_data]
            counts = [n for _, _, n, _ in bin_data]
            offset = (idx - len(eval_exps) / 2 + 0.5) * width
            bars = ax.bar(x + offset, rates, width, label=e.label,
                          color=colors[idx], edgecolor="white", alpha=0.85)
            for bar, r, c in zip(bars, rates, counts):
                if c > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"n={c}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Tool Calls per Trajectory")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Tool Call Count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figs["4_success_by_tc_bin"] = fig

    # 5. Tool frequency heatmap
    ref = next((e for e in experiments if e.label == "baseline"),
               next((e for e in experiments if e.rewards), experiments[0]))
    all_tools_set = set()
    for e in experiments:
        all_tools_set |= set(e.tool_freq.keys())
    # Filter to canonical tools only
    all_tools_set &= VALID_TOOLS
    tools_sorted = sorted(all_tools_set, key=lambda t: ref.tool_freq.get(t, 0), reverse=True)

    if tools_sorted:
        matrix = np.zeros((len(tools_sorted), len(experiments)))
        for j, e in enumerate(experiments):
            for i, t in enumerate(tools_sorted):
                matrix[i, j] = e.tool_freq.get(t, 0) / e.num_traj * 100

        fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.5), max(5, len(tools_sorted) * 0.4)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels([e.label for e in experiments], rotation=30, ha="right")
        ax.set_yticks(range(len(tools_sorted)))
        ax.set_yticklabels(tools_sorted, fontsize=8)
        for i in range(len(tools_sorted)):
            for j in range(len(experiments)):
                v = matrix[i, j]
                color = "white" if v > 50 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=color)
        ax.set_title("Tool Frequency (% of trajectories)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        figs["5_tool_frequency_heatmap"] = fig

    # 6. Per-round tool call heatmap (rounds 1, 2, 3 as subplots)
    rounds = [1, 2, 3]
    # Pre-compute freqs and eligible for each round
    round_freqs = {}
    round_eligible = {}
    for n in rounds:
        round_freqs[n] = [e.round_tool_freq(n) for e in experiments]
        round_eligible[n] = [
            sum(1 for s in e.traj_stats if len(s.tool_calls) >= n)
            for e in experiments
        ]
    # Build unified tool list across all rounds
    all_round_tools = set()
    for n in rounds:
        for f in round_freqs[n]:
            all_round_tools |= set(f.keys())
    # Keep canonical + any non-canonical with >= 2% in any round/experiment
    keep = set()
    for t in all_round_tools:
        if t in VALID_TOOLS:
            keep.add(t)
        else:
            for n in rounds:
                for f, el in zip(round_freqs[n], round_eligible[n]):
                    if el and f.get(t, 0) / el >= 0.02:
                        keep.add(t)
                        break
    # Sort by baseline's total frequency across all rounds
    baseline_idx = next(
        (i for i, e in enumerate(experiments) if e.label == "baseline"),
        next((i for i, e in enumerate(experiments) if e.rewards), 0),
    )

    def _tool_sort_key(t):
        total = 0
        for n in rounds:
            el = round_eligible[n][baseline_idx]
            total += round_freqs[n][baseline_idx].get(t, 0) if el else 0
        return total
    tools_sorted = sorted(keep, key=_tool_sort_key, reverse=True)
    # Drop tools with 0 across all rounds and experiments
    tools_sorted = [
        t for t in tools_sorted
        if any(round_freqs[n][j].get(t, 0)
               for n in rounds for j in range(len(experiments)))
    ]

    if tools_sorted:
        n_exp = len(experiments)
        n_rounds = len(rounds)
        n_tools = len(tools_sorted)
        panel_w = max(5, n_exp * 1.1)
        fig_h = max(7, n_tools * 0.5 + 2)
        fig, axes = plt.subplots(
            1, n_rounds,
            figsize=(panel_w * n_rounds + 3, fig_h),
            sharey=True,
        )
        vmax = 0
        matrices = []
        for ri, n in enumerate(rounds):
            mat = np.zeros((n_tools, n_exp))
            for j, (f, el) in enumerate(zip(round_freqs[n], round_eligible[n])):
                for i, t in enumerate(tools_sorted):
                    mat[i, j] = f.get(t, 0) / el * 100 if el else 0
            matrices.append(mat)
            vmax = max(vmax, mat.max())

        for ri, (n, mat, ax) in enumerate(zip(rounds, matrices, axes)):
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(n_exp))
            ax.set_xticklabels([e.label for e in experiments], rotation=30, ha="right")
            if ri == 0:
                ax.set_yticks(range(n_tools))
                ax.set_yticklabels(tools_sorted, fontsize=10)
            ax.set_title(f"Round {n}", fontsize=12)
            for i in range(n_tools):
                for j in range(n_exp):
                    v = mat[i, j]
                    color = "white" if v > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=9, color=color)

        fig.suptitle("Per-Round Tool Call Distribution", fontsize=13)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                     label="% of eligible trajectories")
        figs["6_round_tool_call"] = fig

    # 7. Unique tools per trajectory (box plot)
    fig, ax = plt.subplots(figsize=(max(5, len(all_exps) * 1.2), 4))
    data = [e.unique_tool_counts() for e in all_exps]
    bp = ax.boxplot(data, tick_labels=[e.label for e in all_exps],
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Unique Tools per Trajectory")
    ax.set_title("Unique Tool Count Distribution")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    figs["7_unique_tool_count"] = fig

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=200)
            plt.close(fig)
        print(f"\nPlots saved to {save_dir}/")
    else:
        plt.show()


# -- CLI ----------------------------------------------------------------------

def _parse_kv(value: str) -> tuple[str, str]:
    """Parse 'label=path' argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected label=path, got: {value}")
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retry behavior and trajectory quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectory", action="append", default=[], type=_parse_kv, metavar="LABEL=PATH",
        help="Trajectory JSONL to analyze (repeatable).",
    )
    parser.add_argument(
        "--records", action="append", default=[], type=_parse_kv, metavar="LABEL=PATH",
        help="Records JSONL with rewards, matched by label (repeatable).",
    )
    parser.add_argument(
        "--reference", action="append", default=[], type=_parse_kv, metavar="LABEL=PATH",
        help="Reference trajectory (no records needed, e.g. glm-5).",
    )
    parser.add_argument(
        "--dataset", action="append", default=[], type=_parse_kv, metavar="LABEL=PATH",
        help="HF dataset on disk as training-data reference (repeatable).",
    )
    parser.add_argument("--dataset-domain", default=None,
                        help="Filter dataset by domain column (e.g. 'retail').")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save plots.")
    args = parser.parse_args()

    if not args.trajectory and not args.reference and not args.dataset:
        parser.error("Provide at least one --trajectory, --reference, or --dataset.")

    # Save the command used to reproduce this run
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        import sys, shlex
        cmd = "python " + " ".join(shlex.quote(a) for a in sys.argv)
        with open(os.path.join(args.save_dir, "command.sh"), "w") as f:
            f.write(cmd + "\n")

    records_map = dict(args.records)
    experiments: list[ExperimentStats] = []

    # Training dataset reference
    for label, path in args.dataset:
        print(f"Loading dataset: {label} ({path})")
        trajs = load_dataset_as_trajectories(path, domain=args.dataset_domain)
        exp = build_dataset_experiment(label, trajs)
        experiments.append(exp)
        print(f"  {exp.num_traj} trajectories")

    # Reference trajectories (no rewards)
    for label, path in args.reference:
        print(f"Loading reference: {label} ({path})")
        trajs = load_trajectories(path)
        exp = build_experiment(label, trajs, reward_map=None)
        experiments.append(exp)
        print(f"  {exp.num_traj} trajectories")

    # Evaluated trajectories (with rewards)
    for label, path in args.trajectory:
        rec_path = records_map.get(label)
        reward_map = load_records(rec_path) if rec_path else None
        print(f"Loading trajectory: {label} ({path})")
        if rec_path:
            print(f"  Records: {rec_path}")
        trajs = load_trajectories(path)
        exp = build_experiment(label, trajs, reward_map=reward_map)
        experiments.append(exp)
        print(f"  {exp.num_traj} trajectories, "
              f"success={exp.success_rate*100:.1f}%" if exp.rewards else "")

    print()
    print_summary(experiments)
    print_tool_frequency(experiments)
    print_hallucinations(experiments)
    print_round_tool_frequency(experiments)
    plot_comparison(experiments, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
