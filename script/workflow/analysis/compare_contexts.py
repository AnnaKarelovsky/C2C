#!/usr/bin/env python3
"""
Compare metrics across different context configurations.

This script aligns metrics from multiple perplexity analysis runs by original_uid
and computes deltas to understand how context transformations affect model behavior.

Usage:
    # Compare two runs
    python script/workflow/analysis/compare_contexts.py \
        --runs baseline_metrics.csv summarized_metrics.csv \
        --output comparison.csv

    # Compare with specific metrics
    python script/workflow/analysis/compare_contexts.py \
        --runs baseline_metrics.csv summarized_metrics.csv \
        --metrics neg_log_prob perplexity \
        --output comparison.csv

    # With visualization
    python script/workflow/analysis/compare_contexts.py \
        --runs baseline_metrics.csv summarized_metrics.csv \
        --output comparison.csv \
        --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class TokenMetrics:
    """Metrics for a single token."""

    conversation_id: str
    token_idx: int
    section_idx: int
    message_uid: int
    original_uid: int
    transform_type: str
    role: str
    content_type: str
    metrics: Dict[str, float]

    @property
    def effective_role(self) -> str:
        """Get role with transform_type suffix for tools.

        Returns 'tool_original' or 'tool_summarized' for tool messages,
        otherwise returns the original role.
        """
        if self.role == "tool":
            return f"tool_{self.transform_type}"
        return self.role


@dataclass
class MessageMetrics:
    """Aggregated metrics for a message (by original_uid)."""

    conversation_id: str
    original_uid: int
    role: str
    token_count: int
    transform_type: str
    metrics: Dict[str, float]  # metric_name -> mean value


def load_metrics_csv(path: Path) -> List[TokenMetrics]:
    """Load token-level metrics from CSV.

    Args:
        path: Path to metrics CSV file.

    Returns:
        List of TokenMetrics objects.
    """
    results = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Parse core fields
            conv_id = row.get("conversation_id", "unknown")
            token_idx = int(row.get("token_idx", -1))
            section_idx = int(row.get("section_idx", -1))
            message_uid = int(row.get("message_uid", -1))
            original_uid = int(row.get("original_uid", message_uid))
            transform_type = row.get("transform_type", "original")
            role = row.get("role", "unknown")
            content_type = row.get("content_type", "unknown")

            # Extract metric columns
            metrics = {}
            skip_cols = {
                "conversation_id",
                "token_idx",
                "section_idx",
                "message_uid",
                "original_uid",
                "transform_type",
                "role",
                "content_type",
            }
            for key, value in row.items():
                if key not in skip_cols and value:
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        continue

            results.append(
                TokenMetrics(
                    conversation_id=conv_id,
                    token_idx=token_idx,
                    section_idx=section_idx,
                    message_uid=message_uid,
                    original_uid=original_uid,
                    transform_type=transform_type,
                    role=role,
                    content_type=content_type,
                    metrics=metrics,
                )
            )

    return results


def aggregate_by_message(tokens: List[TokenMetrics]) -> Dict[Tuple[str, int], MessageMetrics]:
    """Aggregate token metrics by (conversation_id, original_uid).

    Args:
        tokens: List of token-level metrics.

    Returns:
        Dict mapping (conversation_id, original_uid) to MessageMetrics.
    """
    # Group by (conversation_id, original_uid)
    groups: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        key = (token.conversation_id, token.original_uid)
        groups[key].append(token)

    results = {}
    for key, token_list in groups.items():
        conv_id, orig_uid = key

        # Aggregate metrics
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)

        role = token_list[0].role
        transform_type = token_list[0].transform_type

        for token in token_list:
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        # Compute means
        mean_metrics = {}
        for metric_name, total in metric_sums.items():
            count = metric_counts[metric_name]
            if count > 0:
                mean_metrics[metric_name] = total / count

        results[key] = MessageMetrics(
            conversation_id=conv_id,
            original_uid=orig_uid,
            role=role,
            token_count=len(token_list),
            transform_type=transform_type,
            metrics=mean_metrics,
        )

    return results


def aggregate_by_role(tokens: List[TokenMetrics]) -> Dict[Tuple[str, str], MessageMetrics]:
    """Aggregate token metrics by (conversation_id, role).

    Args:
        tokens: List of token-level metrics.

    Returns:
        Dict mapping (conversation_id, role) to aggregated MessageMetrics.
    """
    # Group by (conversation_id, role)
    groups: Dict[Tuple[str, str], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        key = (token.conversation_id, token.role)
        groups[key].append(token)

    results = {}
    for key, token_list in groups.items():
        conv_id, role = key

        # Aggregate metrics
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)

        for token in token_list:
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        # Compute means
        mean_metrics = {}
        for metric_name, total in metric_sums.items():
            count = metric_counts[metric_name]
            if count > 0:
                mean_metrics[metric_name] = total / count

        results[key] = MessageMetrics(
            conversation_id=conv_id,
            original_uid=-1,  # Not applicable for role aggregation
            role=role,
            token_count=len(token_list),
            transform_type="mixed",
            metrics=mean_metrics,
        )

    return results


def aggregate_by_effective_role(tokens: List[TokenMetrics]) -> Dict[Tuple[str, str], MessageMetrics]:
    """Aggregate token metrics by (conversation_id, effective_role).

    This separates tool messages by transform_type (tool_original vs tool_summarized).

    Args:
        tokens: List of token-level metrics.

    Returns:
        Dict mapping (conversation_id, effective_role) to aggregated MessageMetrics.
    """
    # Group by (conversation_id, effective_role)
    groups: Dict[Tuple[str, str], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        key = (token.conversation_id, token.effective_role)
        groups[key].append(token)

    results = {}
    for key, token_list in groups.items():
        conv_id, eff_role = key

        # Aggregate metrics
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)

        transform_type = token_list[0].transform_type

        for token in token_list:
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        # Compute means
        mean_metrics = {}
        for metric_name, total in metric_sums.items():
            count = metric_counts[metric_name]
            if count > 0:
                mean_metrics[metric_name] = total / count

        results[key] = MessageMetrics(
            conversation_id=conv_id,
            original_uid=-1,
            role=eff_role,
            token_count=len(token_list),
            transform_type=transform_type,
            metrics=mean_metrics,
        )

    return results


def compare_runs_by_message(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str = "run1",
    run2_name: str = "run2",
    metric_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare two runs aligned by (conversation_id, original_uid).

    Args:
        run1_tokens: Token metrics from first run.
        run2_tokens: Token metrics from second run.
        run1_name: Name for first run.
        run2_name: Name for second run.
        metric_names: List of metrics to compare. If None, use all common metrics.

    Returns:
        List of comparison dicts with metrics from both runs and deltas.
    """
    # Aggregate by message
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common keys
    all_keys = set(run1_by_msg.keys()) | set(run2_by_msg.keys())

    # Determine metrics to compare
    if metric_names is None:
        # Find common metrics
        all_metrics: Set[str] = set()
        for msg in run1_by_msg.values():
            all_metrics.update(msg.metrics.keys())
        for msg in run2_by_msg.values():
            all_metrics.update(msg.metrics.keys())
        metric_names = sorted(all_metrics)

    results = []
    for key in sorted(all_keys):
        conv_id, orig_uid = key

        msg1 = run1_by_msg.get(key)
        msg2 = run2_by_msg.get(key)

        row = {
            "conversation_id": conv_id,
            "original_uid": orig_uid,
            "role": msg1.role if msg1 else (msg2.role if msg2 else "unknown"),
            f"{run1_name}_tokens": msg1.token_count if msg1 else 0,
            f"{run2_name}_tokens": msg2.token_count if msg2 else 0,
            f"{run1_name}_transform": msg1.transform_type if msg1 else "missing",
            f"{run2_name}_transform": msg2.transform_type if msg2 else "missing",
        }

        for metric in metric_names:
            val1 = msg1.metrics.get(metric) if msg1 else None
            val2 = msg2.metrics.get(metric) if msg2 else None

            row[f"{run1_name}_{metric}"] = val1 if val1 is not None else ""
            row[f"{run2_name}_{metric}"] = val2 if val2 is not None else ""

            if val1 is not None and val2 is not None:
                row[f"delta_{metric}"] = val2 - val1
            else:
                row[f"delta_{metric}"] = ""

        results.append(row)

    return results


def compare_runs_by_role(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str = "run1",
    run2_name: str = "run2",
    metric_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare two runs aggregated by (conversation_id, role).

    Args:
        run1_tokens: Token metrics from first run.
        run2_tokens: Token metrics from second run.
        run1_name: Name for first run.
        run2_name: Name for second run.
        metric_names: List of metrics to compare.

    Returns:
        List of comparison dicts aggregated by role.
    """
    run1_by_role = aggregate_by_role(run1_tokens)
    run2_by_role = aggregate_by_role(run2_tokens)

    all_keys = set(run1_by_role.keys()) | set(run2_by_role.keys())

    if metric_names is None:
        all_metrics: Set[str] = set()
        for msg in run1_by_role.values():
            all_metrics.update(msg.metrics.keys())
        for msg in run2_by_role.values():
            all_metrics.update(msg.metrics.keys())
        metric_names = sorted(all_metrics)

    results = []
    for key in sorted(all_keys):
        conv_id, role = key

        msg1 = run1_by_role.get(key)
        msg2 = run2_by_role.get(key)

        row = {
            "conversation_id": conv_id,
            "role": role,
            f"{run1_name}_tokens": msg1.token_count if msg1 else 0,
            f"{run2_name}_tokens": msg2.token_count if msg2 else 0,
        }

        for metric in metric_names:
            val1 = msg1.metrics.get(metric) if msg1 else None
            val2 = msg2.metrics.get(metric) if msg2 else None

            row[f"{run1_name}_{metric}"] = val1 if val1 is not None else ""
            row[f"{run2_name}_{metric}"] = val2 if val2 is not None else ""

            if val1 is not None and val2 is not None:
                row[f"delta_{metric}"] = val2 - val1
            else:
                row[f"delta_{metric}"] = ""

        results.append(row)

    return results


def save_comparison_csv(rows: List[Dict[str, Any]], output_path: Path):
    """Save comparison results to CSV.

    Args:
        rows: List of comparison dicts.
        output_path: Output file path.
    """
    if not rows:
        print("No comparison results to save")
        return

    fieldnames = list(rows[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved comparison to {output_path}")


def print_summary(
    rows: List[Dict[str, Any]],
    run1_name: str,
    run2_name: str,
    metric_names: List[str],
):
    """Print summary statistics for comparison.

    Args:
        rows: List of comparison dicts.
        run1_name: Name of first run.
        run2_name: Name of second run.
        metric_names: List of metrics to summarize.
    """
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for metric in metric_names:
        delta_key = f"delta_{metric}"
        deltas = [
            row[delta_key]
            for row in rows
            if isinstance(row.get(delta_key), (int, float))
        ]

        if not deltas:
            continue

        mean_delta = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)

        print(f"\n{metric}:")
        print(f"  Mean delta ({run2_name} - {run1_name}): {mean_delta:+.4f}")
        print(f"  {run2_name} higher: {positive} ({100*positive/len(deltas):.1f}%)")
        print(f"  {run1_name} higher: {negative} ({100*negative/len(deltas):.1f}%)")

    print("\n" + "=" * 70)


def analyze_by_transform_pair(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str = "run1",
    run2_name: str = "run2",
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Analyze metrics grouped by transform type pair.

    Groups messages by their (run1_transform, run2_transform) combination
    to understand the effect of transformations.

    Args:
        run1_tokens: Token metrics from first run.
        run2_tokens: Token metrics from second run.
        run1_name: Name for first run.
        run2_name: Name for second run.
        metric_names: Metrics to analyze.

    Returns:
        Dict mapping transform_pair to statistics.
    """
    # Aggregate by message
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common keys
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    if not common_keys:
        return {}

    # Determine metrics
    if metric_names is None:
        all_metrics: Set[str] = set()
        for msg in run1_by_msg.values():
            all_metrics.update(msg.metrics.keys())
        # Exclude perplexity by default
        metric_names = sorted(m for m in all_metrics if m != "perplexity")

    # Group by transform pair
    pairs: Dict[str, List[Tuple[MessageMetrics, MessageMetrics]]] = defaultdict(list)

    for key in common_keys:
        msg1 = run1_by_msg[key]
        msg2 = run2_by_msg[key]
        pair_key = f"{msg1.transform_type} -> {msg2.transform_type}"
        pairs[pair_key].append((msg1, msg2))

    # Compute statistics for each pair
    results = {}
    for pair_key, msg_pairs in pairs.items():
        stats = {
            "count": len(msg_pairs),
            f"{run1_name}_tokens": sum(m1.token_count for m1, _ in msg_pairs),
            f"{run2_name}_tokens": sum(m2.token_count for _, m2 in msg_pairs),
            "metrics": {},
        }

        for metric in metric_names:
            vals1 = [m1.metrics.get(metric) for m1, _ in msg_pairs if m1.metrics.get(metric) is not None]
            vals2 = [m2.metrics.get(metric) for _, m2 in msg_pairs if m2.metrics.get(metric) is not None]

            if vals1 and vals2:
                # Filter to matching pairs with valid values
                valid_pairs = [
                    (m1.metrics.get(metric), m2.metrics.get(metric))
                    for m1, m2 in msg_pairs
                    if m1.metrics.get(metric) is not None and m2.metrics.get(metric) is not None
                ]
                if valid_pairs:
                    deltas = [v2 - v1 for v1, v2 in valid_pairs]
                    stats["metrics"][metric] = {
                        f"{run1_name}_mean": np.mean(vals1),
                        f"{run1_name}_std": np.std(vals1),
                        f"{run2_name}_mean": np.mean(vals2),
                        f"{run2_name}_std": np.std(vals2),
                        "delta_mean": np.mean(deltas),
                        f"{run2_name}_higher": sum(1 for d in deltas if d > 0),
                        "total": len(deltas),
                    }

        results[pair_key] = stats

    return results


def print_transform_pair_summary(
    analysis: Dict[str, Dict[str, Any]],
    run1_name: str,
    run2_name: str,
):
    """Print summary of metrics by transform pair.

    Args:
        analysis: Output from analyze_by_transform_pair.
        run1_name: Name of first run.
        run2_name: Name of second run.
    """
    print("\n" + "=" * 70)
    print("METRICS BY TRANSFORM PAIR")
    print("=" * 70)

    for pair_key in sorted(analysis.keys()):
        # Skip unknown -> unknown
        if "unknown" in pair_key:
            continue

        stats = analysis[pair_key]
        print(f"\n--- {pair_key} (n={stats['count']}) ---")
        print(f"Tokens: {run1_name}={stats[f'{run1_name}_tokens']:,}, {run2_name}={stats[f'{run2_name}_tokens']:,}")

        for metric, mstats in stats["metrics"].items():
            r1_mean = mstats[f"{run1_name}_mean"]
            r2_mean = mstats[f"{run2_name}_mean"]
            delta = mstats["delta_mean"]
            higher = mstats[f"{run2_name}_higher"]
            total = mstats["total"]

            print(f"  {metric}: {run1_name}={r1_mean:.4f}, {run2_name}={r2_mean:.4f}, "
                  f"delta={delta:+.4f} ({run2_name} higher: {higher}/{total})")

    print("\n" + "=" * 70)


def get_tokens_by_transform_pair(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
) -> Tuple[
    Dict[str, Tuple[List[TokenMetrics], List[TokenMetrics]]],
    Dict[Tuple[str, int, int], int],
]:
    """Group tokens by their transform pair.

    Args:
        run1_tokens: Tokens from first run.
        run2_tokens: Tokens from second run.

    Returns:
        Tuple of:
        - Dict mapping transform_pair (e.g., "original -> summarized") to
          (run1_tokens_subset, run2_tokens_subset) for that pair.
        - Dict mapping (conversation_id, original_uid, section_token_idx) to
          run1's token_idx for position alignment.
    """
    # Aggregate by message to determine transform types
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common keys and their transform pairs
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    # Map original_uid to transform pair
    uid_to_pair: Dict[Tuple[str, int], str] = {}
    for key in common_keys:
        msg1 = run1_by_msg[key]
        msg2 = run2_by_msg[key]
        pair_key = f"{msg1.transform_type} -> {msg2.transform_type}"
        uid_to_pair[key] = pair_key

    # Build position mapping from run1 (baseline)
    # Maps (conv_id, original_uid, token_offset_in_section) -> run1's token_idx
    # We need to track relative position within each message section
    uid_token_positions: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
    for token in run1_tokens:
        key = (token.conversation_id, token.original_uid)
        if key in uid_to_pair:
            uid_token_positions[key].append((token.token_idx, token.section_idx))

    # Sort by token_idx to get ordering within each message
    position_map: Dict[Tuple[str, int, int], int] = {}
    for (conv_id, orig_uid), positions in uid_token_positions.items():
        positions.sort(key=lambda x: x[0])  # Sort by token_idx
        for offset, (token_idx, _) in enumerate(positions):
            position_map[(conv_id, orig_uid, offset)] = token_idx

    # Group tokens by transform pair
    pairs: Dict[str, Tuple[List[TokenMetrics], List[TokenMetrics]]] = defaultdict(
        lambda: ([], [])
    )

    for token in run1_tokens:
        key = (token.conversation_id, token.original_uid)
        if key in uid_to_pair:
            pair_key = uid_to_pair[key]
            pairs[pair_key][0].append(token)

    for token in run2_tokens:
        key = (token.conversation_id, token.original_uid)
        if key in uid_to_pair:
            pair_key = uid_to_pair[key]
            pairs[pair_key][1].append(token)

    return dict(pairs), position_map


def plot_metric_by_position_for_transform_pair(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    metric_name: str,
    transform_pair: str,
    output_path: Path,
    position_map: Dict[Tuple[str, int, int], int],
    bin_size: int = 100,
    roles: Optional[List[str]] = None,
):
    """Plot metric by position for a specific transform pair, comparing two runs.

    Creates a subplot for each role, showing run1 vs run2 lines.
    Both runs are plotted against run1's (baseline) token positions.

    Args:
        run1_tokens: Tokens from first run (filtered to this transform pair).
        run2_tokens: Tokens from second run (filtered to this transform pair).
        run1_name: Name of first run.
        run2_name: Name of second run.
        metric_name: Metric to plot.
        transform_pair: Transform pair label (e.g., "original -> summarized").
        output_path: Output file path.
        position_map: Maps (conv_id, original_uid, offset) to run1's token_idx.
        bin_size: Bin size for position grouping.
        roles: Roles to include.
    """
    import matplotlib.pyplot as plt

    if roles is None:
        roles = ["assistant", "tool"]

    # Role colors for run1 and run2
    run_colors = {
        run1_name: {
            "system": "#1a7a1a",
            "user": "#1a5a9e",
            "assistant": "#6a3a9e",
            "tool": "#9e2a2a",
        },
        run2_name: {
            "system": "#5acc5a",
            "user": "#5a9aee",
            "assistant": "#ba8aee",
            "tool": "#ee6a6a",
        },
    }

    # Collect position -> values for run1 (use own token_idx)
    def collect_run1_by_position(
        tokens: List[TokenMetrics],
    ) -> Dict[str, Dict[int, List[float]]]:
        role_pos_vals: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for token in tokens:
            if metric_name in token.metrics:
                val = token.metrics[metric_name]
                if math.isfinite(val):
                    role_pos_vals[token.role][token.token_idx].append(val)
        return role_pos_vals

    # Collect position -> values for run2 (use run1's token_idx via position_map)
    def collect_run2_by_position(
        tokens: List[TokenMetrics],
    ) -> Dict[str, Dict[int, List[float]]]:
        role_pos_vals: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Group tokens by (conv_id, original_uid) to determine offset
        uid_tokens: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
        for token in tokens:
            key = (token.conversation_id, token.original_uid)
            uid_tokens[key].append(token)

        # Sort each group by token_idx and map to run1 positions
        for (conv_id, orig_uid), token_list in uid_tokens.items():
            token_list.sort(key=lambda t: t.token_idx)
            for offset, token in enumerate(token_list):
                if metric_name in token.metrics:
                    val = token.metrics[metric_name]
                    if math.isfinite(val):
                        # Look up run1's position for this offset
                        map_key = (conv_id, orig_uid, offset)
                        if map_key in position_map:
                            run1_pos = position_map[map_key]
                            role_pos_vals[token.role][run1_pos].append(val)
        return role_pos_vals

    run1_data = collect_run1_by_position(run1_tokens)
    run2_data = collect_run2_by_position(run2_tokens)

    # Determine which roles have data
    available_roles = [
        r for r in roles
        if r in run1_data or r in run2_data
    ]

    if not available_roles:
        print(f"No data for roles {roles} in transform pair {transform_pair}")
        return

    # Create subplots
    fig, axes = plt.subplots(1, len(available_roles), figsize=(7 * len(available_roles), 5))
    if len(available_roles) == 1:
        axes = [axes]

    for ax, role in zip(axes, available_roles):
        # Get max position from run1 data (baseline positions)
        max_pos = 0
        if role in run1_data and run1_data[role]:
            max_pos = max(max_pos, max(run1_data[role].keys()))

        if max_pos == 0:
            ax.set_title(f"{role} (no data)")
            continue

        num_bins = (max_pos // bin_size) + 1

        # Plot each run
        for run_name, run_data_dict in [(run1_name, run1_data), (run2_name, run2_data)]:
            if role not in run_data_dict:
                continue

            positions = run_data_dict[role]
            bin_centers = []
            bin_means = []
            bin_stds = []

            for bin_idx in range(num_bins):
                bin_start = bin_idx * bin_size
                bin_end = (bin_idx + 1) * bin_size

                bin_values = []
                for pos in range(bin_start, bin_end):
                    if pos in positions:
                        bin_values.extend(positions[pos])

                if bin_values:
                    bin_centers.append((bin_start + bin_end) / 2)
                    bin_means.append(np.mean(bin_values))
                    bin_stds.append(np.std(bin_values))

            if not bin_centers:
                continue

            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)

            # Convert to K (thousands) for x-axis
            bin_centers_k = bin_centers / 1000

            color = run_colors[run_name].get(role, "#888888")

            ax.plot(bin_centers_k, bin_means, color=color, label=run_name, linewidth=2)
            ax.fill_between(
                bin_centers_k,
                bin_means - bin_stds,
                bin_means + bin_stds,
                color=color,
                alpha=0.2,
            )

        ax.set_xlabel("Token Position (K)")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{role}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{metric_name.replace('_', ' ').title()} by Position\n"
        f"Transform: {transform_pair}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {transform_pair} position plot to {output_path}")


def plot_metric_delta_by_position_for_transform_pair(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    metric_name: str,
    transform_pair: str,
    output_path: Path,
    position_map: Dict[Tuple[str, int, int], int],
    position_bins: int = 50,
    delta_bins: int = 50,
    roles: Optional[List[str]] = None,
):
    """Plot metric delta (run2 - run1) by position as heatmap for a specific transform pair.

    Creates subplots for each role showing density of delta values by position.

    Args:
        run1_tokens: Tokens from first run (filtered to this transform pair).
        run2_tokens: Tokens from second run (filtered to this transform pair).
        run1_name: Name of first run.
        run2_name: Name of second run.
        metric_name: Metric to plot.
        transform_pair: Transform pair label (e.g., "original -> summarized").
        output_path: Output file path.
        position_map: Maps (conv_id, original_uid, offset) to run1's token_idx.
        position_bins: Number of bins for position axis.
        delta_bins: Number of bins for delta axis.
        roles: Roles to include.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if roles is None:
        roles = ["assistant", "tool"]

    # Build run1 position -> metric value mapping
    # Key: (conv_id, original_uid, offset) -> (position, value, role)
    run1_values: Dict[Tuple[str, int, int], Tuple[int, float, str]] = {}
    uid_tokens_run1: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
    for token in run1_tokens:
        key = (token.conversation_id, token.original_uid)
        uid_tokens_run1[key].append(token)

    for (conv_id, orig_uid), token_list in uid_tokens_run1.items():
        token_list.sort(key=lambda t: t.token_idx)
        for offset, token in enumerate(token_list):
            if metric_name in token.metrics:
                val = token.metrics[metric_name]
                if math.isfinite(val):
                    map_key = (conv_id, orig_uid, offset)
                    run1_values[map_key] = (token.token_idx, val, token.role)

    # Build run2 offset -> metric value mapping
    run2_values: Dict[Tuple[str, int, int], Tuple[float, str]] = {}
    uid_tokens_run2: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
    for token in run2_tokens:
        key = (token.conversation_id, token.original_uid)
        uid_tokens_run2[key].append(token)

    for (conv_id, orig_uid), token_list in uid_tokens_run2.items():
        token_list.sort(key=lambda t: t.token_idx)
        for offset, token in enumerate(token_list):
            if metric_name in token.metrics:
                val = token.metrics[metric_name]
                if math.isfinite(val):
                    map_key = (conv_id, orig_uid, offset)
                    run2_values[map_key] = (val, token.role)

    # Compute deltas for matching tokens
    # Structure: {role: [(position, delta), ...]}
    role_deltas: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    common_keys = set(run1_values.keys()) & set(run2_values.keys())
    for key in common_keys:
        pos, val1, role = run1_values[key]
        val2, _ = run2_values[key]
        delta = val2 - val1
        if math.isfinite(delta):
            role_deltas[role].append((pos, delta))

    # Filter to requested roles that have data
    available_roles = [r for r in roles if r in role_deltas and role_deltas[r]]

    if not available_roles:
        print(f"No delta data for roles {roles} in transform pair {transform_pair}")
        return

    # Create subplots for each role
    fig, axes = plt.subplots(1, len(available_roles), figsize=(7 * len(available_roles), 5))
    if len(available_roles) == 1:
        axes = [axes]

    for ax, role in zip(axes, available_roles):
        if role not in role_deltas or not role_deltas[role]:
            ax.set_title(f"{role} (no data)")
            continue

        positions, deltas = zip(*role_deltas[role])
        positions = np.array(positions)
        deltas = np.array(deltas)

        # Convert to K (thousands) for x-axis
        positions_k = positions / 1000

        # Create 2D histogram (heatmap)
        h, xedges, yedges, im = ax.hist2d(
            positions_k,
            deltas,
            bins=[position_bins, delta_bins],
            cmap='viridis',
            norm=LogNorm(),
            cmin=1,  # Minimum count to show
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

        # Add mean trend line
        bin_size = (positions_k.max() - positions_k.min()) / position_bins if positions_k.max() > positions_k.min() else 1
        bin_centers = []
        bin_means = []

        for i in range(position_bins):
            bin_start = positions_k.min() + i * bin_size
            bin_end = bin_start + bin_size
            mask = (positions_k >= bin_start) & (positions_k < bin_end)
            if mask.sum() > 0:
                bin_centers.append((bin_start + bin_end) / 2)
                bin_means.append(deltas[mask].mean())

        if bin_centers:
            ax.plot(bin_centers, bin_means, color='red', linewidth=2, label='Mean')
            ax.legend(loc='upper right')

        ax.set_xlabel("Token Position (K)")
        ax.set_ylabel(f"Δ {metric_name}")
        ax.set_title(f"{role} (n={len(deltas):,})")

    plt.suptitle(
        f"Δ {metric_name.replace('_', ' ').title()} ({run2_name} - {run1_name}) by Position\n"
        f"Transform: {transform_pair}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {transform_pair} delta heatmap to {output_path}")


def plot_transform_pair_position_metrics(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    metric_names: List[str],
    output_dir: Path,
    roles: Optional[List[str]] = None,
):
    """Generate position-metric plots for each transform pair.

    Args:
        run1_tokens: Tokens from first run.
        run2_tokens: Tokens from second run.
        run1_name: Name of first run.
        run2_name: Name of second run.
        metric_names: Metrics to plot.
        output_dir: Output directory.
        roles: Roles to include in plots.
    """
    # Group tokens by transform pair and get position mapping
    pairs, position_map = get_tokens_by_transform_pair(run1_tokens, run2_tokens)

    if not pairs:
        print("No transform pairs found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_key, (run1_subset, run2_subset) in pairs.items():
        # Skip unknown pairs
        if "unknown" in pair_key:
            continue

        # Create sanitized filename
        pair_label = pair_key.replace(" -> ", "_to_").replace(" ", "_")

        for metric in metric_names:
            # Position plot (both runs)
            output_path = output_dir / f"{metric}_by_position_{pair_label}.png"
            plot_metric_by_position_for_transform_pair(
                run1_subset,
                run2_subset,
                run1_name,
                run2_name,
                metric,
                pair_key,
                output_path,
                position_map=position_map,
                roles=roles,
            )

            # Delta scatter plot
            delta_path = output_dir / f"{metric}_delta_by_position_{pair_label}.png"
            plot_metric_delta_by_position_for_transform_pair(
                run1_subset,
                run2_subset,
                run1_name,
                run2_name,
                metric,
                pair_key,
                delta_path,
                position_map=position_map,
                roles=roles,
            )


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_token_reduction_by_role(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    output_path: Path,
    split_tool_by_transform: bool = True,
):
    """Plot token count comparison by role.

    Shows how many tokens each role has before and after transformation.
    Messages are matched by original_uid first to ensure fair comparison.

    Args:
        split_tool_by_transform: If True, show tool_original and tool_summarized separately.
    """
    import matplotlib.pyplot as plt

    # Build message-level aggregation for matching by original_uid
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    # Map original_uid to run2's transform_type for consistent categorization
    uid_to_transform: Dict[Tuple[str, int], str] = {}
    for key in common_keys:
        msg2 = run2_by_msg[key]
        uid_to_transform[key] = msg2.transform_type

    # Aggregate tokens by role (using run2's transform_type for tools)
    run1_by_role: Dict[str, int] = defaultdict(int)
    run2_by_role: Dict[str, int] = defaultdict(int)

    for token in run1_tokens:
        key = (token.conversation_id, token.original_uid)
        if key not in common_keys:
            continue
        if split_tool_by_transform and token.role == "tool":
            role_key = f"tool_{uid_to_transform.get(key, 'original')}"
        else:
            role_key = token.role
        run1_by_role[role_key] += 1

    for token in run2_tokens:
        key = (token.conversation_id, token.original_uid)
        if key not in common_keys:
            continue
        if split_tool_by_transform and token.role == "tool":
            role_key = f"tool_{uid_to_transform.get(key, 'original')}"
        else:
            role_key = token.role
        run2_by_role[role_key] += 1

    # Get all roles
    all_roles = sorted(set(run1_by_role.keys()) | set(run2_by_role.keys()))

    # Filter to main roles (including tool_original and tool_summarized)
    main_role_order = ["system", "user", "assistant", "tool", "tool_original", "tool_summarized", "developer"]
    main_roles = [r for r in main_role_order if r in all_roles]
    if not main_roles:
        main_roles = all_roles[:7]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(main_roles))
    width = 0.35

    run1_counts = [run1_by_role.get(r, 0) for r in main_roles]
    run2_counts = [run2_by_role.get(r, 0) for r in main_roles]

    bars1 = ax.bar(x - width/2, run1_counts, width, label=run1_name, color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, run2_counts, width, label=run2_name, color='coral', alpha=0.8)

    ax.set_xlabel('Role', fontsize=12)
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_title('Token Count by Role: Before vs After Transformation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(main_roles, rotation=30, ha='right')
    ax.legend()

    # Add value labels on bars
    for bar, count in zip(bars1, run1_counts):
        if count > 0:
            ax.annotate(f'{count:,}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, count in zip(bars2, run2_counts):
        if count > 0:
            ax.annotate(f'{count:,}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)

    # Add reduction percentage annotation for tool_summarized
    if "tool_summarized" in main_roles:
        tool_idx = main_roles.index("tool_summarized")
        t1 = run1_counts[tool_idx]
        t2 = run2_counts[tool_idx]
        if t1 > 0:
            reduction = (1 - t2/t1) * 100
            ax.annotate(f'{reduction:.1f}% reduction',
                       xy=(tool_idx, max(t1, t2) * 1.1),
                       ha='center', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved token reduction plot to {output_path}")


def plot_metric_comparison_by_role(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    metric_name: str,
    output_path: Path,
    split_tool_by_transform: bool = True,
):
    """Plot metric comparison by role as grouped bar chart.

    Messages are matched by original_uid first, then aggregated by role.
    For tool messages, the role is determined by run2's transform_type
    (tool_original vs tool_summarized) to show the effect of transformation.

    Args:
        split_tool_by_transform: If True, show tool_original and tool_summarized separately.
    """
    import matplotlib.pyplot as plt

    # Aggregate by message to align by original_uid
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common messages
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    # Aggregate matched messages by role (using run2's transform_type for tools)
    role_metrics1: Dict[str, List[float]] = defaultdict(list)
    role_metrics2: Dict[str, List[float]] = defaultdict(list)

    for key in common_keys:
        msg1 = run1_by_msg[key]
        msg2 = run2_by_msg[key]

        # Determine role key (use run2's transform_type for tool categorization)
        if split_tool_by_transform and msg1.role == "tool":
            role_key = f"tool_{msg2.transform_type}"
        else:
            role_key = msg1.role

        if metric_name in msg1.metrics:
            role_metrics1[role_key].append(msg1.metrics[metric_name])
        if metric_name in msg2.metrics:
            role_metrics2[role_key].append(msg2.metrics[metric_name])

    # Filter to main roles (including tool_original and tool_summarized)
    main_role_order = ["system", "user", "assistant", "tool", "tool_original", "tool_summarized", "developer"]
    main_roles = [r for r in main_role_order
                  if r in role_metrics1 or r in role_metrics2]

    if not main_roles:
        print(f"No data for {metric_name} by role")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(main_roles))
    width = 0.35

    run1_means = [np.mean(role_metrics1.get(r, [np.nan])) for r in main_roles]
    run2_means = [np.mean(role_metrics2.get(r, [np.nan])) for r in main_roles]
    run1_stds = [np.std(role_metrics1.get(r, [0])) for r in main_roles]
    run2_stds = [np.std(role_metrics2.get(r, [0])) for r in main_roles]

    bars1 = ax.bar(x - width/2, run1_means, width, yerr=run1_stds,
                   label=run1_name, color='steelblue', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, run2_means, width, yerr=run2_stds,
                   label=run2_name, color='coral', alpha=0.8, capsize=3)

    ax.set_xlabel('Role', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} by Role: {run1_name} vs {run2_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(main_roles, rotation=30, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {metric_name} comparison plot to {output_path}")


def plot_metric_scatter(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    metric_name: str,
    output_path: Path,
    split_tool_by_transform: bool = True,
):
    """Scatter plot comparing metrics between two runs, colored by role.

    Args:
        split_tool_by_transform: If True, show tool_original and tool_summarized separately.
    """
    import matplotlib.pyplot as plt

    # Aggregate by message to align data points
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common keys
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    if not common_keys:
        print(f"No common messages to compare for {metric_name}")
        return

    # Collect data points by effective_role
    data_by_role: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for key in common_keys:
        msg1 = run1_by_msg[key]
        msg2 = run2_by_msg[key]

        val1 = msg1.metrics.get(metric_name)
        val2 = msg2.metrics.get(metric_name)

        if val1 is not None and val2 is not None and math.isfinite(val1) and math.isfinite(val2):
            # Use effective_role if splitting tools
            if split_tool_by_transform and msg1.role == "tool":
                # Use the transform type from the second run (which shows what happened)
                role_key = f"tool_{msg2.transform_type}"
            else:
                role_key = msg1.role
            data_by_role[role_key].append((val1, val2))

    if not data_by_role:
        print(f"No valid data for {metric_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        'system': 'green',
        'user': 'blue',
        'assistant': 'orange',
        'tool': 'red',
        'tool_original': 'darkred',
        'tool_summarized': 'lightcoral',
        'developer': 'purple'
    }

    for role, points in sorted(data_by_role.items()):
        if points:
            x_vals, y_vals = zip(*points)
            ax.scatter(x_vals, y_vals, label=role, alpha=0.6,
                      c=colors.get(role, 'gray'), s=50)

    # Add diagonal line (y=x)
    all_vals = []
    for points in data_by_role.values():
        for v1, v2 in points:
            all_vals.extend([v1, v2])

    if all_vals:
        min_val, max_val = min(all_vals), max(all_vals)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

    ax.set_xlabel(f'{metric_name} ({run1_name})', fontsize=12)
    ax.set_ylabel(f'{metric_name} ({run2_name})', fontsize=12)
    ax.set_title(f'{metric_name}: {run1_name} vs {run2_name}', fontsize=14)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {metric_name} scatter plot to {output_path}")


def plot_delta_distribution(
    comparison: List[Dict[str, Any]],
    run1_name: str,
    run2_name: str,
    metric_name: str,
    output_path: Path,
):
    """Plot distribution of metric deltas (histogram)."""
    import matplotlib.pyplot as plt

    delta_key = f"delta_{metric_name}"
    deltas = [
        row[delta_key]
        for row in comparison
        if isinstance(row.get(delta_key), (int, float)) and math.isfinite(row[delta_key])
    ]

    if not deltas:
        print(f"No delta data for {metric_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(deltas, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

    # Color bars based on positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0:
            patch.set_facecolor('coral')
        else:
            patch.set_facecolor('steelblue')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=np.mean(deltas), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(deltas):.3f}')

    ax.set_xlabel(f'Delta {metric_name} ({run2_name} - {run1_name})', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {metric_name} Changes', fontsize=14)
    ax.legend()

    # Add summary text
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)
    textstr = f'Increased: {positive} ({100*positive/len(deltas):.1f}%)\nDecreased: {negative} ({100*negative/len(deltas):.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {metric_name} delta distribution to {output_path}")


def plot_transformation_summary(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    comparison_by_role: List[Dict[str, Any]],
    run1_name: str,
    run2_name: str,
    output_path: Path,
):
    """Create a comprehensive summary figure showing transformation effects.

    Separates tool metrics into tool_original and tool_summarized categories.
    Messages are matched by original_uid first to ensure fair comparison.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12))

    # Build message-level aggregation for matching by original_uid
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    # Build token index sets for common messages
    run1_common_tokens = []
    run2_common_tokens = []
    for token in run1_tokens:
        key = (token.conversation_id, token.original_uid)
        if key in common_keys:
            run1_common_tokens.append(token)
    for token in run2_tokens:
        key = (token.conversation_id, token.original_uid)
        if key in common_keys:
            run2_common_tokens.append(token)

    # 1. Token count comparison (top left) - with tool split
    # Use run2's transform_type to categorize roles consistently
    ax1 = fig.add_subplot(2, 2, 1)

    run1_by_role: Dict[str, int] = defaultdict(int)
    run2_by_role: Dict[str, int] = defaultdict(int)

    # Map original_uid to run2's transform_type
    uid_to_transform: Dict[Tuple[str, int], str] = {}
    for key in common_keys:
        msg2 = run2_by_msg[key]
        uid_to_transform[key] = msg2.transform_type

    for token in run1_common_tokens:
        key = (token.conversation_id, token.original_uid)
        if token.role == "tool":
            role_key = f"tool_{uid_to_transform.get(key, 'original')}"
        else:
            role_key = token.role
        run1_by_role[role_key] += 1

    for token in run2_common_tokens:
        key = (token.conversation_id, token.original_uid)
        if token.role == "tool":
            role_key = f"tool_{uid_to_transform.get(key, 'original')}"
        else:
            role_key = token.role
        run2_by_role[role_key] += 1

    # Include tool_original and tool_summarized
    main_roles = ["system", "user", "assistant", "tool_original", "tool_summarized"]
    # Filter to roles that exist
    main_roles = [r for r in main_roles if r in run1_by_role or r in run2_by_role]

    x = np.arange(len(main_roles))
    width = 0.35

    run1_counts = [run1_by_role.get(r, 0) for r in main_roles]
    run2_counts = [run2_by_role.get(r, 0) for r in main_roles]

    ax1.bar(x - width/2, run1_counts, width, label=run1_name, color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, run2_counts, width, label=run2_name, color='coral', alpha=0.8)
    ax1.set_xlabel('Role')
    ax1.set_ylabel('Token Count')
    ax1.set_title('Token Count by Role (Tools Split by Transform)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(main_roles, rotation=30, ha='right')
    ax1.legend()

    # Add reduction annotation
    total1 = sum(run1_counts)
    total2 = sum(run2_counts)
    reduction = (1 - total2/total1) * 100 if total1 > 0 else 0
    ax1.annotate(f'Total reduction: {reduction:.1f}%',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=11, color='red', fontweight='bold')

    # 2. Metric comparison for neg_log_prob (top right)
    ax2 = fig.add_subplot(2, 2, 2)

    role_metrics1: Dict[str, List[float]] = defaultdict(list)
    role_metrics2: Dict[str, List[float]] = defaultdict(list)

    for token in run1_common_tokens:
        if "neg_log_prob" in token.metrics:
            key = (token.conversation_id, token.original_uid)
            if token.role == "tool":
                role_key = f"tool_{uid_to_transform.get(key, 'original')}"
            else:
                role_key = token.role
            role_metrics1[role_key].append(token.metrics["neg_log_prob"])

    for token in run2_common_tokens:
        if "neg_log_prob" in token.metrics:
            key = (token.conversation_id, token.original_uid)
            if token.role == "tool":
                role_key = f"tool_{uid_to_transform.get(key, 'original')}"
            else:
                role_key = token.role
            role_metrics2[role_key].append(token.metrics["neg_log_prob"])

    run1_means = [np.mean(role_metrics1.get(r, [np.nan])) for r in main_roles]
    run2_means = [np.mean(role_metrics2.get(r, [np.nan])) for r in main_roles]

    ax2.bar(x - width/2, run1_means, width, label=run1_name, color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, run2_means, width, label=run2_name, color='coral', alpha=0.8)
    ax2.set_xlabel('Role')
    ax2.set_ylabel('neg_log_prob')
    ax2.set_title('Negative Log Probability by Role')
    ax2.set_xticks(x)
    ax2.set_xticklabels(main_roles, rotation=30, ha='right')
    ax2.legend()

    # 3. Entropy comparison (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)

    role_entropy1: Dict[str, List[float]] = defaultdict(list)
    role_entropy2: Dict[str, List[float]] = defaultdict(list)

    for token in run1_common_tokens:
        if "estimated_entropy" in token.metrics:
            key = (token.conversation_id, token.original_uid)
            if token.role == "tool":
                role_key = f"tool_{uid_to_transform.get(key, 'original')}"
            else:
                role_key = token.role
            role_entropy1[role_key].append(token.metrics["estimated_entropy"])

    for token in run2_common_tokens:
        if "estimated_entropy" in token.metrics:
            key = (token.conversation_id, token.original_uid)
            if token.role == "tool":
                role_key = f"tool_{uid_to_transform.get(key, 'original')}"
            else:
                role_key = token.role
            role_entropy2[role_key].append(token.metrics["estimated_entropy"])

    run1_entropy = [np.mean(role_entropy1.get(r, [np.nan])) for r in main_roles]
    run2_entropy = [np.mean(role_entropy2.get(r, [np.nan])) for r in main_roles]

    ax3.bar(x - width/2, run1_entropy, width, label=run1_name, color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, run2_entropy, width, label=run2_name, color='coral', alpha=0.8)
    ax3.set_xlabel('Role')
    ax3.set_ylabel('Estimated Entropy')
    ax3.set_title('Entropy by Role')
    ax3.set_xticks(x)
    ax3.set_xticklabels(main_roles, rotation=30, ha='right')
    ax3.legend()

    # 4. Summary statistics (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Calculate tool stats
    tool_orig_1 = run1_by_role.get('tool_original', 0)
    tool_orig_2 = run2_by_role.get('tool_original', 0)
    tool_sum_1 = run1_by_role.get('tool_summarized', 0)
    tool_sum_2 = run2_by_role.get('tool_summarized', 0)

    summary_text = f"""
    TRANSFORMATION SUMMARY
    ═══════════════════════════════════════

    Total Tokens:
      {run1_name}: {total1:,}
      {run2_name}: {total2:,}
      Reduction: {reduction:.1f}%

    Tool (Original) Tokens:
      {run1_name}: {tool_orig_1:,}
      {run2_name}: {tool_orig_2:,}

    Tool (Summarized) Tokens:
      {run1_name}: {tool_sum_1:,}
      {run2_name}: {tool_sum_2:,}
      Reduction: {(1 - tool_sum_2/max(tool_sum_1, 1)) * 100:.1f}%

    Mean neg_log_prob (tool_summarized):
      {run1_name}: {np.mean(role_metrics1.get('tool_summarized', [np.nan])):.3f}
      {run2_name}: {np.mean(role_metrics2.get('tool_summarized', [np.nan])):.3f}

    Mean Entropy (tool_summarized):
      {run1_name}: {np.mean(role_entropy1.get('tool_summarized', [np.nan])):.3f}
      {run2_name}: {np.mean(role_entropy2.get('tool_summarized', [np.nan])):.3f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Context Transformation Analysis: {run1_name} → {run2_name}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved transformation summary to {output_path}")


def generate_all_plots(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    comparison_by_message: List[Dict[str, Any]],
    comparison_by_role: List[Dict[str, Any]],
    run1_name: str,
    run2_name: str,
    output_dir: Path,
    metric_names: List[str],
):
    """Generate all visualization plots.

    Args:
        run1_tokens: Token metrics from first run.
        run2_tokens: Token metrics from second run.
        comparison_by_message: Comparison data by message.
        comparison_by_role: Comparison data by role.
        run1_name: Name of first run.
        run2_name: Name of second run.
        output_dir: Directory for output plots.
        metric_names: List of metrics to plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Token reduction by role
    plot_token_reduction_by_role(
        run1_tokens, run2_tokens, run1_name, run2_name,
        output_dir / "token_reduction_by_role.png"
    )

    # 2. Comprehensive summary figure
    plot_transformation_summary(
        run1_tokens, run2_tokens, comparison_by_role,
        run1_name, run2_name,
        output_dir / "transformation_summary.png"
    )

    # 3. Metric comparison and scatter plots for key metrics
    key_metrics = ["neg_log_prob", "estimated_entropy", "topk_entropy"]
    for metric in key_metrics:
        if metric in metric_names:
            # Comparison by role
            plot_metric_comparison_by_role(
                run1_tokens, run2_tokens, run1_name, run2_name, metric,
                output_dir / f"{metric}_comparison_by_role.png"
            )

            # Scatter plot
            plot_metric_scatter(
                run1_tokens, run2_tokens, run1_name, run2_name, metric,
                output_dir / f"{metric}_scatter.png"
            )

            # Delta distribution
            plot_delta_distribution(
                comparison_by_message, run1_name, run2_name, metric,
                output_dir / f"{metric}_delta_distribution.png"
            )

    # 4. Position-metric plots by transform pair (original->original, original->summarized)
    print("\nGenerating position-metric plots by transform pair...")
    plot_transform_pair_position_metrics(
        run1_tokens,
        run2_tokens,
        run1_name,
        run2_name,
        key_metrics,
        output_dir / "by_transform_pair",
        roles=["assistant", "tool"],
    )

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics across different context configurations."
    )
    parser.add_argument(
        "--runs",
        nargs=2,
        required=True,
        help="Paths to two metrics CSV files to compare",
    )
    parser.add_argument(
        "--run-names",
        nargs=2,
        default=None,
        help="Names for the two runs (default: derived from file names)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compare (default: all common metrics)",
    )
    parser.add_argument(
        "--aggregate-by",
        choices=["message", "role"],
        default="message",
        help="Aggregation level: message (by original_uid) or role",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary statistics",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plots (default: same as output file directory)",
    )
    args = parser.parse_args()

    # Determine run names
    if args.run_names:
        run1_name, run2_name = args.run_names
    else:
        run1_name = Path(args.runs[0]).stem.replace("_metrics", "")
        run2_name = Path(args.runs[1]).stem.replace("_metrics", "")

    print(f"Comparing runs: {run1_name} vs {run2_name}")

    # Load metrics
    print(f"\nLoading: {args.runs[0]}")
    run1_tokens = load_metrics_csv(Path(args.runs[0]))
    print(f"  Loaded {len(run1_tokens)} tokens")

    print(f"Loading: {args.runs[1]}")
    run2_tokens = load_metrics_csv(Path(args.runs[1]))
    print(f"  Loaded {len(run2_tokens)} tokens")

    # Compare
    print(f"\nComparing by {args.aggregate_by}...")
    if args.aggregate_by == "message":
        comparison = compare_runs_by_message(
            run1_tokens,
            run2_tokens,
            run1_name,
            run2_name,
            args.metrics,
        )
    else:
        comparison = compare_runs_by_role(
            run1_tokens,
            run2_tokens,
            run1_name,
            run2_name,
            args.metrics,
        )

    # Save
    save_comparison_csv(comparison, Path(args.output))

    # Determine metrics used
    if args.metrics:
        metric_names = args.metrics
    else:
        # Extract from first row
        if comparison:
            metric_names = [
                k.replace("delta_", "")
                for k in comparison[0].keys()
                if k.startswith("delta_")
            ]
        else:
            metric_names = []

    # Filter out perplexity by default
    metric_names = [m for m in metric_names if m != "perplexity"]

    # Summary
    if not args.no_summary:
        print_summary(comparison, run1_name, run2_name, metric_names)

        # Transform pair analysis
        transform_analysis = analyze_by_transform_pair(
            run1_tokens, run2_tokens, run1_name, run2_name, metric_names
        )
        print_transform_pair_summary(transform_analysis, run1_name, run2_name)

    # Generate plots
    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.output).parent
        print(f"\nGenerating plots in {plot_dir}...")

        # Also need comparison by role for some plots
        comparison_by_role = compare_runs_by_role(
            run1_tokens, run2_tokens, run1_name, run2_name, args.metrics
        )

        generate_all_plots(
            run1_tokens=run1_tokens,
            run2_tokens=run2_tokens,
            comparison_by_message=comparison,
            comparison_by_role=comparison_by_role,
            run1_name=run1_name,
            run2_name=run2_name,
            output_dir=plot_dir,
            metric_names=metric_names,
        )


if __name__ == "__main__":
    main()
