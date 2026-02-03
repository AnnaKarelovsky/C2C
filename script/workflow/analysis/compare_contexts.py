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


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_token_reduction_by_role(
    run1_tokens: List[TokenMetrics],
    run2_tokens: List[TokenMetrics],
    run1_name: str,
    run2_name: str,
    output_path: Path,
):
    """Plot token count comparison by role.

    Shows how many tokens each role has before and after transformation.
    """
    import matplotlib.pyplot as plt

    # Aggregate tokens by role
    run1_by_role: Dict[str, int] = defaultdict(int)
    run2_by_role: Dict[str, int] = defaultdict(int)

    for token in run1_tokens:
        run1_by_role[token.role] += 1
    for token in run2_tokens:
        run2_by_role[token.role] += 1

    # Get all roles
    all_roles = sorted(set(run1_by_role.keys()) | set(run2_by_role.keys()))

    # Filter to main roles
    main_roles = [r for r in all_roles if r in ["system", "user", "assistant", "tool", "developer"]]
    if not main_roles:
        main_roles = all_roles[:5]

    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xticklabels(main_roles)
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

    # Add reduction percentage annotation for tool role
    if "tool" in main_roles:
        tool_idx = main_roles.index("tool")
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
):
    """Plot metric comparison by role as grouped bar chart."""
    import matplotlib.pyplot as plt

    # Aggregate metrics by role
    run1_by_role = aggregate_by_role(run1_tokens)
    run2_by_role = aggregate_by_role(run2_tokens)

    # Get all roles across all conversations
    role_metrics1: Dict[str, List[float]] = defaultdict(list)
    role_metrics2: Dict[str, List[float]] = defaultdict(list)

    for (conv_id, role), msg in run1_by_role.items():
        if metric_name in msg.metrics:
            role_metrics1[role].append(msg.metrics[metric_name])
    for (conv_id, role), msg in run2_by_role.items():
        if metric_name in msg.metrics:
            role_metrics2[role].append(msg.metrics[metric_name])

    # Filter to main roles
    main_roles = [r for r in ["system", "user", "assistant", "tool", "developer"]
                  if r in role_metrics1 or r in role_metrics2]

    if not main_roles:
        print(f"No data for {metric_name} by role")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xticklabels(main_roles)
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
):
    """Scatter plot comparing metrics between two runs, colored by role."""
    import matplotlib.pyplot as plt

    # Aggregate by message to align data points
    run1_by_msg = aggregate_by_message(run1_tokens)
    run2_by_msg = aggregate_by_message(run2_tokens)

    # Find common keys
    common_keys = set(run1_by_msg.keys()) & set(run2_by_msg.keys())

    if not common_keys:
        print(f"No common messages to compare for {metric_name}")
        return

    # Collect data points by role
    data_by_role: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for key in common_keys:
        msg1 = run1_by_msg[key]
        msg2 = run2_by_msg[key]

        val1 = msg1.metrics.get(metric_name)
        val2 = msg2.metrics.get(metric_name)

        if val1 is not None and val2 is not None and math.isfinite(val1) and math.isfinite(val2):
            data_by_role[msg1.role].append((val1, val2))

    if not data_by_role:
        print(f"No valid data for {metric_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'system': 'green', 'user': 'blue', 'assistant': 'orange', 'tool': 'red', 'developer': 'purple'}

    for role, points in data_by_role.items():
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
    """Create a comprehensive summary figure showing transformation effects."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12))

    # 1. Token count comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)

    run1_by_role: Dict[str, int] = defaultdict(int)
    run2_by_role: Dict[str, int] = defaultdict(int)
    for token in run1_tokens:
        run1_by_role[token.role] += 1
    for token in run2_tokens:
        run2_by_role[token.role] += 1

    main_roles = ["system", "user", "assistant", "tool"]
    x = np.arange(len(main_roles))
    width = 0.35

    run1_counts = [run1_by_role.get(r, 0) for r in main_roles]
    run2_counts = [run2_by_role.get(r, 0) for r in main_roles]

    ax1.bar(x - width/2, run1_counts, width, label=run1_name, color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, run2_counts, width, label=run2_name, color='coral', alpha=0.8)
    ax1.set_xlabel('Role')
    ax1.set_ylabel('Token Count')
    ax1.set_title('Token Count by Role')
    ax1.set_xticks(x)
    ax1.set_xticklabels(main_roles)
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

    for token in run1_tokens:
        if "neg_log_prob" in token.metrics:
            role_metrics1[token.role].append(token.metrics["neg_log_prob"])
    for token in run2_tokens:
        if "neg_log_prob" in token.metrics:
            role_metrics2[token.role].append(token.metrics["neg_log_prob"])

    run1_means = [np.mean(role_metrics1.get(r, [np.nan])) for r in main_roles]
    run2_means = [np.mean(role_metrics2.get(r, [np.nan])) for r in main_roles]

    ax2.bar(x - width/2, run1_means, width, label=run1_name, color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, run2_means, width, label=run2_name, color='coral', alpha=0.8)
    ax2.set_xlabel('Role')
    ax2.set_ylabel('neg_log_prob')
    ax2.set_title('Negative Log Probability by Role')
    ax2.set_xticks(x)
    ax2.set_xticklabels(main_roles)
    ax2.legend()

    # 3. Entropy comparison (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)

    role_entropy1: Dict[str, List[float]] = defaultdict(list)
    role_entropy2: Dict[str, List[float]] = defaultdict(list)

    for token in run1_tokens:
        if "estimated_entropy" in token.metrics:
            role_entropy1[token.role].append(token.metrics["estimated_entropy"])
    for token in run2_tokens:
        if "estimated_entropy" in token.metrics:
            role_entropy2[token.role].append(token.metrics["estimated_entropy"])

    run1_entropy = [np.mean(role_entropy1.get(r, [np.nan])) for r in main_roles]
    run2_entropy = [np.mean(role_entropy2.get(r, [np.nan])) for r in main_roles]

    ax3.bar(x - width/2, run1_entropy, width, label=run1_name, color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, run2_entropy, width, label=run2_name, color='coral', alpha=0.8)
    ax3.set_xlabel('Role')
    ax3.set_ylabel('Estimated Entropy')
    ax3.set_title('Entropy by Role')
    ax3.set_xticks(x)
    ax3.set_xticklabels(main_roles)
    ax3.legend()

    # 4. Summary statistics (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = f"""
    TRANSFORMATION SUMMARY
    ═══════════════════════════════════════

    Total Tokens:
      {run1_name}: {total1:,}
      {run2_name}: {total2:,}
      Reduction: {reduction:.1f}%

    Tool Tokens:
      {run1_name}: {run1_by_role.get('tool', 0):,}
      {run2_name}: {run2_by_role.get('tool', 0):,}
      Reduction: {(1 - run2_by_role.get('tool', 0)/max(run1_by_role.get('tool', 1), 1)) * 100:.1f}%

    Mean neg_log_prob (tool):
      {run1_name}: {np.mean(role_metrics1.get('tool', [np.nan])):.3f}
      {run2_name}: {np.mean(role_metrics2.get('tool', [np.nan])):.3f}
      Change: {np.mean(role_metrics2.get('tool', [0])) - np.mean(role_metrics1.get('tool', [0])):+.3f}

    Mean Entropy (tool):
      {run1_name}: {np.mean(role_entropy1.get('tool', [np.nan])):.3f}
      {run2_name}: {np.mean(role_entropy2.get('tool', [np.nan])):.3f}
      Change: {np.mean(role_entropy2.get('tool', [0])) - np.mean(role_entropy1.get('tool', [0])):+.3f}
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

    # Summary
    if not args.no_summary:
        print_summary(comparison, run1_name, run2_name, metric_names)

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
