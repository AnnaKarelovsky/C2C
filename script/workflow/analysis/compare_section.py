#!/usr/bin/env python3
"""
Compare metrics across runs by section index.

Takes multiple metrics CSVs, plots metrics by section_idx (x-axis) with each
CSV as a legend line, and includes grouped bar plots by content_type.

Usage:
    # Compare two runs
    python script/workflow/analysis/compare_section.py \
        --runs metrics_a.csv metrics_b.csv \
        --output /tmp/section_test \
        --plot

    # Compare with custom names and specific metrics
    python script/workflow/analysis/compare_section.py \
        --runs baseline.csv summarized.csv context.csv \
        --run-names baseline summarized context \
        --metrics neg_log_prob estimated_entropy \
        --output comparison_dir \
        --plot --no-summary
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Peer import: reuse load_metrics_csv and TokenMetrics from compare_contexts.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_contexts import TokenMetrics, load_metrics_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Color palette for runs (tab10-based)
# ---------------------------------------------------------------------------
RUN_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Preferred ordering for content types
CONTENT_TYPE_ORDER = ["text", "reasoning", "tool_call"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SectionAgg:
    """Aggregated metrics for a single section index across conversations."""

    section_idx: int
    token_count: int
    num_conversations: int
    dominant_content_type: str
    metric_means: Dict[str, float] = field(default_factory=dict)
    metric_stds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContentTypeAgg:
    """Aggregated metrics for a single content type across conversations."""

    content_type: str
    token_count: int
    num_conversations: int
    metric_means: Dict[str, float] = field(default_factory=dict)
    metric_stds: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_by_section(tokens: List[TokenMetrics]) -> Dict[int, SectionAgg]:
    """Aggregate token metrics by section_idx (two-level).

    Level 1: Group tokens by (conversation_id, section_idx), compute per-metric
             mean within each group.
    Level 2: Group those per-conversation means by section_idx, compute mean +
             std across conversations.

    Returns:
        Dict mapping section_idx to SectionAgg.
    """
    # Level 1: per (conversation_id, section_idx)
    groups: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        if token.section_idx < 0:
            continue
        groups[(token.conversation_id, token.section_idx)].append(token)

    # Compute per-group means and collect content_type counts
    # Key: (conv_id, section_idx) -> {metric: mean}
    group_means: Dict[Tuple[str, int], Dict[str, float]] = {}
    group_token_counts: Dict[Tuple[str, int], int] = {}
    group_content_types: Dict[Tuple[str, int], Dict[str, int]] = {}

    for key, token_list in groups.items():
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)
        ct_counts: Dict[str, int] = defaultdict(int)

        for token in token_list:
            ct_counts[token.content_type] += 1
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        means = {}
        for m, total in metric_sums.items():
            if metric_counts[m] > 0:
                means[m] = total / metric_counts[m]
        group_means[key] = means
        group_token_counts[key] = len(token_list)
        group_content_types[key] = dict(ct_counts)

    # Level 2: aggregate across conversations per section_idx
    section_groups: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for (conv_id, sidx) in group_means:
        section_groups[sidx].append((conv_id, sidx))

    results: Dict[int, SectionAgg] = {}
    for sidx, keys in sorted(section_groups.items()):
        # Collect metric values across conversations
        metric_values: Dict[str, List[float]] = defaultdict(list)
        total_tokens = 0
        ct_totals: Dict[str, int] = defaultdict(int)

        for key in keys:
            total_tokens += group_token_counts[key]
            for m, v in group_means[key].items():
                metric_values[m].append(v)
            for ct, cnt in group_content_types[key].items():
                ct_totals[ct] += cnt

        # Dominant content type
        dominant_ct = max(ct_totals, key=ct_totals.get) if ct_totals else "unknown"

        means = {m: float(np.mean(vals)) for m, vals in metric_values.items()}
        stds = {m: float(np.std(vals)) for m, vals in metric_values.items()}

        results[sidx] = SectionAgg(
            section_idx=sidx,
            token_count=total_tokens,
            num_conversations=len(keys),
            dominant_content_type=dominant_ct,
            metric_means=means,
            metric_stds=stds,
        )

    return results


def aggregate_by_content_type(tokens: List[TokenMetrics]) -> Dict[str, ContentTypeAgg]:
    """Aggregate token metrics by content_type (two-level).

    Level 1: Group tokens by (conversation_id, content_type), compute per-metric
             mean within each group.
    Level 2: Group those per-conversation means by content_type, compute mean +
             std across conversations.

    Returns:
        Dict mapping content_type to ContentTypeAgg.
    """
    # Level 1: per (conversation_id, content_type)
    groups: Dict[Tuple[str, str], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        groups[(token.conversation_id, token.content_type)].append(token)

    group_means: Dict[Tuple[str, str], Dict[str, float]] = {}
    group_token_counts: Dict[Tuple[str, str], int] = {}

    for key, token_list in groups.items():
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)

        for token in token_list:
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        means = {}
        for m, total in metric_sums.items():
            if metric_counts[m] > 0:
                means[m] = total / metric_counts[m]
        group_means[key] = means
        group_token_counts[key] = len(token_list)

    # Level 2: aggregate across conversations per content_type
    ct_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for (conv_id, ct) in group_means:
        ct_groups[ct].append((conv_id, ct))

    results: Dict[str, ContentTypeAgg] = {}
    for ct, keys in sorted(ct_groups.items()):
        metric_values: Dict[str, List[float]] = defaultdict(list)
        total_tokens = 0

        for key in keys:
            total_tokens += group_token_counts[key]
            for m, v in group_means[key].items():
                metric_values[m].append(v)

        means = {m: float(np.mean(vals)) for m, vals in metric_values.items()}
        stds = {m: float(np.std(vals)) for m, vals in metric_values.items()}

        results[ct] = ContentTypeAgg(
            content_type=ct,
            token_count=total_tokens,
            num_conversations=len(keys),
            metric_means=means,
            metric_stds=stds,
        )

    return results


def aggregate_by_norm_section(
    tokens: List[TokenMetrics], n_bins: int = 20
) -> Dict[int, SectionAgg]:
    """Aggregate token metrics by normalized section index (two-level).

    Each conversation's section indices are normalized to [0, 1] based on
    its max section_idx, then quantized into *n_bins* bins.

    Args:
        tokens: List of token-level metrics.
        n_bins: Number of bins for the [0, 1] range.

    Returns:
        Dict mapping bin_idx (0 .. n_bins-1) to SectionAgg.
    """
    # Find max section_idx per conversation
    conv_max_sidx: Dict[str, int] = {}
    for token in tokens:
        if token.section_idx < 0:
            continue
        cid = token.conversation_id
        conv_max_sidx[cid] = max(conv_max_sidx.get(cid, 0), token.section_idx)

    # Level 1: group tokens by (conversation_id, bin_idx)
    groups: Dict[Tuple[str, int], List[TokenMetrics]] = defaultdict(list)
    for token in tokens:
        if token.section_idx < 0:
            continue
        cid = token.conversation_id
        max_s = conv_max_sidx.get(cid, 0)
        if max_s > 0:
            norm = token.section_idx / max_s
            bin_idx = min(int(norm * n_bins), n_bins - 1)
        else:
            bin_idx = 0
        groups[(cid, bin_idx)].append(token)

    # Compute per-group means and collect content_type counts
    group_means: Dict[Tuple[str, int], Dict[str, float]] = {}
    group_token_counts: Dict[Tuple[str, int], int] = {}
    group_content_types: Dict[Tuple[str, int], Dict[str, int]] = {}

    for key, token_list in groups.items():
        metric_sums: Dict[str, float] = defaultdict(float)
        metric_counts: Dict[str, int] = defaultdict(int)
        ct_counts: Dict[str, int] = defaultdict(int)

        for token in token_list:
            ct_counts[token.content_type] += 1
            for metric_name, value in token.metrics.items():
                if math.isfinite(value):
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1

        means = {}
        for m, total in metric_sums.items():
            if metric_counts[m] > 0:
                means[m] = total / metric_counts[m]
        group_means[key] = means
        group_token_counts[key] = len(token_list)
        group_content_types[key] = dict(ct_counts)

    # Level 2: aggregate across conversations per bin_idx
    bin_groups: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for (conv_id, bin_idx) in group_means:
        bin_groups[bin_idx].append((conv_id, bin_idx))

    results: Dict[int, SectionAgg] = {}
    for bin_idx, keys in sorted(bin_groups.items()):
        metric_values: Dict[str, List[float]] = defaultdict(list)
        total_tokens = 0
        ct_totals: Dict[str, int] = defaultdict(int)

        for key in keys:
            total_tokens += group_token_counts[key]
            for m, v in group_means[key].items():
                metric_values[m].append(v)
            for ct, cnt in group_content_types[key].items():
                ct_totals[ct] += cnt

        dominant_ct = max(ct_totals, key=ct_totals.get) if ct_totals else "unknown"
        means = {m: float(np.mean(vals)) for m, vals in metric_values.items()}
        stds = {m: float(np.std(vals)) for m, vals in metric_values.items()}

        results[bin_idx] = SectionAgg(
            section_idx=bin_idx,
            token_count=total_tokens,
            num_conversations=len(keys),
            dominant_content_type=dominant_ct,
            metric_means=means,
            metric_stds=stds,
        )

    return results


# ---------------------------------------------------------------------------
# Discover metric names
# ---------------------------------------------------------------------------
def discover_metrics(tokens: List[TokenMetrics], exclude: Optional[List[str]] = None) -> List[str]:
    """Discover available metric names from tokens, excluding specified ones."""
    all_metrics: set = set()
    for token in tokens:
        all_metrics.update(token.metrics.keys())
    if exclude:
        all_metrics -= set(exclude)
    return sorted(all_metrics)


# ---------------------------------------------------------------------------
# Ordered content types
# ---------------------------------------------------------------------------
def ordered_content_types(ct_aggs: Dict[str, ContentTypeAgg]) -> List[str]:
    """Return content types in preferred order: known first, then alpha."""
    known = [ct for ct in CONTENT_TYPE_ORDER if ct in ct_aggs]
    extra = sorted(set(ct_aggs.keys()) - set(CONTENT_TYPE_ORDER))
    return known + extra


# ---------------------------------------------------------------------------
# Plotting: metrics by section (shared helpers + public API)
# ---------------------------------------------------------------------------
def _plot_section_grid(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    output_path: Path,
    run_names: List[str],
    max_key: Optional[int] = None,
    min_conversations: int = 2,
    x_transform: Optional[Any] = None,
    xlabel: str = "Section Index",
    suptitle: str = "Metrics by Section Index",
):
    """Internal: grid of subplots (one per metric, up to 3 columns).

    Args:
        x_transform: If given, callable mapping raw key -> x-value.
        xlabel: Label for the x-axis.
        suptitle: Figure super-title.
    """
    import matplotlib.pyplot as plt

    ncols = min(3, len(metric_names))
    nrows = math.ceil(len(metric_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, metric in enumerate(metric_names):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        for run_idx, run_name in enumerate(run_names):
            agg = run_aggs[run_name]
            color = RUN_COLORS[run_idx % len(RUN_COLORS)]

            x_values: List[float] = []
            means: List[float] = []
            stds: List[float] = []

            for sidx in sorted(agg.keys()):
                if max_key is not None and sidx > max_key:
                    break
                sa = agg[sidx]
                if sa.num_conversations < min_conversations:
                    continue
                if metric not in sa.metric_means:
                    continue
                x_values.append(x_transform(sidx) if x_transform else sidx)
                means.append(sa.metric_means[metric])
                stds.append(sa.metric_stds.get(metric, 0.0))

            if not x_values:
                continue

            xs = np.array(x_values)
            ys = np.array(means)
            es = np.array(stds)

            ax.plot(xs, ys, color=color, linewidth=2, label=run_name)
            ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Y-axis percentile clipping
        all_means: List[float] = []
        for run_name in run_names:
            agg = run_aggs[run_name]
            for sidx, sa in agg.items():
                if max_key is not None and sidx > max_key:
                    continue
                if sa.num_conversations < min_conversations:
                    continue
                if metric in sa.metric_means:
                    all_means.append(sa.metric_means[metric])
        if all_means:
            sorted_means = np.sort(all_means)
            clip_idx = min(len(sorted_means) - 1, int(len(sorted_means) * 0.99))
            y_max = sorted_means[clip_idx]
            y_min = sorted_means[0]
            margin = (y_max - y_min) * 0.05
            ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # Hide unused axes
    for idx in range(len(metric_names), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved section grid plot to {output_path}")


def _plot_single_section(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_name: str,
    output_path: Path,
    run_names: List[str],
    max_key: Optional[int] = None,
    min_conversations: int = 2,
    x_transform: Optional[Any] = None,
    xlabel: str = "Section Index",
):
    """Internal: single metric plot with all runs overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for run_idx, run_name in enumerate(run_names):
        agg = run_aggs[run_name]
        color = RUN_COLORS[run_idx % len(RUN_COLORS)]

        x_values: List[float] = []
        means: List[float] = []
        stds: List[float] = []

        for sidx in sorted(agg.keys()):
            if max_key is not None and sidx > max_key:
                break
            sa = agg[sidx]
            if sa.num_conversations < min_conversations:
                continue
            if metric_name not in sa.metric_means:
                continue
            x_values.append(x_transform(sidx) if x_transform else sidx)
            means.append(sa.metric_means[metric_name])
            stds.append(sa.metric_stds.get(metric_name, 0.0))

        if not x_values:
            continue

        xs = np.array(x_values)
        ys = np.array(means)
        es = np.array(stds)

        ax.plot(xs, ys, color=color, linewidth=2, label=run_name)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by {xlabel}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric_name} section plot to {output_path}")


# --- Public wrappers: raw section index ---


def plot_all_metrics_by_section(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    output_path: Path,
    run_names: List[str],
    max_section_idx: Optional[int] = None,
    min_conversations: int = 2,
):
    """Grid of subplots (one per metric) with raw section_idx on x-axis."""
    _plot_section_grid(
        run_aggs, metric_names, output_path, run_names,
        max_key=max_section_idx,
        min_conversations=min_conversations,
    )


def plot_single_metric_by_section(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_name: str,
    output_path: Path,
    run_names: List[str],
    max_section_idx: Optional[int] = None,
    min_conversations: int = 2,
):
    """Single metric plot with raw section_idx on x-axis."""
    _plot_single_section(
        run_aggs, metric_name, output_path, run_names,
        max_key=max_section_idx,
        min_conversations=min_conversations,
    )


# --- Public wrappers: normalized section position ---


def plot_all_metrics_by_norm_section(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    output_path: Path,
    run_names: List[str],
    n_bins: int = 20,
    min_conversations: int = 2,
):
    """Grid of subplots with normalized [0, 1] section position on x-axis."""
    _plot_section_grid(
        run_aggs, metric_names, output_path, run_names,
        min_conversations=min_conversations,
        x_transform=lambda b: (b + 0.5) / n_bins,
        xlabel="Normalized Section Position",
        suptitle="Metrics by Normalized Section Position",
    )


def plot_single_metric_by_norm_section(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_name: str,
    output_path: Path,
    run_names: List[str],
    n_bins: int = 20,
    min_conversations: int = 2,
):
    """Single metric plot with normalized [0, 1] section position on x-axis."""
    _plot_single_section(
        run_aggs, metric_name, output_path, run_names,
        min_conversations=min_conversations,
        x_transform=lambda b: (b + 0.5) / n_bins,
        xlabel="Normalized Section Position",
    )


# ---------------------------------------------------------------------------
# Plotting: content type bars
# ---------------------------------------------------------------------------
def plot_content_type_bars(
    run_ct_aggs: Dict[str, Dict[str, ContentTypeAgg]],
    output_path: Path,
    run_names: List[str],
):
    """3 subplots side-by-side: token count, neg_log_prob, estimated_entropy.

    X-axis: content types (grouped bars per run).
    """
    import matplotlib.pyplot as plt

    # Determine content type ordering from all runs
    all_cts: set = set()
    for ct_agg in run_ct_aggs.values():
        all_cts.update(ct_agg.keys())
    known = [ct for ct in CONTENT_TYPE_ORDER if ct in all_cts]
    extra = sorted(all_cts - set(CONTENT_TYPE_ORDER))
    ct_order = known + extra

    if not ct_order:
        print("No content type data to plot")
        return

    num_runs = len(run_names)
    bar_width = 0.8 / num_runs

    subplot_specs = [
        ("Token Count", "token_count", False),
        ("Neg Log Prob", "neg_log_prob", True),
        ("Estimated Entropy", "estimated_entropy", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 6))

    for ax, (title, key, is_metric) in zip(axes, subplot_specs):
        for run_idx, run_name in enumerate(run_names):
            ct_agg = run_ct_aggs[run_name]
            color = RUN_COLORS[run_idx % len(RUN_COLORS)]
            offset = (run_idx - num_runs / 2 + 0.5) * bar_width

            x_positions = np.arange(len(ct_order)) + offset
            values = []
            errors = []

            for ct in ct_order:
                if ct in ct_agg:
                    if is_metric:
                        values.append(ct_agg[ct].metric_means.get(key, 0.0))
                        errors.append(ct_agg[ct].metric_stds.get(key, 0.0))
                    else:
                        values.append(ct_agg[ct].token_count)
                        errors.append(0.0)
                else:
                    values.append(0.0)
                    errors.append(0.0)

            if is_metric:
                ax.bar(
                    x_positions, values, bar_width, yerr=errors,
                    label=run_name, color=color, alpha=0.8, capsize=3,
                )
            else:
                ax.bar(
                    x_positions, values, bar_width,
                    label=run_name, color=color, alpha=0.8,
                )

        ax.set_xlabel("Content Type")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(ct_order)))
        ax.set_xticklabels(ct_order, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Content Type Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved content type bar plot to {output_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def save_section_comparison_csv(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    run_names: List[str],
    output_path: Path,
):
    """Save section-level comparison CSV.

    Columns: section_idx, dominant_content_type,
             {run}_tokens, {run}_n_conv, {run}_{metric}_mean, {run}_{metric}_std, ...
    """
    # Collect all section indices
    all_sidx: set = set()
    for agg in run_aggs.values():
        all_sidx.update(agg.keys())

    if not all_sidx:
        print("No section data to save")
        return

    # Build field names
    fieldnames = ["section_idx", "dominant_content_type"]
    for rn in run_names:
        fieldnames.append(f"{rn}_tokens")
        fieldnames.append(f"{rn}_n_conv")
        for m in metric_names:
            fieldnames.append(f"{rn}_{m}_mean")
            fieldnames.append(f"{rn}_{m}_std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sidx in sorted(all_sidx):
            row: Dict[str, Any] = {"section_idx": sidx}

            # Use dominant content type from first run that has this section
            dom_ct = "unknown"
            for rn in run_names:
                if sidx in run_aggs[rn]:
                    dom_ct = run_aggs[rn][sidx].dominant_content_type
                    break
            row["dominant_content_type"] = dom_ct

            for rn in run_names:
                sa = run_aggs[rn].get(sidx)
                if sa:
                    row[f"{rn}_tokens"] = sa.token_count
                    row[f"{rn}_n_conv"] = sa.num_conversations
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = f"{sa.metric_means.get(m, '')}"
                        row[f"{rn}_{m}_std"] = f"{sa.metric_stds.get(m, '')}"
                else:
                    row[f"{rn}_tokens"] = 0
                    row[f"{rn}_n_conv"] = 0
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = ""
                        row[f"{rn}_{m}_std"] = ""

            writer.writerow(row)

    print(f"Saved section comparison to {output_path}")


def save_norm_section_comparison_csv(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    run_names: List[str],
    output_path: Path,
    n_bins: int = 20,
):
    """Save normalized-section-level comparison CSV.

    Columns: bin_idx, norm_position, dominant_content_type,
             {run}_tokens, {run}_n_conv, {run}_{metric}_mean, {run}_{metric}_std, ...
    """
    all_bins: set = set()
    for agg in run_aggs.values():
        all_bins.update(agg.keys())

    if not all_bins:
        print("No norm-section data to save")
        return

    fieldnames = ["bin_idx", "norm_position", "dominant_content_type"]
    for rn in run_names:
        fieldnames.append(f"{rn}_tokens")
        fieldnames.append(f"{rn}_n_conv")
        for m in metric_names:
            fieldnames.append(f"{rn}_{m}_mean")
            fieldnames.append(f"{rn}_{m}_std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bidx in sorted(all_bins):
            row: Dict[str, Any] = {
                "bin_idx": bidx,
                "norm_position": f"{(bidx + 0.5) / n_bins:.4f}",
            }

            dom_ct = "unknown"
            for rn in run_names:
                if bidx in run_aggs[rn]:
                    dom_ct = run_aggs[rn][bidx].dominant_content_type
                    break
            row["dominant_content_type"] = dom_ct

            for rn in run_names:
                sa = run_aggs[rn].get(bidx)
                if sa:
                    row[f"{rn}_tokens"] = sa.token_count
                    row[f"{rn}_n_conv"] = sa.num_conversations
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = f"{sa.metric_means.get(m, '')}"
                        row[f"{rn}_{m}_std"] = f"{sa.metric_stds.get(m, '')}"
                else:
                    row[f"{rn}_tokens"] = 0
                    row[f"{rn}_n_conv"] = 0
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = ""
                        row[f"{rn}_{m}_std"] = ""

            writer.writerow(row)

    print(f"Saved norm-section comparison to {output_path}")


def save_content_type_comparison_csv(
    run_ct_aggs: Dict[str, Dict[str, ContentTypeAgg]],
    metric_names: List[str],
    run_names: List[str],
    output_path: Path,
):
    """Save content-type-level comparison CSV.

    Columns: content_type, {run}_tokens, {run}_n_conv,
             {run}_{metric}_mean, {run}_{metric}_std, ...
    """
    all_cts: set = set()
    for ct_agg in run_ct_aggs.values():
        all_cts.update(ct_agg.keys())

    if not all_cts:
        print("No content type data to save")
        return

    known = [ct for ct in CONTENT_TYPE_ORDER if ct in all_cts]
    extra = sorted(all_cts - set(CONTENT_TYPE_ORDER))
    ct_order = known + extra

    fieldnames = ["content_type"]
    for rn in run_names:
        fieldnames.append(f"{rn}_tokens")
        fieldnames.append(f"{rn}_n_conv")
        for m in metric_names:
            fieldnames.append(f"{rn}_{m}_mean")
            fieldnames.append(f"{rn}_{m}_std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ct in ct_order:
            row: Dict[str, Any] = {"content_type": ct}
            for rn in run_names:
                ca = run_ct_aggs[rn].get(ct)
                if ca:
                    row[f"{rn}_tokens"] = ca.token_count
                    row[f"{rn}_n_conv"] = ca.num_conversations
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = f"{ca.metric_means.get(m, '')}"
                        row[f"{rn}_{m}_std"] = f"{ca.metric_stds.get(m, '')}"
                else:
                    row[f"{rn}_tokens"] = 0
                    row[f"{rn}_n_conv"] = 0
                    for m in metric_names:
                        row[f"{rn}_{m}_mean"] = ""
                        row[f"{rn}_{m}_std"] = ""

            writer.writerow(row)

    print(f"Saved content type comparison to {output_path}")


# ---------------------------------------------------------------------------
# Text summaries
# ---------------------------------------------------------------------------
def print_section_summary(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    run_names: List[str],
    max_section_idx: Optional[int] = None,
):
    """Print section-level summary table to console."""
    print("\n" + "=" * 70)
    print("SECTION SUMMARY")
    print("=" * 70)

    # Collect all section indices
    all_sidx: set = set()
    for agg in run_aggs.values():
        all_sidx.update(agg.keys())

    for sidx in sorted(all_sidx):
        if max_section_idx is not None and sidx > max_section_idx:
            break

        print(f"\n--- Section {sidx} ---")
        for rn in run_names:
            sa = run_aggs[rn].get(sidx)
            if sa is None:
                print(f"  {rn}: (no data)")
                continue
            parts = [f"tokens={sa.token_count:,}", f"convs={sa.num_conversations}"]
            parts.append(f"type={sa.dominant_content_type}")
            for m in metric_names:
                if m in sa.metric_means:
                    parts.append(f"{m}={sa.metric_means[m]:.4f}±{sa.metric_stds.get(m, 0):.4f}")
            print(f"  {rn}: {', '.join(parts)}")

    print("\n" + "=" * 70)


def print_norm_section_summary(
    run_aggs: Dict[str, Dict[int, SectionAgg]],
    metric_names: List[str],
    run_names: List[str],
    n_bins: int = 20,
):
    """Print normalized-section-level summary table to console."""
    print("\n" + "=" * 70)
    print("NORMALIZED SECTION SUMMARY")
    print("=" * 70)

    all_bins: set = set()
    for agg in run_aggs.values():
        all_bins.update(agg.keys())

    for bidx in sorted(all_bins):
        norm_pos = (bidx + 0.5) / n_bins
        print(f"\n--- Bin {bidx} (pos={norm_pos:.2f}) ---")
        for rn in run_names:
            sa = run_aggs[rn].get(bidx)
            if sa is None:
                print(f"  {rn}: (no data)")
                continue
            parts = [f"tokens={sa.token_count:,}", f"convs={sa.num_conversations}"]
            parts.append(f"type={sa.dominant_content_type}")
            for m in metric_names:
                if m in sa.metric_means:
                    parts.append(f"{m}={sa.metric_means[m]:.4f}±{sa.metric_stds.get(m, 0):.4f}")
            print(f"  {rn}: {', '.join(parts)}")

    print("\n" + "=" * 70)


def print_content_type_summary(
    run_ct_aggs: Dict[str, Dict[str, ContentTypeAgg]],
    metric_names: List[str],
    run_names: List[str],
):
    """Print content-type-level summary table to console."""
    print("\n" + "=" * 70)
    print("CONTENT TYPE SUMMARY")
    print("=" * 70)

    all_cts: set = set()
    for ct_agg in run_ct_aggs.values():
        all_cts.update(ct_agg.keys())

    known = [ct for ct in CONTENT_TYPE_ORDER if ct in all_cts]
    extra = sorted(all_cts - set(CONTENT_TYPE_ORDER))
    ct_order = known + extra

    for ct in ct_order:
        print(f"\n--- {ct} ---")
        for rn in run_names:
            ca = run_ct_aggs[rn].get(ct)
            if ca is None:
                print(f"  {rn}: (no data)")
                continue
            parts = [f"tokens={ca.token_count:,}", f"convs={ca.num_conversations}"]
            for m in metric_names:
                if m in ca.metric_means:
                    parts.append(f"{m}={ca.metric_means[m]:.4f}±{ca.metric_stds.get(m, 0):.4f}")
            print(f"  {rn}: {', '.join(parts)}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics across runs by section index."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="2+ metrics CSV files to compare",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="Legend names for runs (default: derived from filenames)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot (default: all minus perplexity)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plots (default: --output dir)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip text summary",
    )
    parser.add_argument(
        "--max-section-idx",
        type=int,
        default=None,
        help="Cap x-axis at this section index",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of bins for normalized section position (default: 20)",
    )
    args = parser.parse_args()

    if len(args.runs) < 2:
        parser.error("Need at least 2 runs to compare")

    # Determine run names
    if args.run_names:
        if len(args.run_names) != len(args.runs):
            parser.error("Number of --run-names must match number of --runs")
        run_names = args.run_names
    else:
        run_names = [
            Path(p).stem.replace("_metrics", "") for p in args.runs
        ]

    output_dir = Path(args.output)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir

    # Load all runs
    all_tokens: Dict[str, List[TokenMetrics]] = {}
    for run_path, run_name in zip(args.runs, run_names):
        print(f"Loading: {run_path}")
        tokens = load_metrics_csv(Path(run_path))
        print(f"  Loaded {len(tokens)} tokens")
        all_tokens[run_name] = tokens

    # Determine metrics
    if args.metrics:
        metric_names = args.metrics
    else:
        # Discover from first run, exclude perplexity
        first_tokens = all_tokens[run_names[0]]
        metric_names = discover_metrics(first_tokens, exclude=["perplexity"])

    print(f"\nMetrics: {', '.join(metric_names)}")

    # Aggregate
    n_bins = args.n_bins

    print("\nAggregating by section...")
    run_aggs: Dict[str, Dict[int, SectionAgg]] = {}
    for rn in run_names:
        run_aggs[rn] = aggregate_by_section(all_tokens[rn])

    print(f"Aggregating by normalized section ({n_bins} bins)...")
    run_norm_aggs: Dict[str, Dict[int, SectionAgg]] = {}
    for rn in run_names:
        run_norm_aggs[rn] = aggregate_by_norm_section(all_tokens[rn], n_bins=n_bins)

    print("Aggregating by content type...")
    run_ct_aggs: Dict[str, Dict[str, ContentTypeAgg]] = {}
    for rn in run_names:
        run_ct_aggs[rn] = aggregate_by_content_type(all_tokens[rn])

    # Print summaries
    if not args.no_summary:
        print_section_summary(run_aggs, metric_names, run_names, args.max_section_idx)
        print_norm_section_summary(run_norm_aggs, metric_names, run_names, n_bins)
        print_content_type_summary(run_ct_aggs, metric_names, run_names)

    # Save CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    save_section_comparison_csv(
        run_aggs, metric_names, run_names,
        output_dir / "section_comparison.csv",
    )
    save_norm_section_comparison_csv(
        run_norm_aggs, metric_names, run_names,
        output_dir / "norm_section_comparison.csv",
        n_bins=n_bins,
    )
    save_content_type_comparison_csv(
        run_ct_aggs, metric_names, run_names,
        output_dir / "content_type_comparison.csv",
    )

    # Generate plots
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots in {plot_dir}...")

        # --- Raw section index plots ---
        plot_all_metrics_by_section(
            run_aggs, metric_names,
            plot_dir / "all_metrics_by_section.png",
            run_names,
            max_section_idx=args.max_section_idx,
        )

        for metric in metric_names:
            plot_single_metric_by_section(
                run_aggs, metric,
                plot_dir / f"{metric}_by_section.png",
                run_names,
                max_section_idx=args.max_section_idx,
            )

        # --- Normalized section position plots ---
        plot_all_metrics_by_norm_section(
            run_norm_aggs, metric_names,
            plot_dir / "all_metrics_by_norm_section.png",
            run_names,
            n_bins=n_bins,
        )

        for metric in metric_names:
            plot_single_metric_by_norm_section(
                run_norm_aggs, metric,
                plot_dir / f"{metric}_by_norm_section.png",
                run_names,
                n_bins=n_bins,
            )

        # --- Content type bars ---
        plot_content_type_bars(
            run_ct_aggs,
            plot_dir / "content_type_bars.png",
            run_names,
        )

        print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
