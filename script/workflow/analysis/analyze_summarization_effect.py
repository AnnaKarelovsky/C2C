#!/usr/bin/env python3
"""
Analyze the effect of tool-call summarization on model metrics.

Compares a "summarized" run (full_summarized_full_N: last N tool results kept
in full, earlier ones summarized) against a "standard" run (all original).

Analyses performed:
  1. Within-run: metrics of summarized vs original tool sections
  2. Within-run: metrics of assistant turns following each zone
  3. Cross-run:  same assistant turns under standard vs summarized context
  4. Figures:    bar charts, progression plots, cross-run deltas

Usage:
    python script/workflow/analysis/analyze_summarization_effect.py \
        --summarized local/analysis/gpt_oss/120b_to_120b/full_summarized_full_4/baseline_metrics.csv \
        --standard   local/analysis/gpt_oss/120b_to_120b/standard/baseline_metrics.csv \
        --keep-last 4 \
        --output     local/analysis/gpt_oss/120b_to_120b/summarization_effect
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_contexts import TokenMetrics, load_metrics_csv  # noqa: E402

METRICS = ["neg_log_prob", "estimated_entropy", "topk_entropy"]
METRIC_LABELS = {
    "neg_log_prob": "Neg Log Prob",
    "estimated_entropy": "Estimated Entropy",
    "topk_entropy": "Top-K Entropy",
}

# Colors
C_SUMMARIZED = "#e74c3c"  # red
C_ORIGINAL = "#2ca02c"  # green
C_STANDARD = "#1f77b4"  # blue
C_SUMRUN = "#ff7f0e"  # orange (summarized run)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def classify_tool_sections(
    tokens: List[TokenMetrics], keep_last: int = 4
) -> Dict[str, Dict[int, str]]:
    """Classify each tool section as 'summarized' or 'original'.

    The last *keep_last* tool sections per conversation are 'original';
    earlier ones are 'summarized'.

    Returns: {conversation_id: {section_idx: 'summarized'|'original'}}
    """
    # Collect tool section indices per conversation
    conv_tool_sections: Dict[str, set] = defaultdict(set)
    for t in tokens:
        if t.role == "tool":
            conv_tool_sections[t.conversation_id].add(t.section_idx)

    result: Dict[str, Dict[int, str]] = {}
    for conv_id, sections in conv_tool_sections.items():
        ordered = sorted(sections)
        n = len(ordered)
        mapping = {}
        for i, sidx in enumerate(ordered):
            mapping[sidx] = "original" if i >= n - keep_last else "summarized"
        result[conv_id] = mapping
    return result


def label_section(
    token: TokenMetrics,
    tool_classes: Dict[str, Dict[int, str]],
) -> Optional[str]:
    """Return zone label for a token based on its role and context.

    - tool tokens      → 'tool_summarized' or 'tool_original'
    - assistant tokens  → 'asst_after_summarized' or 'asst_after_original'
      (determined by the immediately preceding tool section)
    - others           → None (excluded)
    """
    conv = tool_classes.get(token.conversation_id, {})
    if not conv:
        return None

    if token.role == "tool":
        cls = conv.get(token.section_idx)
        return f"tool_{cls}" if cls else None

    if token.role == "assistant":
        # Find the nearest preceding tool section
        tool_sections = sorted(conv.keys())
        preceding = [s for s in tool_sections if s < token.section_idx]
        if not preceding:
            return None
        nearest_cls = conv[preceding[-1]]
        return f"asst_after_{nearest_cls}"

    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_by_zone(
    tokens: List[TokenMetrics],
    tool_classes: Dict[str, Dict[int, str]],
    metrics: List[str],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """Aggregate metrics by zone label.

    Two-level: per-conversation mean → cross-conversation mean ± std.

    Returns: {zone: {metric: (mean, std, n_conversations)}}
    """
    # Level 1: per (conversation, zone) -> metric values
    conv_zone_values: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for t in tokens:
        zone = label_section(t, tool_classes)
        if zone is None:
            continue
        for m in metrics:
            v = t.metrics.get(m)
            if v is not None and math.isfinite(v):
                conv_zone_values[(t.conversation_id, zone)][m].append(v)

    # Per-conv means
    conv_means: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (conv_id, zone), mvals in conv_zone_values.items():
        for m, vals in mvals.items():
            conv_means[zone][m].append(float(np.mean(vals)))

    # Level 2: cross-conversation
    result: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for zone, mdata in conv_means.items():
        result[zone] = {}
        for m, vals in mdata.items():
            result[zone][m] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return result


def aggregate_by_norm_position_and_zone(
    tokens: List[TokenMetrics],
    tool_classes: Dict[str, Dict[int, str]],
    metric: str,
    n_bins: int = 20,
) -> Dict[str, Dict[int, Tuple[float, float, int]]]:
    """Aggregate a single metric by normalized position and zone.

    Returns: {zone: {bin_idx: (mean, std, n)}}
    """
    # Find max section_idx per conversation
    conv_max: Dict[str, int] = {}
    for t in tokens:
        if t.section_idx >= 0:
            conv_max[t.conversation_id] = max(
                conv_max.get(t.conversation_id, 0), t.section_idx
            )

    # Collect per (conv, zone, bin) values
    groups: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for t in tokens:
        zone = label_section(t, tool_classes)
        if zone is None:
            continue
        v = t.metrics.get(metric)
        if v is None or not math.isfinite(v):
            continue
        mx = conv_max.get(t.conversation_id, 0)
        if mx > 0:
            norm = t.section_idx / mx
            b = min(int(norm * n_bins), n_bins - 1)
        else:
            b = 0
        groups[(t.conversation_id, zone, b)].append(v)

    # Per-conv means
    conv_means: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for (conv_id, zone, b), vals in groups.items():
        conv_means[(zone, b)].append(float(np.mean(vals)))

    # Cross-conv
    result: Dict[str, Dict[int, Tuple[float, float, int]]] = defaultdict(dict)
    for (zone, b), vals in conv_means.items():
        result[zone][b] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return dict(result)


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------
def cross_run_comparison(
    std_tokens: List[TokenMetrics],
    sum_tokens: List[TokenMetrics],
    tool_classes: Dict[str, Dict[int, str]],
    metrics: List[str],
) -> Dict[str, Dict[str, Tuple[float, float, float, float, int]]]:
    """Compare assistant metrics between standard and summarized runs.

    Aligns by (conversation_id, original_uid). Groups by zone from the
    summarized run's classification.

    Returns: {zone: {metric: (std_mean, std_std, sum_mean, sum_std, n)}}
    """
    # Aggregate by (conv, original_uid) for each run
    def msg_means(tokens):
        groups = defaultdict(lambda: defaultdict(list))
        for t in tokens:
            if t.role != "assistant":
                continue
            for m in metrics:
                v = t.metrics.get(m)
                if v is not None and math.isfinite(v):
                    groups[(t.conversation_id, t.original_uid)][m].append(v)
        return {k: {m: float(np.mean(v)) for m, v in mv.items()} for k, mv in groups.items()}

    std_msgs = msg_means(std_tokens)
    sum_msgs = msg_means(sum_tokens)

    # Determine zone for each assistant message in summarized run
    msg_zones: Dict[Tuple[str, int], str] = {}
    for t in sum_tokens:
        if t.role == "assistant":
            key = (t.conversation_id, t.original_uid)
            if key not in msg_zones:
                zone = label_section(t, tool_classes)
                if zone:
                    msg_zones[key] = zone

    # Collect matched deltas per zone
    zone_vals: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    common = set(std_msgs.keys()) & set(sum_msgs.keys()) & set(msg_zones.keys())
    for key in common:
        zone = msg_zones[key]
        for m in metrics:
            sv = std_msgs[key].get(m)
            uv = sum_msgs[key].get(m)
            if sv is not None and uv is not None:
                zone_vals[zone][m].append((sv, uv))

    result: Dict[str, Dict[str, Tuple[float, float, float, float, int]]] = {}
    for zone, mdata in zone_vals.items():
        result[zone] = {}
        for m, pairs in mdata.items():
            s_arr = np.array([p[0] for p in pairs])
            u_arr = np.array([p[1] for p in pairs])
            result[zone][m] = (
                float(s_arr.mean()), float(s_arr.std()),
                float(u_arr.mean()), float(u_arr.std()),
                len(pairs),
            )
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_zone_bars(
    zone_agg: Dict[str, Dict[str, Tuple[float, float, int]]],
    output_path: Path,
    metrics: List[str],
):
    """Grouped bar chart: metrics by zone (within-run)."""
    import matplotlib.pyplot as plt

    zones_tool = ["tool_summarized", "tool_original"]
    zones_asst = ["asst_after_summarized", "asst_after_original"]
    zone_labels = {
        "tool_summarized": "Tool\n(summarized)",
        "tool_original": "Tool\n(original)",
        "asst_after_summarized": "Assistant\n(after summ.)",
        "asst_after_original": "Assistant\n(after orig.)",
    }
    zone_colors = {
        "tool_summarized": C_SUMMARIZED,
        "tool_original": C_ORIGINAL,
        "asst_after_summarized": C_SUMMARIZED,
        "asst_after_original": C_ORIGINAL,
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        all_zones = [z for z in zones_tool + zones_asst if z in zone_agg]
        x = np.arange(len(all_zones))
        vals = [zone_agg[z][m][0] if z in zone_agg and m in zone_agg[z] else 0 for z in all_zones]
        errs = [zone_agg[z][m][1] if z in zone_agg and m in zone_agg[z] else 0 for z in all_zones]
        colors = [zone_colors.get(z, "gray") for z in all_zones]

        bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.8, capsize=4, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([zone_labels.get(z, z) for z in all_zones], fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate values
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Metrics by Summarization Zone (within run)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved zone bars to {output_path}")


def plot_cross_run_bars(
    cross: Dict[str, Dict[str, Tuple[float, float, float, float, int]]],
    output_path: Path,
    metrics: List[str],
):
    """Grouped bar chart: standard vs summarized run, split by zone."""
    import matplotlib.pyplot as plt

    zones = ["asst_after_summarized", "asst_after_original"]
    zone_labels = {
        "asst_after_summarized": "After summarized tools",
        "asst_after_original": "After original tools",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    bar_w = 0.35
    for ax, m in zip(axes, metrics):
        avail = [z for z in zones if z in cross and m in cross[z]]
        x = np.arange(len(avail))

        std_vals = [cross[z][m][0] for z in avail]
        std_errs = [cross[z][m][1] for z in avail]
        sum_vals = [cross[z][m][2] for z in avail]
        sum_errs = [cross[z][m][3] for z in avail]

        ax.bar(x - bar_w / 2, std_vals, bar_w, yerr=std_errs,
               label="Standard", color=C_STANDARD, alpha=0.8, capsize=4)
        ax.bar(x + bar_w / 2, sum_vals, bar_w, yerr=sum_errs,
               label="Summarized", color=C_SUMRUN, alpha=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([zone_labels.get(z, z) for z in avail], fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate deltas
        for i, z in enumerate(avail):
            sv, _, uv, _, n = cross[z][m]
            delta = uv - sv
            sign = "+" if delta >= 0 else ""
            ax.text(i, max(sv, uv) * 1.02, f"Δ={sign}{delta:.4f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8, color="black")

    fig.suptitle("Assistant Metrics: Standard vs Summarized Run", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-run bars to {output_path}")


def plot_norm_position_by_zone(
    sum_tokens: List[TokenMetrics],
    tool_classes: Dict[str, Dict[int, str]],
    output_path: Path,
    metrics: List[str],
    n_bins: int = 20,
):
    """Line plots: metric vs normalized position, colored by zone."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    zone_colors = {
        "tool_summarized": C_SUMMARIZED,
        "tool_original": C_ORIGINAL,
        "asst_after_summarized": (0.9, 0.5, 0.5),
        "asst_after_original": (0.5, 0.8, 0.5),
    }
    zone_labels = {
        "tool_summarized": "Tool (summ.)",
        "tool_original": "Tool (orig.)",
        "asst_after_summarized": "Asst. (after summ.)",
        "asst_after_original": "Asst. (after orig.)",
    }

    for ax, m in zip(axes, metrics):
        pos_zone = aggregate_by_norm_position_and_zone(
            sum_tokens, tool_classes, m, n_bins
        )
        for zone in ["tool_summarized", "asst_after_summarized",
                      "tool_original", "asst_after_original"]:
            if zone not in pos_zone:
                continue
            bins = sorted(pos_zone[zone].keys())
            xs = [(b + 0.5) / n_bins for b in bins]
            ys = [pos_zone[zone][b][0] for b in bins]
            es = [pos_zone[zone][b][1] for b in bins]
            ns = [pos_zone[zone][b][2] for b in bins]
            # Filter to bins with enough data
            keep = [i for i, n in enumerate(ns) if n >= 2]
            if not keep:
                continue
            xs = np.array([xs[i] for i in keep])
            ys = np.array([ys[i] for i in keep])
            es = np.array([es[i] for i in keep])
            color = zone_colors.get(zone, "gray")
            ax.plot(xs, ys, color=color, linewidth=2, label=zone_labels.get(zone, zone))
            ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15)

        ax.set_xlabel("Normalized Section Position")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Metric Progression by Zone", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved progression plot to {output_path}")


def plot_cross_run_progression(
    std_tokens: List[TokenMetrics],
    sum_tokens: List[TokenMetrics],
    tool_classes: Dict[str, Dict[int, str]],
    output_path: Path,
    metrics: List[str],
    n_bins: int = 20,
):
    """Line plots: standard vs summarized run metrics over normalized position.

    Only assistant tokens, split into after-summarized and after-original zones.
    """
    import matplotlib.pyplot as plt

    # Build same classification for standard run tokens using summarized run's classes
    def agg_asst(tokens, classes, metric, nb):
        conv_max = {}
        for t in tokens:
            if t.section_idx >= 0:
                conv_max[t.conversation_id] = max(conv_max.get(t.conversation_id, 0), t.section_idx)
        groups = defaultdict(list)
        for t in tokens:
            if t.role != "assistant":
                continue
            zone = label_section(t, classes)
            if zone is None or not zone.startswith("asst_"):
                continue
            v = t.metrics.get(metric)
            if v is None or not math.isfinite(v):
                continue
            mx = conv_max.get(t.conversation_id, 0)
            b = min(int(t.section_idx / mx * nb), nb - 1) if mx > 0 else 0
            groups[(t.conversation_id, zone, b)].append(v)
        # Per-conv means then cross-conv
        conv_means = defaultdict(list)
        for (cid, z, b), vals in groups.items():
            conv_means[(z, b)].append(float(np.mean(vals)))
        result = defaultdict(dict)
        for (z, b), vals in conv_means.items():
            result[z][b] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        return dict(result)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        std_prog = agg_asst(std_tokens, tool_classes, m, n_bins)
        sum_prog = agg_asst(sum_tokens, tool_classes, m, n_bins)

        styles = {
            ("asst_after_summarized", "std"): (C_STANDARD, "--", "Std (after-summ zone)"),
            ("asst_after_summarized", "sum"): (C_SUMRUN, "-", "Sum (after-summ zone)"),
            ("asst_after_original", "std"): (C_STANDARD, "--", "Std (after-orig zone)"),
            ("asst_after_original", "sum"): (C_SUMRUN, "-", "Sum (after-orig zone)"),
        }

        for zone in ["asst_after_summarized", "asst_after_original"]:
            for run_name, prog in [("std", std_prog), ("sum", sum_prog)]:
                if zone not in prog:
                    continue
                bins = sorted(prog[zone].keys())
                keep = [b for b in bins if prog[zone][b][2] >= 2]
                if not keep:
                    continue
                xs = np.array([(b + 0.5) / n_bins for b in keep])
                ys = np.array([prog[zone][b][0] for b in keep])
                es = np.array([prog[zone][b][1] for b in keep])
                color, ls, label = styles[(zone, run_name)]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, label=label)
                ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.1)

        ax.set_xlabel("Normalized Section Position")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Assistant Metrics: Standard vs Summarized Over Position",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-run progression to {output_path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(
    zone_agg: Dict[str, Dict[str, Tuple[float, float, int]]],
    cross: Optional[Dict[str, Dict[str, Tuple[float, float, float, float, int]]]],
    metrics: List[str],
):
    print("\n" + "=" * 72)
    print("WITHIN-RUN: Metrics by Summarization Zone")
    print("=" * 72)
    for zone in ["tool_summarized", "tool_original",
                  "asst_after_summarized", "asst_after_original"]:
        if zone not in zone_agg:
            continue
        print(f"\n  {zone}:")
        for m in metrics:
            if m in zone_agg[zone]:
                mean, std, n = zone_agg[zone][m]
                print(f"    {m:25s}: {mean:.4f} ± {std:.4f}  (n={n} convs)")

    if cross:
        print("\n" + "=" * 72)
        print("CROSS-RUN: Standard vs Summarized (assistant turns)")
        print("=" * 72)
        for zone in ["asst_after_summarized", "asst_after_original"]:
            if zone not in cross:
                continue
            print(f"\n  {zone}:")
            for m in metrics:
                if m in cross[zone]:
                    sm, ss, um, us, n = cross[zone][m]
                    delta = um - sm
                    sign = "+" if delta >= 0 else ""
                    print(f"    {m:25s}: std={sm:.4f}  sum={um:.4f}  "
                          f"Δ={sign}{delta:.4f}  (n={n} msgs)")
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze the effect of tool-call summarization on metrics."
    )
    parser.add_argument("--summarized", required=True,
                        help="Metrics CSV from the summarized run")
    parser.add_argument("--standard", default=None,
                        help="Metrics CSV from the standard (all-original) run")
    parser.add_argument("--keep-last", type=int, default=4,
                        help="Number of last tool sections kept original (default: 4)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for figures and summaries")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metrics to analyze (default: neg_log_prob estimated_entropy topk_entropy)")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Bins for normalized position plots (default: 20)")
    args = parser.parse_args()

    metrics = args.metrics or METRICS
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summarized run
    print(f"Loading summarized run: {args.summarized}")
    sum_tokens = load_metrics_csv(Path(args.summarized))
    print(f"  {len(sum_tokens):,} tokens")

    # Classify tool sections
    tool_classes = classify_tool_sections(sum_tokens, keep_last=args.keep_last)
    n_convs = len(tool_classes)
    n_summ = sum(1 for c in tool_classes.values() for v in c.values() if v == "summarized")
    n_orig = sum(1 for c in tool_classes.values() for v in c.values() if v == "original")
    print(f"  {n_convs} conversations, {n_summ} summarized + {n_orig} original tool sections")

    # Within-run aggregation
    print("\nAggregating by zone...")
    zone_agg = aggregate_by_zone(sum_tokens, tool_classes, metrics)

    # Load standard run (optional)
    std_tokens = None
    cross = None
    if args.standard:
        print(f"\nLoading standard run: {args.standard}")
        std_tokens = load_metrics_csv(Path(args.standard))
        print(f"  {len(std_tokens):,} tokens")

        print("Computing cross-run comparison...")
        cross = cross_run_comparison(std_tokens, sum_tokens, tool_classes, metrics)

    # Print summary
    print_summary(zone_agg, cross, metrics)

    # Figures
    print("\nGenerating figures...")
    plot_zone_bars(zone_agg, output_dir / "zone_bars.png", metrics)
    plot_norm_position_by_zone(
        sum_tokens, tool_classes, output_dir / "zone_progression.png",
        metrics, n_bins=args.n_bins,
    )

    if std_tokens and cross:
        plot_cross_run_bars(cross, output_dir / "cross_run_bars.png", metrics)
        plot_cross_run_progression(
            std_tokens, sum_tokens, tool_classes,
            output_dir / "cross_run_progression.png",
            metrics, n_bins=args.n_bins,
        )

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
