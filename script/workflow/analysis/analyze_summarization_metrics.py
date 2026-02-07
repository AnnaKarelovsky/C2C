#!/usr/bin/env python3
"""
Analyze how summarization affects model confidence metrics.

Compares a summarized run against a standard (full-text) run on the same
questions, using neg_log_prob, top1_neg_log_prob, and topk_mass.

Zone classification: within each conversation, the last K tool sections are
"original" (full text kept), earlier ones are "summarized".  Assistant turns
inherit the zone of their preceding tool section.

Analyses:
  1. Within-run: tool and assistant metrics by zone
  2. Cross-run: standard vs summarized (paired by conversation_id)
  3. Correctness: correct vs incorrect (2-way from summarized run)
  4. 4-way split: both correct / only-summarized / only-standard / neither

Usage:
    python script/workflow/analysis/analyze_summarization_metrics.py \
        --summarized-csv  local/analysis/.../baseline_metrics.csv \
        --standard-csv    local/analysis/.../baseline_metrics.csv \
        --sum-results     local/evaluation/.../results.jsonl \
        --std-results     local/evaluation/.../results.jsonl \
        --output          local/analysis/.../metrics \
        --keep-last 4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Constants ────────────────────────────────────────────────────────────────

METRICS = ["neg_log_prob", "top1_neg_log_prob", "topk_mass"]
METRIC_LABELS = {
    "neg_log_prob": "Neg Log Prob",
    "top1_neg_log_prob": "Top-1 Neg Log Prob",
    "topk_mass": "Top-K Mass",
}

C_SUMMARIZED, C_ORIGINAL = "#e74c3c", "#2ca02c"
C_STANDARD, C_SUMRUN = "#1f77b4", "#ff7f0e"
C_CORRECT, C_INCORRECT = "#27ae60", "#c0392b"

GROUP_COLORS = {
    "both_correct": "#2ca02c",
    "only_summarized": "#1f77b4",
    "only_standard": "#ff7f0e",
    "neither": "#d62728",
}
GROUP_LABELS = {
    "both_correct": "Both correct",
    "only_summarized": "Only summ.",
    "only_standard": "Only std.",
    "neither": "Neither",
}
GROUP_ORDER = ["both_correct", "only_summarized", "only_standard", "neither"]

# ── Data loading ─────────────────────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    cols = [
        "conversation_id", "token_idx", "section_idx",
        "message_uid", "original_uid",
        "transform_type", "role", "content_type",
    ] + METRICS
    df = pd.read_csv(path, usecols=cols, dtype={
        "conversation_id": str, "token_idx": "int32", "section_idx": "int32",
        "message_uid": "int32", "original_uid": "int32",
        "transform_type": str, "role": str, "content_type": str,
    })
    print(f"  {len(df):,} rows, {df['conversation_id'].nunique()} conversations")
    return df


def load_correctness(path: Path) -> dict[str, bool]:
    cmap: dict[str, bool] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cmap[str(r["example_id"])] = bool(r.get("correct_llm", False))
    nc = sum(cmap.values())
    print(f"Correctness: {len(cmap)} examples, {nc} correct, {len(cmap) - nc} incorrect")
    return cmap


def load_4way(sum_path: Path, std_path: Path) -> dict[str, str]:
    sc = load_correctness(sum_path)
    stc = load_correctness(std_path)
    groups: dict[str, str] = {}
    for eid in sorted(set(sc) | set(stc)):
        s, t = sc.get(eid, False), stc.get(eid, False)
        if s and t:
            groups[eid] = "both_correct"
        elif s:
            groups[eid] = "only_summarized"
        elif t:
            groups[eid] = "only_standard"
        else:
            groups[eid] = "neither"
    counts = {g: sum(1 for v in groups.values() if v == g) for g in GROUP_ORDER}
    print(f"4-way split: {counts}")
    return groups


# ── Zone classification ──────────────────────────────────────────────────────


def classify_tool_sections(df: pd.DataFrame, keep_last: int) -> dict:
    """Label each (conversation_id, section_idx) as 'original' or 'summarized'."""
    tool = df[df["role"] == "tool"][["conversation_id", "section_idx"]].drop_duplicates()
    out: dict[tuple, str] = {}
    for cid, grp in tool.groupby("conversation_id"):
        secs = sorted(grp["section_idx"].unique())
        for i, s in enumerate(secs):
            out[(cid, s)] = "original" if i >= len(secs) - keep_last else "summarized"
    return out


def assign_zones(df: pd.DataFrame, tool_classes: dict) -> pd.Series:
    """Return a Series aligned to df with zone labels for tool/assistant rows."""
    zones = pd.Series(np.nan, index=df.index, dtype=object)

    # Tool zones
    mask_t = df["role"] == "tool"
    if mask_t.any():
        keys = list(zip(df.loc[mask_t, "conversation_id"], df.loc[mask_t, "section_idx"]))
        zones.loc[mask_t] = ["tool_" + tool_classes.get(k, "unknown") for k in keys]

    # Assistant zones: inherit from nearest preceding tool section
    mask_a = df["role"] == "assistant"
    if not mask_a.any():
        return zones
    lookup: dict[str, list] = defaultdict(list)
    for (cid, sidx), cls in tool_classes.items():
        lookup[cid].append((sidx, cls))
    for cid in lookup:
        lookup[cid].sort()

    az: list = []
    for cid, sidx in zip(df.loc[mask_a, "conversation_id"], df.loc[mask_a, "section_idx"]):
        secs = lookup.get(cid, [])
        cls = None
        lo, hi = 0, len(secs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if secs[mid][0] < sidx:
                cls = secs[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        az.append(f"asst_after_{cls}" if cls else None)
    zones.loc[mask_a] = az
    return zones


# ── Statistical helpers ──────────────────────────────────────────────────────


def _mwu(a, b):
    """Mann-Whitney U, returns (U, p) or (nan, nan) if too few samples."""
    if len(a) > 1 and len(b) > 1:
        return scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float("nan"), float("nan")


def _wilcoxon(diffs):
    if len(diffs) >= 10:
        return scipy_stats.wilcoxon(diffs)
    return float("nan"), float("nan")


def _cohend(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)


def _pstar(p):
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _filter_assistant_by_content_type(df, zones, zone_list, content_type=None):
    """Filter assistant rows by zone and optional content_type.

    Returns per-conversation metric means (DataFrame indexed by conversation_id).
    """
    asst = df[df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].isin(zone_list)]
    if content_type is not None:
        asst = asst[asst["content_type"] == content_type]
    if len(asst) == 0:
        return pd.DataFrame(columns=METRICS)
    return asst.groupby("conversation_id")[METRICS].mean()


CONTENT_TYPES = ["reasoning", "tool_call"]
CONTENT_TYPE_LABELS = {"reasoning": "Reasoning", "tool_call": "Tool Call"}
CONTENT_TYPE_COLORS = {"reasoning": "#8e44ad", "tool_call": "#2980b9"}


# ── Analyses (text summaries) ───────────────────────────────────────────────


def analysis_within_run(sum_df, tool_classes, zones) -> str:
    """Analyses 1-2: within-run tool and assistant metrics by zone."""
    lines: list[str] = []

    # Tool sections
    tool_df = sum_df[sum_df["role"] == "tool"].copy()
    tool_df["zone"] = [
        tool_classes.get((r["conversation_id"], r["section_idx"]), "unknown")
        for _, r in tool_df.iterrows()
    ]
    results: dict = {}
    for zone in ("summarized", "original"):
        cm = tool_df[tool_df["zone"] == zone].groupby("conversation_id")[METRICS].mean()
        results[zone] = cm
        lines.append(f"\n  {zone} tool sections (n={len(cm)}):")
        for m in METRICS:
            lines.append(f"    {m:25s}: {cm[m].mean():.4f} +/- {cm[m].std():.4f}")
    lines.append("\n  Mann-Whitney U (tool, summarized vs original):")
    for m in METRICS:
        s, o = results["summarized"][m].dropna(), results["original"][m].dropna()
        u, p = _mwu(s, o)
        lines.append(f"    {m:25s}: U={u:.1f}, p={p:.4e}, d={_cohend(s, o):.3f}")

    # Assistant by preceding tool zone
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].isin(["asst_after_summarized", "asst_after_original"])]
    results2: dict = {}
    for zone in ("asst_after_summarized", "asst_after_original"):
        cm = asst[asst["zone"] == zone].groupby("conversation_id")[METRICS].mean()
        results2[zone] = cm
        lines.append(f"\n  {zone} (n={len(cm)}):")
        for m in METRICS:
            lines.append(f"    {m:25s}: {cm[m].mean():.4f} +/- {cm[m].std():.4f}")
    lines.append("\n  Mann-Whitney U (assistant):")
    for m in METRICS:
        s = results2["asst_after_summarized"][m].dropna()
        o = results2["asst_after_original"][m].dropna()
        u, p = _mwu(s, o)
        lines.append(f"    {m:25s}: U={u:.1f}, p={p:.4e}, d={_cohend(s, o):.3f}")

    return "\n".join(lines)


def analysis_cross_run(std_df, sum_df, std_zones, sum_zones) -> tuple[dict, str]:
    """Analysis 3: cross-run aggregate comparison by zone."""
    lines: list[str] = []
    results: dict = {}
    for zone in ("asst_after_summarized", "asst_after_original"):
        std_cm = (std_df[std_df["role"] == "assistant"].assign(zone=std_zones)
                  .query("zone == @zone").groupby("conversation_id")[METRICS].mean())
        sum_cm = (sum_df[sum_df["role"] == "assistant"].assign(zone=sum_zones)
                  .query("zone == @zone").groupby("conversation_id")[METRICS].mean())
        common = std_cm.index.intersection(sum_cm.index)
        lines.append(f"\n  {zone} (n={len(common)}):")
        results[zone] = {}
        for m in METRICS:
            sv, uv = std_cm.loc[common, m].dropna(), sum_cm.loc[common, m].dropna()
            cb = sv.index.intersection(uv.index)
            sv, uv = sv.loc[cb], uv.loc[cb]
            delta = uv.mean() - sv.mean()
            w, p = _wilcoxon(uv.values - sv.values)
            results[zone][m] = dict(
                std_mean=sv.mean(), std_std=sv.std(),
                sum_mean=uv.mean(), sum_std=uv.std(),
                delta=delta, n=len(cb), wilcoxon_p=p,
            )
            sign = "+" if delta >= 0 else ""
            lines.append(f"    {m:25s}: std={sv.mean():.4f}, sum={uv.mean():.4f}, "
                         f"d={sign}{delta:.4f}, p={p:.4e} (n={len(cb)})")
    return results, "\n".join(lines)


def analysis_correctness(sum_df, zones, correct_map) -> str:
    """Analysis 4: correct vs incorrect (2-way)."""
    correct_ids = {c for c, v in correct_map.items() if v}
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    lines: list[str] = []
    for zone in ("asst_after_summarized", "asst_after_original"):
        zd = asst[asst["zone"] == zone]
        if len(zd) == 0:
            continue
        cm = zd.groupby("conversation_id")[METRICS].mean()
        lines.append(f"\n  {zone}:")
        for m in METRICS:
            cv = cm.loc[cm.index.isin(correct_ids), m].dropna()
            iv = cm.loc[~cm.index.isin(correct_ids), m].dropna()
            u, p = _mwu(cv, iv)
            lines.append(f"    {m:25s}: correct={cv.mean():.4f} (n={len(cv)}), "
                         f"incorrect={iv.mean():.4f} (n={len(iv)}), p={p:.4e}")
    return "\n".join(lines)


def analysis_4way(sum_df, std_df, sum_zones, std_zones, groups) -> str:
    """Analysis 5: 4-way correctness split."""
    group_ids: dict[str, set] = defaultdict(set)
    for eid, g in groups.items():
        group_ids[g].add(eid)

    lines: list[str] = [f"Groups: {dict({g: len(group_ids[g]) for g in GROUP_ORDER})}"]

    # Within summarized run
    asst_sum = sum_df[sum_df["role"] == "assistant"].copy()
    asst_sum["zone"] = sum_zones.loc[asst_sum.index]

    for zone in ("asst_after_summarized", "asst_after_original"):
        zd = asst_sum[asst_sum["zone"] == zone]
        if len(zd) == 0:
            continue
        cm = zd.groupby("conversation_id")[METRICS].mean()
        lines.append(f"\n  {zone}:")
        for g in GROUP_ORDER:
            gv = cm.loc[cm.index.isin(group_ids[g])]
            if len(gv) == 0:
                continue
            lines.append(f"    {GROUP_LABELS[g]} (n={len(gv)}):")
            for m in METRICS:
                lines.append(f"      {m:25s}: {gv[m].mean():.4f} +/- {gv[m].std():.4f}")

    # Cross-run last-4 zone
    zone = "asst_after_original"
    asst_std = std_df[std_df["role"] == "assistant"].copy()
    asst_std["zone"] = std_zones.loc[asst_std.index]
    std_cm = asst_std[asst_std["zone"] == zone].groupby("conversation_id")[METRICS].mean()
    sum_cm = asst_sum[asst_sum["zone"] == zone].groupby("conversation_id")[METRICS].mean()

    lines.append("\n  Cross-run (last-4 zone):")
    for g in GROUP_ORDER:
        ids = group_ids[g]
        common = std_cm.index[std_cm.index.isin(ids)].intersection(
            sum_cm.index[sum_cm.index.isin(ids)])
        if len(common) < 3:
            lines.append(f"    {GROUP_LABELS[g]}: n={len(common)}, too few")
            continue
        lines.append(f"    {GROUP_LABELS[g]} (n={len(common)}):")
        for m in METRICS:
            sv, uv = std_cm.loc[common, m], sum_cm.loc[common, m]
            delta = uv.mean() - sv.mean()
            sign = "+" if delta >= 0 else ""
            w, p = _wilcoxon(uv.values - sv.values)
            pstr = f"p={p:.4e}" if not np.isnan(p) else "too few"
            lines.append(f"      {m:25s}: std={sv.mean():.4f}, sum={uv.mean():.4f}, "
                         f"d={sign}{delta:.4f}, {pstr}")

    return "\n".join(lines)


# ── Figures ──────────────────────────────────────────────────────────────────


def fig1_zone_bars(sum_df, tool_classes, zones, out):
    """Within-run metrics by zone."""
    tool_df = sum_df[sum_df["role"] == "tool"].copy()
    tool_df["zone"] = [
        "tool_" + tool_classes.get((r["conversation_id"], r["section_idx"]), "unknown")
        for _, r in tool_df.iterrows()
    ]
    asst_df = sum_df[sum_df["role"] == "assistant"].copy()
    asst_df["zone"] = zones.loc[asst_df.index]
    combined = pd.concat([
        tool_df[["conversation_id", "zone"] + METRICS],
        asst_df[asst_df["zone"].notna()][["conversation_id", "zone"] + METRICS],
    ])

    order = ["tool_summarized", "tool_original", "asst_after_summarized", "asst_after_original"]
    labels = {"tool_summarized": "Tool\n(summ.)", "tool_original": "Tool\n(orig.)",
              "asst_after_summarized": "Asst.\n(after summ.)", "asst_after_original": "Asst.\n(after orig.)"}
    colors = {"tool_summarized": C_SUMMARIZED, "tool_original": C_ORIGINAL,
              "asst_after_summarized": C_SUMMARIZED, "asst_after_original": C_ORIGINAL}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    for ax, m in zip(axes, METRICS):
        cm = combined.groupby(["conversation_id", "zone"])[m].mean().reset_index()
        za = cm.groupby("zone")[m].agg(["mean", "std"])
        avail = [z for z in order if z in za.index]
        x = np.arange(len(avail))
        bars = ax.bar(x, [za.loc[z, "mean"] for z in avail],
                      yerr=[za.loc[z, "std"] for z in avail],
                      color=[colors.get(z, "gray") for z in avail],
                      alpha=0.8, capsize=4, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(z, z) for z in avail], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.grid(True, alpha=0.3, axis="y")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Within-Run Metrics by Summarization Zone (Summarized Run)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig1_zone_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_cross_run_bars(cross_results, out):
    """Cross-run standard vs summarized assistant metrics by zone."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    zone_order = ["asst_after_summarized", "asst_after_original"]
    zone_lbl = {"asst_after_summarized": "After earlier\ntools",
                "asst_after_original": "After last-4\ntools"}
    bw = 0.35
    for ax, m in zip(axes, METRICS):
        avail = [z for z in zone_order if z in cross_results and m in cross_results[z]]
        x = np.arange(len(avail))
        ax.bar(x - bw / 2, [cross_results[z][m]["std_mean"] for z in avail], bw,
               yerr=[cross_results[z][m]["std_std"] for z in avail],
               label="Standard", color=C_STANDARD, alpha=0.8, capsize=4)
        ax.bar(x + bw / 2, [cross_results[z][m]["sum_mean"] for z in avail], bw,
               yerr=[cross_results[z][m]["sum_std"] for z in avail],
               label="Summarized", color=C_SUMRUN, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([zone_lbl.get(z, z) for z in avail], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        for i, z in enumerate(avail):
            d = cross_results[z][m]
            sign = "+" if d["delta"] >= 0 else ""
            sig = "***" if d["wilcoxon_p"] < 0.001 else "**" if d["wilcoxon_p"] < 0.01 \
                else "*" if d["wilcoxon_p"] < 0.05 else "ns"
            ax.text(i, max(d["std_mean"] + d["std_std"],
                           d["sum_mean"] + d["sum_std"]) * 1.02,
                    f"d={sign}{d['delta']:.4f}\np={d['wilcoxon_p']:.3e} {sig}\n(n={d['n']})",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Cross-Run: Standard vs Summarized Assistant Metrics\n"
                 "(per-conversation means, paired by conversation_id)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_cross_run_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_correctness_by_zone(sum_df, zones, correct_map, out):
    """Correct vs incorrect within summarized run, by zone."""
    correct_ids = {c for c, v in correct_map.items() if v}
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].notna()]
    asst["correct"] = asst["conversation_id"].isin(correct_ids)

    zone_order = ["asst_after_summarized", "asst_after_original"]
    zone_lbl = {"asst_after_summarized": "After earlier\ntools",
                "asst_after_original": "After last-4\ntools"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    bw = 0.35
    for ax, m in zip(axes, METRICS):
        avail = [z for z in zone_order if z in asst["zone"].unique()]
        x = np.arange(len(avail))
        cm = asst.groupby(["conversation_id", "zone", "correct"])[m].mean().reset_index()
        for j, (is_c, label, color) in enumerate([
            (True, "Correct", C_CORRECT), (False, "Incorrect", C_INCORRECT)
        ]):
            vals = [cm[(cm["zone"] == z) & (cm["correct"] == is_c)][m].mean() for z in avail]
            errs = [cm[(cm["zone"] == z) & (cm["correct"] == is_c)][m].std() for z in avail]
            ax.bar(x + (j - 0.5) * bw, vals, bw, yerr=errs,
                   label=label, color=color, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([zone_lbl.get(z, z) for z in avail], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Correct vs Incorrect: Assistant Metrics by Zone (Summarized Run)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_correctness_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3a_last4_cross_run_correctness(std_df, sum_df, std_zones, sum_zones,
                                      correct_map, out):
    """Cross-run comparison in last-4 zone with statistical annotations.

    Like fig5 but adds paired Wilcoxon tests between runs within each
    correctness group, plus Mann-Whitney between correct/incorrect within
    each run.
    """
    correct_ids = {c for c, v in correct_map.items() if v}
    incorrect_ids = {c for c, v in correct_map.items() if not v}
    zone = "asst_after_original"

    # Gather per-conversation means for each run
    run_data = {}
    for rdf, rz, run_label in [
        (std_df, std_zones, "Standard"),
        (sum_df, sum_zones, "Summarized"),
    ]:
        cm = _filter_assistant_by_content_type(rdf, rz, [zone])
        run_data[run_label] = cm

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6))
    bw = 0.3
    for ax, m in zip(axes, METRICS):
        x = np.arange(2)  # [Correct, Incorrect]
        bar_data = {}  # (run, group) -> (mean, std, n, values)
        for j, (run_label, color) in enumerate([
            ("Standard", C_STANDARD), ("Summarized", C_SUMRUN),
        ]):
            cm = run_data[run_label]
            vals_list, errs_list, ns_list = [], [], []
            for gname, gids in [("correct", correct_ids), ("incorrect", incorrect_ids)]:
                gv = cm.loc[cm.index.isin(gids), m].dropna()
                vals_list.append(gv.mean() if len(gv) > 0 else 0)
                errs_list.append(gv.std() if len(gv) > 1 else 0)
                ns_list.append(len(gv))
                bar_data[(run_label, gname)] = gv
            ax.bar(x + (j - 0.5) * bw, vals_list, bw, yerr=errs_list,
                   label=run_label, color=color, alpha=0.8, capsize=4)

        # Annotations: cross-run paired Wilcoxon within each correctness group
        for i, gname in enumerate(["correct", "incorrect"]):
            std_v = bar_data[("Standard", gname)]
            sum_v = bar_data[("Summarized", gname)]
            common = std_v.index.intersection(sum_v.index)
            if len(common) >= 10:
                diffs = sum_v.loc[common].values - std_v.loc[common].values
                _, p_paired = _wilcoxon(diffs)
                delta = sum_v.loc[common].mean() - std_v.loc[common].mean()
                sign = "+" if delta >= 0 else ""
                ymax = max(std_v.loc[common].mean() + std_v.loc[common].std(),
                           sum_v.loc[common].mean() + sum_v.loc[common].std())
                ax.text(i, ymax * 1.03,
                        f"\u0394={sign}{delta:.4f}\nWilcoxon p={p_paired:.2e} {_pstar(p_paired)}"
                        f"\n(n={len(common)})",
                        ha="center", va="bottom", fontsize=7)

        # Within-run Mann-Whitney (correct vs incorrect)
        for run_label, x_offset in [("Standard", -0.35), ("Summarized", 0.35)]:
            cv = bar_data[(run_label, "correct")]
            iv = bar_data[(run_label, "incorrect")]
            if len(cv) > 1 and len(iv) > 1:
                _, p_mw = _mwu(cv, iv)
                ax.text(0.5 + x_offset, ax.get_ylim()[1] * 0.98,
                        f"{run_label}: MWU p={p_mw:.2e} {_pstar(p_mw)}",
                        ha="center", va="top", fontsize=6,
                        fontstyle="italic", color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=10)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Last-4 Zone: Standard vs Summarized by Correctness\n"
                 "(paired Wilcoxon cross-run, Mann-Whitney within-run)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3a_last4_cross_run_correctness.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def analysis_fig3a(std_df, sum_df, std_zones, sum_zones, correct_map) -> str:
    """Text analysis for fig3a: cross-run last-4 zone by correctness."""
    correct_ids = {c for c, v in correct_map.items() if v}
    incorrect_ids = {c for c, v in correct_map.items() if not v}
    zone = "asst_after_original"
    lines = ["\nFig3a: Last-4 Zone Cross-Run by Correctness", "-" * 50]

    run_data = {}
    for rdf, rz, run_label in [
        (std_df, std_zones, "Standard"),
        (sum_df, sum_zones, "Summarized"),
    ]:
        cm = _filter_assistant_by_content_type(rdf, rz, [zone])
        run_data[run_label] = cm

    for gname, gids in [("Correct", correct_ids), ("Incorrect", incorrect_ids)]:
        lines.append(f"\n  {gname} conversations:")
        for m in METRICS:
            std_v = run_data["Standard"].loc[
                run_data["Standard"].index.isin(gids), m].dropna()
            sum_v = run_data["Summarized"].loc[
                run_data["Summarized"].index.isin(gids), m].dropna()
            common = std_v.index.intersection(sum_v.index)
            if len(common) < 3:
                lines.append(f"    {m:25s}: too few (n={len(common)})")
                continue
            sv, uv = std_v.loc[common], sum_v.loc[common]
            delta = uv.mean() - sv.mean()
            _, p = _wilcoxon(uv.values - sv.values)
            sign = "+" if delta >= 0 else ""
            lines.append(f"    {m:25s}: std={sv.mean():.4f}, sum={uv.mean():.4f}, "
                         f"d={sign}{delta:.4f}, p={p:.4e} (n={len(common)})")

    # Within-run correct vs incorrect
    lines.append("\n  Within-run Mann-Whitney (correct vs incorrect):")
    for run_label in ["Standard", "Summarized"]:
        cm = run_data[run_label]
        lines.append(f"    {run_label}:")
        for m in METRICS:
            cv = cm.loc[cm.index.isin(correct_ids), m].dropna()
            iv = cm.loc[cm.index.isin(incorrect_ids), m].dropna()
            u, p = _mwu(cv, iv)
            lines.append(f"      {m:25s}: correct={cv.mean():.4f} (n={len(cv)}), "
                         f"incorrect={iv.mean():.4f} (n={len(iv)}), p={p:.4e}")

    return "\n".join(lines)


def fig3b_correctness_by_content_type(sum_df, zones, correct_map, out):
    """Correct vs incorrect split by content_type within summarized run.

    2 rows (reasoning, tool_call) x 3 columns (metrics).
    X-axis per subplot: After earlier tools, After last-4 tools.
    Bars: Correct (green) vs Incorrect (red).
    """
    correct_ids = {c for c, v in correct_map.items() if v}
    zone_order = ["asst_after_summarized", "asst_after_original"]
    zone_lbl = {"asst_after_summarized": "After earlier\ntools",
                "asst_after_original": "After last-4\ntools"}

    fig, axes = plt.subplots(len(CONTENT_TYPES), len(METRICS),
                             figsize=(6 * len(METRICS), 5 * len(CONTENT_TYPES)))
    bw = 0.3
    for row, ct in enumerate(CONTENT_TYPES):
        for col, m in enumerate(METRICS):
            ax = axes[row, col]
            x = np.arange(len(zone_order))
            for j, (is_c, label, color) in enumerate([
                (True, "Correct", C_CORRECT), (False, "Incorrect", C_INCORRECT),
            ]):
                vals, errs, ns = [], [], []
                for zone in zone_order:
                    cm = _filter_assistant_by_content_type(sum_df, zones, [zone], ct)
                    if is_c:
                        gv = cm.loc[cm.index.isin(correct_ids), m].dropna()
                    else:
                        gv = cm.loc[~cm.index.isin(correct_ids), m].dropna()
                    vals.append(gv.mean() if len(gv) > 0 else 0)
                    errs.append(gv.std() if len(gv) > 1 else 0)
                    ns.append(len(gv))
                ax.bar(x + (j - 0.5) * bw, vals, bw, yerr=errs,
                       label=label, color=color, alpha=0.8, capsize=4)

            # Mann-Whitney annotation per zone
            for i, zone in enumerate(zone_order):
                cm = _filter_assistant_by_content_type(sum_df, zones, [zone], ct)
                cv = cm.loc[cm.index.isin(correct_ids), m].dropna()
                iv = cm.loc[~cm.index.isin(correct_ids), m].dropna()
                _, p = _mwu(cv, iv)
                ymax = max(cv.mean() + cv.std() if len(cv) > 1 else 0,
                           iv.mean() + iv.std() if len(iv) > 1 else 0)
                if ymax > 0:
                    ax.text(i, ymax * 1.03,
                            f"p={p:.2e} {_pstar(p)}\nn={len(cv)}+{len(iv)}",
                            ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([zone_lbl[z] for z in zone_order], fontsize=9)
            ax.set_title(f"{CONTENT_TYPE_LABELS[ct]} — {METRIC_LABELS[m]}",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[m])

    fig.suptitle("Correct vs Incorrect by Content Type (Summarized Run)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3b_correctness_by_content_type.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def analysis_fig3b(sum_df, zones, correct_map) -> str:
    """Text analysis for fig3b: correctness by content type."""
    correct_ids = {c for c, v in correct_map.items() if v}
    zone_order = ["asst_after_summarized", "asst_after_original"]
    lines = ["\nFig3b: Correctness by Content Type (Summarized Run)", "-" * 50]

    for ct in CONTENT_TYPES:
        lines.append(f"\n  Content type: {CONTENT_TYPE_LABELS[ct]}")
        for zone in zone_order:
            cm = _filter_assistant_by_content_type(sum_df, zones, [zone], ct)
            if len(cm) == 0:
                lines.append(f"    {zone}: no data")
                continue
            lines.append(f"    {zone}:")
            for m in METRICS:
                cv = cm.loc[cm.index.isin(correct_ids), m].dropna()
                iv = cm.loc[~cm.index.isin(correct_ids), m].dropna()
                u, p = _mwu(cv, iv)
                lines.append(
                    f"      {m:25s}: correct={cv.mean():.4f} (n={len(cv)}), "
                    f"incorrect={iv.mean():.4f} (n={len(iv)}), p={p:.4e}")

    return "\n".join(lines)


def fig3c_cross_run_by_content_type(std_df, sum_df, std_zones, sum_zones,
                                    correct_map, out):
    """Cross-run comparison in last-4 zone, split by content_type.

    2 rows (reasoning, tool_call) x 3 columns (metrics).
    X-axis: Correct vs Incorrect.
    Bars: Standard (blue) vs Summarized (orange).
    """
    correct_ids = {c for c, v in correct_map.items() if v}
    incorrect_ids = {c for c, v in correct_map.items() if not v}
    zone = "asst_after_original"

    fig, axes = plt.subplots(len(CONTENT_TYPES), len(METRICS),
                             figsize=(6 * len(METRICS), 5 * len(CONTENT_TYPES)))
    bw = 0.3
    for row, ct in enumerate(CONTENT_TYPES):
        # Per-conversation means for each run & content type
        std_cm = _filter_assistant_by_content_type(std_df, std_zones, [zone], ct)
        sum_cm = _filter_assistant_by_content_type(sum_df, sum_zones, [zone], ct)

        for col, m in enumerate(METRICS):
            ax = axes[row, col]
            x = np.arange(2)  # [Correct, Incorrect]
            bar_data = {}
            for j, (cm, run_label, color) in enumerate([
                (std_cm, "Standard", C_STANDARD),
                (sum_cm, "Summarized", C_SUMRUN),
            ]):
                vals, errs = [], []
                for gname, gids in [("correct", correct_ids),
                                    ("incorrect", incorrect_ids)]:
                    gv = cm.loc[cm.index.isin(gids), m].dropna()
                    vals.append(gv.mean() if len(gv) > 0 else 0)
                    errs.append(gv.std() if len(gv) > 1 else 0)
                    bar_data[(run_label, gname)] = gv
                ax.bar(x + (j - 0.5) * bw, vals, bw, yerr=errs,
                       label=run_label, color=color, alpha=0.8, capsize=4)

            # Cross-run Wilcoxon per correctness group
            for i, gname in enumerate(["correct", "incorrect"]):
                std_v = bar_data[("Standard", gname)]
                sum_v = bar_data[("Summarized", gname)]
                common = std_v.index.intersection(sum_v.index)
                if len(common) >= 10:
                    diffs = sum_v.loc[common].values - std_v.loc[common].values
                    _, p_paired = _wilcoxon(diffs)
                    delta = sum_v.loc[common].mean() - std_v.loc[common].mean()
                    sign = "+" if delta >= 0 else ""
                    ymax = max(std_v.loc[common].mean() + std_v.loc[common].std(),
                               sum_v.loc[common].mean() + sum_v.loc[common].std())
                    if ymax > 0:
                        ax.text(i, ymax * 1.03,
                                f"\u0394={sign}{delta:.4f}\np={p_paired:.2e} "
                                f"{_pstar(p_paired)}\n(n={len(common)})",
                                ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(["Correct", "Incorrect"], fontsize=10)
            ax.set_title(f"{CONTENT_TYPE_LABELS[ct]} — {METRIC_LABELS[m]}",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[m])

    fig.suptitle("Last-4 Zone: Standard vs Summarized by Content Type & Correctness",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3c_cross_run_by_content_type.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def analysis_fig3c(std_df, sum_df, std_zones, sum_zones, correct_map) -> str:
    """Text analysis for fig3c: cross-run by content type."""
    correct_ids = {c for c, v in correct_map.items() if v}
    incorrect_ids = {c for c, v in correct_map.items() if not v}
    zone = "asst_after_original"
    lines = ["\nFig3c: Last-4 Zone Cross-Run by Content Type", "-" * 50]

    for ct in CONTENT_TYPES:
        lines.append(f"\n  Content type: {CONTENT_TYPE_LABELS[ct]}")
        std_cm = _filter_assistant_by_content_type(std_df, std_zones, [zone], ct)
        sum_cm = _filter_assistant_by_content_type(sum_df, sum_zones, [zone], ct)

        for gname, gids in [("Correct", correct_ids),
                            ("Incorrect", incorrect_ids)]:
            lines.append(f"    {gname}:")
            for m in METRICS:
                std_v = std_cm.loc[std_cm.index.isin(gids), m].dropna()
                sum_v = sum_cm.loc[sum_cm.index.isin(gids), m].dropna()
                common = std_v.index.intersection(sum_v.index)
                if len(common) < 3:
                    lines.append(f"      {m:25s}: too few (n={len(common)})")
                    continue
                sv, uv = std_v.loc[common], sum_v.loc[common]
                delta = uv.mean() - sv.mean()
                _, p = _wilcoxon(uv.values - sv.values)
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"      {m:25s}: std={sv.mean():.4f}, sum={uv.mean():.4f}, "
                    f"d={sign}{delta:.4f}, p={p:.4e} (n={len(common)})")

    return "\n".join(lines)


def fig4_progression(sum_df, zones, correct_map, out, n_bins=15):
    """Metric progression by normalized position, correct vs incorrect."""
    correct_ids = {c for c, v in correct_map.items() if v}
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].notna()]
    asst["correct"] = asst["conversation_id"].isin(correct_ids)
    mx = asst.groupby("conversation_id")["section_idx"].max()
    asst = asst.merge(mx.rename("max_sec"), on="conversation_id")
    asst["pos_bin"] = (asst["section_idx"] / asst["max_sec"].clip(lower=1) * n_bins
                       ).clip(upper=n_bins - 1).astype(int)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    for ax, m in zip(axes, METRICS):
        for is_c, label, color, ls in [
            (True, "Correct", C_CORRECT, "-"), (False, "Incorrect", C_INCORRECT, "--")
        ]:
            sub = asst[asst["correct"] == is_c]
            ba = sub.groupby(["conversation_id", "pos_bin"])[m].mean().groupby("pos_bin").agg(
                ["mean", "std", "count"])
            ba = ba[ba["count"] >= 3]
            if len(ba) == 0:
                continue
            xs = (ba.index + 0.5) / n_bins
            ax.plot(xs, ba["mean"], color=color, linestyle=ls, linewidth=2, label=label)
            ax.fill_between(xs, ba["mean"] - ba["std"], ba["mean"] + ba["std"],
                            color=color, alpha=0.1)
        ax.set_xlabel("Normalized Position")
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Assistant Metric Progression: Correct vs Incorrect (Summarized Run)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_cross_run_by_correctness(std_df, sum_df, std_zones, sum_zones, correct_map, out):
    """Cross-run comparison in last-4 zone, split by correct/incorrect."""
    correct_ids = {c for c, v in correct_map.items() if v}
    incorrect_ids = {c for c, v in correct_map.items() if not v}
    zone = "asst_after_original"
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    bw = 0.2
    for ax, m in zip(axes, METRICS):
        x = np.arange(2)
        for j, (rdf, rz, label, color) in enumerate([
            (std_df, std_zones, "Standard", C_STANDARD),
            (sum_df, sum_zones, "Summarized", C_SUMRUN),
        ]):
            a = rdf[rdf["role"] == "assistant"].copy()
            a["zone"] = rz.loc[a.index]
            cm = a[a["zone"] == zone].groupby("conversation_id")[m].mean()
            vals = [cm.loc[cm.index.isin(correct_ids)].mean(),
                    cm.loc[cm.index.isin(incorrect_ids)].mean()]
            errs = [cm.loc[cm.index.isin(correct_ids)].std(),
                    cm.loc[cm.index.isin(incorrect_ids)].std()]
            ax.bar(x + (j - 0.5) * bw, vals, bw, yerr=errs,
                   label=label, color=color, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=10)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Last-4 Zone Assistant Metrics: Standard vs Summarized, by Correctness",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig5_cross_run_correctness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_4way_by_zone(sum_df, zones, groups, out):
    """4-way split: metrics by zone within summarized run."""
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].notna()]
    asst["group"] = asst["conversation_id"].map(groups)
    asst = asst[asst["group"].notna()]

    zone_order = ["asst_after_summarized", "asst_after_original"]
    zone_lbl = {"asst_after_summarized": "After earlier tools",
                "asst_after_original": "After last-4 tools"}

    fig, axes = plt.subplots(len(METRICS), 2, figsize=(12, 4 * len(METRICS)), sharey="row")
    for row, m in enumerate(METRICS):
        for col, zone in enumerate(zone_order):
            ax = axes[row, col]
            zd = asst[asst["zone"] == zone]
            cm = zd.groupby(["conversation_id", "group"])[m].mean().reset_index()
            avail = [g for g in GROUP_ORDER if g in cm["group"].values]
            x = np.arange(len(avail))
            vals, errs, ns = [], [], []
            for g in avail:
                gv = cm[cm["group"] == g][m]
                vals.append(gv.mean())
                errs.append(gv.std())
                ns.append(len(gv))
            bars = ax.bar(x, vals, yerr=errs,
                          color=[GROUP_COLORS[g] for g in avail],
                          alpha=0.8, capsize=4, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{GROUP_LABELS[g]}\n(n={n})" for g, n in zip(avail, ns)],
                               fontsize=8)
            ax.set_title(f"{METRIC_LABELS[m]} — {zone_lbl[zone]}", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[m])
    fig.suptitle("4-Way Correctness Split: Assistant Metrics by Zone (Summarized Run)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig6_4way_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_4way_cross_run(std_df, sum_df, std_zones, sum_zones, groups, out):
    """4-way split: cross-run comparison in last-4 zone."""
    zone = "asst_after_original"
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6))
    bw = 0.18
    for ax, m in zip(axes, METRICS):
        x = np.arange(len(GROUP_ORDER))
        for j, (rdf, rz, label, color) in enumerate([
            (std_df, std_zones, "Standard", C_STANDARD),
            (sum_df, sum_zones, "Summarized", C_SUMRUN),
        ]):
            a = rdf[rdf["role"] == "assistant"].copy()
            a["zone"] = rz.loc[a.index]
            cm = a[a["zone"] == zone].groupby("conversation_id")[m].mean()
            means, stds = [], []
            for g in GROUP_ORDER:
                gids = {e for e, gr in groups.items() if gr == g}
                gv = cm.loc[cm.index.isin(gids)]
                means.append(gv.mean() if len(gv) > 0 else 0)
                stds.append(gv.std() if len(gv) > 1 else 0)
            ax.bar(x + (j - 0.5) * bw, means, bw, yerr=stds,
                   label=label, color=color, alpha=0.8, capsize=4)
        gns = {g: sum(1 for v in groups.values() if v == g) for g in GROUP_ORDER}
        ax.set_xticks(x)
        ax.set_xticklabels([f"{GROUP_LABELS[g]}\n(n={gns[g]})" for g in GROUP_ORDER], fontsize=8)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Last-4 Zone: Standard vs Summarized by 4-Way Correctness Group",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig7_4way_cross_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig8_4way_progression(sum_df, zones, groups, out, n_bins=15):
    """4-way split: metric progression by normalized position."""
    asst = sum_df[sum_df["role"] == "assistant"].copy()
    asst["zone"] = zones.loc[asst.index]
    asst = asst[asst["zone"].notna()]
    asst["group"] = asst["conversation_id"].map(groups)
    asst = asst[asst["group"].notna()]
    mx = asst.groupby("conversation_id")["section_idx"].max()
    asst = asst.merge(mx.rename("max_sec"), on="conversation_id")
    asst["pos_bin"] = (asst["section_idx"] / asst["max_sec"].clip(lower=1) * n_bins
                       ).clip(upper=n_bins - 1).astype(int)
    ls_map = {"both_correct": "-", "only_summarized": "--",
              "only_standard": "-.", "neither": ":"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    for ax, m in zip(axes, METRICS):
        for g in GROUP_ORDER:
            sub = asst[asst["group"] == g]
            if len(sub) == 0:
                continue
            ba = sub.groupby(["conversation_id", "pos_bin"])[m].mean().groupby("pos_bin").agg(
                ["mean", "std", "count"])
            ba = ba[ba["count"] >= 3]
            if len(ba) == 0:
                continue
            xs = (ba.index + 0.5) / n_bins
            ax.plot(xs, ba["mean"], color=GROUP_COLORS[g], linestyle=ls_map[g],
                    linewidth=2, label=GROUP_LABELS[g])
            ax.fill_between(xs, ba["mean"] - ba["std"], ba["mean"] + ba["std"],
                            color=GROUP_COLORS[g], alpha=0.07)
        ax.set_xlabel("Normalized Position")
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(METRIC_LABELS[m])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Assistant Metric Progression by 4-Way Group (Summarized Run)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig8_4way_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarized-csv", type=Path, required=True,
                   help="Metrics CSV for the summarized run")
    p.add_argument("--standard-csv", type=Path, required=True,
                   help="Metrics CSV for the standard (full-text) run")
    p.add_argument("--sum-results", type=Path, required=True,
                   help="results.jsonl for the summarized run")
    p.add_argument("--std-results", type=Path, required=True,
                   help="results.jsonl for the standard run")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for figures and summaries")
    p.add_argument("--keep-last", type=int, default=4,
                   help="Number of trailing tool sections to keep as 'original' (default: 4)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    sum_df = load_csv(args.summarized_csv)
    std_df = load_csv(args.standard_csv)
    correct_map = load_correctness(args.sum_results)
    groups = load_4way(args.sum_results, args.std_results)

    # Classify zones
    print("\nClassifying zones...")
    sum_tc = classify_tool_sections(sum_df, args.keep_last)
    std_tc = classify_tool_sections(std_df, args.keep_last)
    sum_zones = assign_zones(sum_df, sum_tc)
    std_zones = assign_zones(std_df, std_tc)

    # Run analyses and collect text summaries
    parts = ["PERPLEXITY METRICS ANALYSIS: SUMMARIZATION EFFECT",
             "=" * 72,
             f"Metrics: {', '.join(METRICS)}",
             f"Summarized run: {sum_df['conversation_id'].nunique()} convs, {len(sum_df):,} tokens",
             f"Standard run: {std_df['conversation_id'].nunique()} convs, {len(std_df):,} tokens",
             ""]

    print("\n--- Within-run ---")
    parts.append(analysis_within_run(sum_df, sum_tc, sum_zones))

    print("\n--- Cross-run ---")
    cross_results, cross_text = analysis_cross_run(std_df, sum_df, std_zones, sum_zones)
    parts.append(cross_text)

    print("\n--- Correctness (2-way) ---")
    parts.append(analysis_correctness(sum_df, sum_zones, correct_map))

    print("\n--- 4-way split ---")
    parts.append(analysis_4way(sum_df, std_df, sum_zones, std_zones, groups))

    print("\n--- Fig3a: Last-4 cross-run by correctness ---")
    parts.append(analysis_fig3a(std_df, sum_df, std_zones, sum_zones, correct_map))

    print("\n--- Fig3b: Correctness by content type ---")
    parts.append(analysis_fig3b(sum_df, sum_zones, correct_map))

    print("\n--- Fig3c: Cross-run by content type ---")
    parts.append(analysis_fig3c(std_df, sum_df, std_zones, sum_zones, correct_map))

    # Save summary
    (args.output / "summary.txt").write_text("\n\n".join(parts))
    print(f"\nSaved summary.txt")

    # Generate figures
    print("\nGenerating figures...")
    fig1_zone_bars(sum_df, sum_tc, sum_zones, args.output)
    fig2_cross_run_bars(cross_results, args.output)
    fig3_correctness_by_zone(sum_df, sum_zones, correct_map, args.output)
    fig3a_last4_cross_run_correctness(std_df, sum_df, std_zones, sum_zones,
                                      correct_map, args.output)
    fig3b_correctness_by_content_type(sum_df, sum_zones, correct_map, args.output)
    fig3c_cross_run_by_content_type(std_df, sum_df, std_zones, sum_zones,
                                    correct_map, args.output)
    fig4_progression(sum_df, sum_zones, correct_map, args.output)
    fig5_cross_run_by_correctness(std_df, sum_df, std_zones, sum_zones, correct_map, args.output)
    fig6_4way_by_zone(sum_df, sum_zones, groups, args.output)
    fig7_4way_cross_run(std_df, sum_df, std_zones, sum_zones, groups, args.output)
    fig8_4way_progression(sum_df, sum_zones, groups, args.output)
    print(f"\nAll outputs saved to {args.output}")


if __name__ == "__main__":
    main()
