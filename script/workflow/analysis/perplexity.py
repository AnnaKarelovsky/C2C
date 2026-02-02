#!/usr/bin/env python3
"""
Perplexity and entropy analysis for conversation histories.

Loads evaluation results, prefills sequences through a model, and analyzes
how entropy/perplexity varies by token position and message type.

Usage:
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model Qwen/Qwen3-32B \
        --output-dir local/analysis/perplexity

Features:
    - Per-token entropy and perplexity computation
    - Breakdown by message role (system, user, assistant, tool)
    - Support for comparing metrics before/after context transformations
    - Visualization of metric distributions
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.utils.core import (
    DEFAULT_METRICS,
    EntropyMetric,
    NegLogProbMetric,
    PerplexityMetric,
    TokenMetric,
    align_metrics,
    get_metric_by_name,
    prefill_and_compute_metrics,
)
from camel.toolkits import FunctionTool

from rosetta.workflow.analysis.interface import (
    TokenizedConversation,
    apply_context_transform,
    extract_conversations,
    load_evaluation_results,
    save_token_plot_data_csv
)
from rosetta.workflow.analysis.oss_tokenizer import (
    batch_tokenize_with_sections,
    tokenize_conversation_with_sections,
)
from rosetta.workflow.browse_searcher import get_document, search

from rosetta.workflow.analysis.plot import (
    plot_metric_by_position,
    plot_metric_by_role,
    plot_metric_distribution,
)

# =============================================================================
# Analysis Results
# =============================================================================


@dataclass
class SectionMetrics:
    """Metrics aggregated for a single section."""

    role: str
    content_type: str
    token_count: int
    start_idx: int  # Start position in token sequence
    end_idx: int    # End position (exclusive)
    metrics: Dict[str, float]  # metric_name -> mean value


@dataclass
class AnalysisResult:
    """Result of analyzing a single conversation."""

    conversation_id: str
    token_count: int
    sections: List[SectionMetrics]
    metrics_by_position: Dict[str, List[float]]  # metric_name -> values
    overall_metrics: Dict[str, float]  # metric_name -> mean over all tokens


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple conversations."""

    num_conversations: int
    total_tokens: int
    by_role: Dict[str, Dict[str, float]]  # role -> metric -> mean
    by_content_type: Dict[str, Dict[str, float]]  # content_type -> metric -> mean
    overall: Dict[str, float]  # metric -> mean
    by_position_normalized: Dict[str, List[float]]  # metric -> values at normalized positions


@dataclass
class ComparisonResult:
    """Result of comparing metrics before/after transformation."""

    conversation_id: str
    transform_name: str
    original_tokens: int
    transformed_tokens: int
    original_metrics: Dict[str, float]
    transformed_metrics: Dict[str, float]
    delta_metrics: Dict[str, float]  # transformed - original
    by_role_delta: Dict[str, Dict[str, float]]  # role -> metric -> delta


# =============================================================================
# Perplexity Analyzer
# =============================================================================


class PerplexityAnalyzer:
    """Main class for perplexity/entropy analysis of conversations."""

    def __init__(
        self,
        model,
        tokenizer,
        metrics: Optional[List[TokenMetric]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the analyzer.

        Args:
            model: HuggingFace model or RosettaModel.
            tokenizer: Tokenizer for the model.
            metrics: List of metrics to compute. Defaults to entropy, perplexity, neg_log_prob.
            device: Device for computation. If None, lets the model handle device
                placement (works with device_map="auto" for multi-GPU setups).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics or [EntropyMetric(), PerplexityMetric(), NegLogProbMetric()]
        self.device = device  # None means let model handle placement

    def analyze_conversation(
        self,
        conversation: TokenizedConversation,
        align_mode: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze a single tokenized conversation.

        Args:
            conversation: Tokenized conversation with section tracking.
            align_mode: If set, aligns all metrics to the same length.
                Options: "truncate" (trim longer to shorter), "pad_nan", "pad_zero".
                Default None keeps original lengths.

        Returns:
            AnalysisResult with per-token and per-section metrics.
        """
        input_ids = conversation.input_ids
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        # Compute metrics (model handles device placement if device is None)
        metric_values = prefill_and_compute_metrics(
            self.model,
            input_ids,
            self.metrics,
            device=self.device,
        )

        # Optionally align all metrics to same length
        if align_mode:
            metric_values = align_metrics(metric_values, self.metrics, mode=align_mode)

        # Build metric lookup for is_shifted property
        metric_lookup = {m.name: m for m in self.metrics}

        # Per-section aggregation
        section_metrics = []
        for section in conversation.sections:
            section_data = {"role": section.role, "content_type": section.content_type}
            section_data["token_count"] = section.length
            section_data["metrics"] = {}

            for metric_name, values in metric_values.items():
                # Use is_shifted property if available, else infer from length
                metric_obj = metric_lookup.get(metric_name)
                is_shifted = (
                    metric_obj.is_shifted if metric_obj else len(values) < conversation.seq_len
                )

                if is_shifted and not align_mode:
                    # Shifted metrics: value[i] predicts token[i+1]
                    # Map section indices to metric indices (shift by 1)
                    start = max(0, section.start_idx - 1)
                    end = min(len(values), section.end_idx - 1)
                else:
                    start = section.start_idx
                    end = section.end_idx

                if end > start:
                    section_values = values[start:end]
                    section_data["metrics"][metric_name] = section_values.mean().item()
                else:
                    section_data["metrics"][metric_name] = float("nan")

            section_metrics.append(
                SectionMetrics(
                    role=section.role,
                    content_type=section.content_type,
                    token_count=section.length,
                    start_idx=section.start_idx,
                    end_idx=section.end_idx,
                    metrics=section_data["metrics"],
                )
            )

        # Overall metrics
        overall = {}
        metrics_by_position = {}
        for metric_name, values in metric_values.items():
            overall[metric_name] = values.mean().item()
            metrics_by_position[metric_name] = values.tolist()

        return AnalysisResult(
            conversation_id=conversation.conversation_id or "unknown",
            token_count=conversation.seq_len,
            sections=section_metrics,
            metrics_by_position=metrics_by_position,
            overall_metrics=overall,
        )

    def analyze_file(
        self,
        path: Path,
        limit: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        exclude_final: bool = True,
    ) -> List[AnalysisResult]:
        """Analyze all conversations in a JSONL file.

        Args:
            path: Path to the evaluation JSONL file.
            limit: Maximum number of conversations to analyze.
            max_length: Maximum sequence length (skip longer).
            show_progress: Whether to show progress bar.
            tools: Optional list of tool schemas for section detection.
            exclude_final: If True, exclude final message to include all reasoning.

        Returns:
            List of AnalysisResult objects.
        """
        # Load and extract conversations
        records = load_evaluation_results(path)
        if limit:
            records = records[:limit]

        conversations = extract_conversations(records)

        # Use section-aware tokenizer for gpt-oss models
        tokenized = batch_tokenize_with_sections(
            conversations,
            self.tokenizer,
            tools=tools,
            max_length=max_length,
            show_progress=show_progress,
            exclude_final=exclude_final,
            convert_reasoning=True,
        )

        # Analyze each conversation
        results = []
        iterator = tokenized
        if show_progress:
            try:
                from rich.progress import track

                iterator = track(tokenized, description="Analyzing...")
            except ImportError:
                pass

        for conv in iterator:
            try:
                result = self.analyze_conversation(conv)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to analyze {conv.conversation_id}: {e}")
                continue

        return results

    def aggregate(self, results: List[AnalysisResult]) -> AggregatedMetrics:
        """Aggregate metrics across multiple conversations.

        Args:
            results: List of AnalysisResult objects.

        Returns:
            AggregatedMetrics with by-role and overall statistics.
        """
        if not results:
            return AggregatedMetrics(
                num_conversations=0,
                total_tokens=0,
                by_role={},
                by_content_type={},
                overall={},
                by_position_normalized={},
            )

        # Collect by role
        role_metrics: Dict[str, Dict[str, List[float]]] = {}
        content_type_metrics: Dict[str, Dict[str, List[float]]] = {}
        overall_values: Dict[str, List[float]] = {}

        total_tokens = 0
        for result in results:
            total_tokens += result.token_count

            # Overall
            for metric_name, value in result.overall_metrics.items():
                if metric_name not in overall_values:
                    overall_values[metric_name] = []
                overall_values[metric_name].append(value)

            # By section
            for section in result.sections:
                role = section.role
                ctype = section.content_type

                if role not in role_metrics:
                    role_metrics[role] = {}
                if ctype not in content_type_metrics:
                    content_type_metrics[ctype] = {}

                for metric_name, value in section.metrics.items():
                    if metric_name not in role_metrics[role]:
                        role_metrics[role][metric_name] = []
                    role_metrics[role][metric_name].append(value)

                    if metric_name not in content_type_metrics[ctype]:
                        content_type_metrics[ctype][metric_name] = []
                    content_type_metrics[ctype][metric_name].append(value)

        # Compute means
        by_role = {}
        for role, metrics in role_metrics.items():
            by_role[role] = {
                name: sum(values) / len(values) for name, values in metrics.items() if values
            }

        by_content_type = {}
        for ctype, metrics in content_type_metrics.items():
            by_content_type[ctype] = {
                name: sum(values) / len(values) for name, values in metrics.items() if values
            }

        overall = {
            name: sum(values) / len(values) for name, values in overall_values.items() if values
        }

        return AggregatedMetrics(
            num_conversations=len(results),
            total_tokens=total_tokens,
            by_role=by_role,
            by_content_type=by_content_type,
            overall=overall,
            by_position_normalized={},  # TODO: implement normalized position analysis
        )


# =============================================================================
# Comparison Analysis
# =============================================================================


class TransformComparison:
    """Compare metrics before/after context transformation."""

    def __init__(self, analyzer: PerplexityAnalyzer):
        self.analyzer = analyzer

    def compare(
        self,
        conversation: TokenizedConversation,
        transform_fn: Callable,
        transform_name: str = "transform",
    ) -> ComparisonResult:
        """Compare metrics before and after transformation.

        Args:
            conversation: Original tokenized conversation.
            transform_fn: Function that transforms message list.
            transform_name: Name of the transformation.

        Returns:
            ComparisonResult with original, transformed, and delta metrics.
        """
        # Analyze original
        original_result = self.analyzer.analyze_conversation(conversation)

        # Transform and re-tokenize
        transform_result = apply_context_transform(
            conversation,
            transform_fn,
            self.analyzer.tokenizer,
            transform_name,
        )

        # Analyze transformed
        transformed_result = self.analyzer.analyze_conversation(transform_result.transformed)

        # Compute deltas
        delta_metrics = {}
        for metric_name in original_result.overall_metrics:
            orig = original_result.overall_metrics.get(metric_name, 0)
            trans = transformed_result.overall_metrics.get(metric_name, 0)
            delta_metrics[metric_name] = trans - orig

        # By-role deltas
        by_role_delta = {}
        original_by_role = {}
        transformed_by_role = {}

        for section in original_result.sections:
            if section.role not in original_by_role:
                original_by_role[section.role] = {}
            for name, value in section.metrics.items():
                if name not in original_by_role[section.role]:
                    original_by_role[section.role][name] = []
                original_by_role[section.role][name].append(value)

        for section in transformed_result.sections:
            if section.role not in transformed_by_role:
                transformed_by_role[section.role] = {}
            for name, value in section.metrics.items():
                if name not in transformed_by_role[section.role]:
                    transformed_by_role[section.role][name] = []
                transformed_by_role[section.role][name].append(value)

        all_roles = set(original_by_role.keys()) | set(transformed_by_role.keys())
        for role in all_roles:
            by_role_delta[role] = {}
            orig_metrics = original_by_role.get(role, {})
            trans_metrics = transformed_by_role.get(role, {})
            all_metrics = set(orig_metrics.keys()) | set(trans_metrics.keys())
            for metric in all_metrics:
                orig_values = orig_metrics.get(metric, [0])
                trans_values = trans_metrics.get(metric, [0])
                orig_mean = sum(orig_values) / len(orig_values) if orig_values else 0
                trans_mean = sum(trans_values) / len(trans_values) if trans_values else 0
                by_role_delta[role][metric] = trans_mean - orig_mean

        return ComparisonResult(
            conversation_id=conversation.conversation_id or "unknown",
            transform_name=transform_name,
            original_tokens=original_result.token_count,
            transformed_tokens=transformed_result.token_count,
            original_metrics=original_result.overall_metrics,
            transformed_metrics=transformed_result.overall_metrics,
            delta_metrics=delta_metrics,
            by_role_delta=by_role_delta,
        )

# =============================================================================
# Output
# =============================================================================

def save_results(
    results: List[AnalysisResult],
    output_path: Path,
):
    """Save analysis results to JSONL.

    Args:
        results: List of analysis results.
        output_path: Path for output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            # Convert dataclass to dict, handling nested dataclasses
            data = {
                "conversation_id": result.conversation_id,
                "token_count": result.token_count,
                "overall_metrics": result.overall_metrics,
                "sections": [
                    {
                        "role": s.role,
                        "content_type": s.content_type,
                        "token_count": s.token_count,
                        "metrics": s.metrics,
                    }
                    for s in result.sections
                ],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def save_aggregated(
    aggregated: AggregatedMetrics,
    output_path: Path,
):
    """Save aggregated metrics to JSON.

    Args:
        aggregated: Aggregated metrics.
        output_path: Path for output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "num_conversations": aggregated.num_conversations,
        "total_tokens": aggregated.total_tokens,
        "by_role": aggregated.by_role,
        "by_content_type": aggregated.by_content_type,
        "overall": aggregated.overall,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated metrics to {output_path}")


def print_summary(aggregated: AggregatedMetrics):
    """Print a summary of aggregated metrics."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Conversations: {aggregated.num_conversations}")
    print(f"Total tokens: {aggregated.total_tokens}")

    print("\nOverall Metrics:")
    for metric, value in aggregated.overall.items():
        print(f"  {metric}: {value:.4f}")

    print("\nMetrics by Role:")
    for role, metrics in sorted(aggregated.by_role.items()):
        print(f"  {role}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    if aggregated.by_content_type:
        print("\nMetrics by Content Type:")
        for ctype, metrics in sorted(aggregated.by_content_type.items()):
            print(f"  {ctype}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze perplexity and entropy of conversation histories."
    )
    parser.add_argument(
        "--input",
        "-i",
        # required=True,
        default="local/evaluation/gpt_oss_20b/singletool/browsecomp/full_full_full/results.jsonl",
        help="Path to evaluation results JSONL file",
    )
    parser.add_argument(
        "--model",
        "-m",
        # required=True,
        default="/share/public/public_models/gpt-oss-20b",
        help="Model name or path (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="local/analysis/perplexity",
        help="Directory for output files",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["entropy", "perplexity", "neg_log_prob"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of conversations to analyze",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (skip longer)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--plot-data-csv",
        nargs="?",
        const="__AUTO__",
        default=None,
        help=(
            "Optional: write token-level plot source data to CSV (.csv or .csv.gz) for re-plotting. "
            "If provided without a path, writes to <output-dir>/plot_data.csv.gz."
        ),
    )
    parser.add_argument(
        "--transform",
        choices=["none", "summarize_tool", "summarize_all"],
        default="none",
        help="Apply context transformation before analysis",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include final message (default: exclude to include all reasoning)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    # Initialize metrics
    metrics = [get_metric_by_name(name) for name in args.metrics]

    # Initialize analyzer
    analyzer = PerplexityAnalyzer(model, tokenizer, metrics)

    # Get tool schemas from FunctionTool (browsecomp evaluation)
    default_tools = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    # Analyze
    print(f"\nAnalyzing: {input_path}")
    print(f"Exclude final message: {not args.include_final}")
    results = analyzer.analyze_file(
        input_path,
        limit=args.limit,
        max_length=args.max_length,
        tools=default_tools,
        exclude_final=not args.include_final,
    )

    if not results:
        print("No results to analyze")
        return

    # Aggregate
    aggregated = analyzer.aggregate(results)

    # Output
    save_results(results, output_dir / "results.jsonl")
    save_aggregated(aggregated, output_dir / "aggregated.json")
    print_summary(aggregated)

    if args.plot_data_csv:
        if args.plot_data_csv == "__AUTO__":
            plot_data_path = output_dir / "plot_data.csv.gz"
        else:
            plot_data_path = Path(args.plot_data_csv)
        save_token_plot_data_csv(results=results, metrics=metrics, output_path=plot_data_path)

    # Plots
    if not args.no_plot:
        for metric in args.metrics:
            plot_metric_by_role(
                aggregated,
                metric,
                output_dir / f"{metric}_by_role.png",
            )
            plot_metric_distribution(
                results,
                metric,
                output_dir / f"{metric}_distribution.png",
            )
            # New: plot metric by token position with role coloring
            plot_metric_by_position(
                results,
                metric,
                output_dir / f"{metric}_by_position.png",
                bin_size=50,
                show_range=True,
            )

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
