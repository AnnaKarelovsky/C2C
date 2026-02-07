#!/usr/bin/env python3
"""
Perplexity and entropy analysis for conversation histories.

Loads evaluation results, prefills sequences through a model, and analyzes
how entropy/perplexity varies by token position and message type.

Supports context transformations with UID tracking for cross-run comparison.
Summaries are cached to disk, so re-running with different delay values
does not require re-summarization.

Usage:
    # Local HuggingFace model
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b

    # Fireworks API
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b \
        --backend fireworks

    # With context transformation (summaries are cached)
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b \
        --context-config full_full_sum_0 \
        --context-model accounts/fireworks/models/qwen3-235b-a22b-instruct-2507

    # Re-run with different delay (reuses cached summaries)
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b \
        --context-config full_full_sum_1

    # With none mode (drop content instead of summarize)
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b \
        --context-config full_full_none_0

Context Config Format:
    {reasoning}_{assistant}_{tool}_{delay}

    Modes:
        full - Keep original content
        sum  - Summarize content (requires --context-model)
        none - Drop content (replace with placeholder)

    Examples:
        full_full_full_0  - Baseline (no transformation)
        full_full_sum_0   - Summarize tool responses
        full_full_none_0  - Drop tool responses
        none_full_sum_1   - Drop reasoning, summarize tools, delay=1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from camel.toolkits import FunctionTool

from rosetta.workflow.analysis.interface import (
    AggregatedMetrics,
    AnalysisResult,
    aggregate_results,
    analyze_conversation,
    extract_conversations,
    load_evaluation_results,
    save_token_plot_data_csv,
    save_transform_log_csv,
)
from rosetta.utils.core import (
    EstimatedEntropyUnified,
    NegLogProbUnified,
    PerplexityUnified,
    Top1NegLogProbUnified,
    TopKEntropyUnified,
    TopKMassUnified,
)
from rosetta.workflow.analysis.oss_tokenizer import batch_tokenize_with_sections
from rosetta.workflow.analysis.plot import (
    plot_metric_by_norm_section_from_csv,
    plot_metric_by_position,
    plot_metric_by_position_from_csv,
    plot_metric_by_role,
    plot_metric_by_section_from_csv,
    plot_metric_distribution,
    plot_metric_scatter_subplots_by_role_from_csv,
)
from rosetta.workflow.analysis.uid_tracking import assign_message_uids
from rosetta.workflow.analysis.summary_cache import apply_context_transformations
from rosetta.workflow.browse_searcher import get_document, search
from rosetta.workflow.camel_utils import create_model

try:
    from rich.progress import track
except ImportError:
    track = None


# =============================================================================
# Output
# =============================================================================


def save_results(results: List[AnalysisResult], output_path: Path):
    """Save analysis results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
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


def save_aggregated(aggregated: AggregatedMetrics, output_path: Path):
    """Save aggregated metrics to JSON."""
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

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze perplexity and entropy of conversation histories."
    )
    parser.add_argument(
        "--input", "-i",
        default="local/evaluation/gpt_oss_20b/singletool/browsecomp/full_full_full/results.jsonl",
        help="Path to evaluation results JSONL file",
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-oss-20b",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="local/analysis/perplexity",
        help="Directory for output files",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "fireworks"],
        default="local",
        help="Backend: local (HuggingFace) or fireworks (API)",
    )
    parser.add_argument(
        "--fireworks-api-key",
        default=None,
        help="Fireworks API key (or set FIREWORKS_API_KEY env var)",
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
        "--no-csv",
        action="store_true",
        help="Skip generating CSV output",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include final message (default: exclude to include all reasoning)",
    )
    parser.add_argument(
        "--context-config",
        type=str,
        default=None,
        help="Context config string: {reasoning}_{assistant}_{tool}_{delay} "
             "(e.g., full_full_sum_0 for summarized tools)",
    )
    parser.add_argument(
        "--context-model",
        type=str,
        default=None,
        help="Model for summarization (Fireworks format, e.g., accounts/fireworks/models/qwen3-235b-a22b-instruct-2507)",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Force rebuilding the summary cache even if it exists",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=25,
        help="Maximum concurrent summarization calls (default: 25)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for summary cache (default: same as output-dir). "
             "Use a shared directory to reuse summaries across runs with different delays.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load model for local backend
    model = None
    fireworks_model = None
    if args.backend == "local":
        print(f"Loading model: {args.model}")

        # Check for quantization in model config and warn
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        quant_config = getattr(model_config, "quantization_config", None)
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown") if isinstance(quant_config, dict) else getattr(quant_config, "quant_method", "unknown")
            print(f"\nWARNING: Model uses {quant_method} quantization.")
            print("Logprobs from quantized models may differ significantly from")
            print("full-precision inference (e.g., Fireworks API). Consider using")
            print("--backend fireworks for more accurate perplexity analysis.\n")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
    else:
        # Convert HuggingFace model name to Fireworks format
        if args.model.startswith("accounts/"):
            fireworks_model = args.model
        else:
            model_name = args.model.split("/")[-1]
            # Fireworks model names are typically lowercase
            fireworks_model = f"accounts/fireworks/models/{model_name.lower()}"
        print(f"Using Fireworks API with model: {fireworks_model}")

    # Determine run name from context config
    run_name = args.context_config if args.context_config else "baseline"
    print(f"Run name: {run_name}")

    # Load and tokenize conversations
    print(f"\nLoading: {input_path}")
    records = load_evaluation_results(input_path)
    if args.limit:
        records = records[:args.limit]
    conversations = extract_conversations(records)

    # Apply context transformations if specified
    transform_logs = None
    if args.context_config:
        print(f"\nApplying context config: {args.context_config}")

        # Create context model if needed
        context_model = None
        if args.context_model:
            context_model = create_model(
                "fireworks",
                model_type=args.context_model,
                temperature=0.0,
                max_tokens=32768,
                stream=True,
            )
            print(f"Using context model: {args.context_model}")

        # Use dedicated cache_dir if provided, otherwise use output_dir
        cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        conversations, transform_logs = apply_context_transformations(
            conversations,
            args.context_config,
            context_model,
            show_progress=True,
            cache_dir=cache_dir,
            input_path=input_path,
            force_rebuild_cache=args.force_rebuild_cache,
            max_concurrency=args.max_concurrency,
        )

        # Count transformations
        total_transforms = sum(len(logs) for _, logs in transform_logs)
        print(f"Applied {total_transforms} transformations across {len(conversations)} conversations")
    else:
        # Assign UIDs even without transformation for consistent tracking
        new_conversations = []
        transform_logs = []
        for conv_id, messages in conversations:
            conv = assign_message_uids(messages, conv_id)
            new_conversations.append((conv_id, conv.messages))
            transform_logs.append((conv_id, []))
        conversations = new_conversations

    # Tool schemas for section detection
    default_tools = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    tokenized = batch_tokenize_with_sections(
        conversations,
        tokenizer,
        tools=default_tools,
        max_length=args.max_length,
        show_progress=True,
        exclude_final=not args.include_final,
        convert_reasoning=True,
    )

    # Analyze each conversation
    analysis_model = fireworks_model if args.backend == "fireworks" else args.model
    print(f"\nAnalyzing {len(tokenized)} conversations with {analysis_model}...")
    results = []
    if track:
        iterator = track(tokenized, description=f"Analyzing ({analysis_model})...")
    else:
        iterator = tokenized

    for conv in iterator:
        try:
            result = analyze_conversation(
                conv,
                backend=args.backend,
                model=model,
                fireworks_model=fireworks_model,
                fireworks_api_key=args.fireworks_api_key,
            )
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to analyze {conv.conversation_id}: {e}")
            continue

    if not results:
        print("No results to analyze")
        return

    # Aggregate and output
    aggregated = aggregate_results(results)
    save_results(results, output_dir / f"{run_name}_results.jsonl")
    save_aggregated(aggregated, output_dir / f"{run_name}_aggregated.json")
    print_summary(aggregated)

    # Save config info
    config_info = {
        "context_config": args.context_config or "baseline",
        "context_model": args.context_model,
        "model": args.model,
        "backend": args.backend,
        "input": str(input_path),
        "limit": args.limit,
        "max_length": args.max_length,
        "include_final": args.include_final,
    }
    config_path = output_dir / f"{run_name}_config.json"
    with config_path.open("w") as f:
        json.dump(config_info, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save transform log
    if transform_logs:
        transform_log_path = output_dir / f"{run_name}_transforms.csv"
        save_transform_log_csv(transform_logs, transform_log_path)

    # CSV export
    csv_path = output_dir / f"{run_name}_metrics.csv"
    if not args.no_csv:
        csv_metrics = [
            NegLogProbUnified(),
            Top1NegLogProbUnified(),    # Confidence in best prediction
            PerplexityUnified(),
            EstimatedEntropyUnified(),  # Exact for HF, tight lower bound for API
            TopKEntropyUnified(),       # Renormalized top-k entropy (comparable across backends)
            TopKMassUnified(),
        ]
        save_token_plot_data_csv(
            results=results,
            metrics=csv_metrics,
            output_path=csv_path,
            include_uid=True,
        )

    # Plots
    if not args.no_plot:
        metric_names = list(results[0].overall_metrics.keys()) if results else []
        # Exclude perplexity plots by default (slow to render, less useful)
        metric_names = [m for m in metric_names if "perplexity" not in m.lower()]
        for metric in metric_names:
            plot_metric_by_role(aggregated, metric, output_dir / f"{metric}_by_role.png")
            plot_metric_distribution(results, metric, output_dir / f"{metric}_distribution.png")
            plot_metric_by_position(results, metric, output_dir / f"{metric}_by_position.png")

        # CSV-based plots (if CSV was generated)
        if not args.no_csv and csv_path.exists():
            for metric in metric_names:
                # By position with only assistant and tool (subplots)
                plot_metric_by_position_from_csv(
                    csv_path=csv_path,
                    metric_name=metric,
                    output_path=output_dir / f"{metric}_by_position_assistant_tool.png",
                    roles=["assistant", "tool"],
                )
                # Scatter plots
                plot_metric_scatter_subplots_by_role_from_csv(
                    csv_path=csv_path,
                    metric_name=metric,
                    output_path=output_dir / f"{metric}_scatter_by_role.png",
                    roles=["assistant", "tool"],
                )
                # By section index (content_type as legend)
                plot_metric_by_section_from_csv(
                    csv_path=csv_path,
                    metric_name=metric,
                    output_path=output_dir / f"{metric}_by_section.png",
                )
                # By normalized section index (content_type as legend)
                plot_metric_by_norm_section_from_csv(
                    csv_path=csv_path,
                    metric_name=metric,
                    output_path=output_dir / f"{metric}_by_norm_section.png",
                )

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
