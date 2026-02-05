#!/usr/bin/env python3
"""
Filter evaluation results by token count.

Usage:
    python script/workflow/analysis/filter.py \
        --input local/evaluation/gpt_oss_20b/singletool/browsecomp/full_full_full/results.jsonl \
        --output local/evaluation/gpt_oss_20b/singletool/browsecomp/full_full_full/results_filtered.jsonl \
        --max-tokens 16000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer
from camel.toolkits import FunctionTool

from rosetta.workflow.analysis.interface import (
    extract_conversations,
    load_evaluation_results,
)
from rosetta.workflow.analysis.oss_tokenizer import tokenize_conversation_with_sections
from rosetta.workflow.browse_searcher import get_document, search


def get_token_count(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: List[Dict[str, Any]],
) -> int:
    """Get token count for a conversation."""
    try:
        conv = tokenize_conversation_with_sections(
            messages,
            tokenizer,
            tools=tools,
            exclude_final=True,
            convert_reasoning=True,
        )
        return conv.seq_len
    except Exception as e:
        print(f"Warning: Failed to tokenize: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(description="Filter evaluation results by token count.")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-oss-20b",
        help="Tokenizer model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count (default: no limit)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Tool schemas for tokenization
    tools = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    print(f"Loading: {input_path}")
    records = load_evaluation_results(input_path)
    print(f"Loaded {len(records)} records")

    # Extract conversations and compute token counts
    conversations = extract_conversations(records)

    # Build id -> token_count mapping
    id_to_tokens: Dict[str, int] = {}

    try:
        from rich.progress import track
        iterator = track(conversations, description="Counting tokens...")
    except ImportError:
        iterator = conversations

    for conv_id, messages in iterator:
        token_count = get_token_count(messages, tokenizer, tools)
        id_to_tokens[conv_id] = token_count

    # Filter records
    filtered_records = []
    for record in records:
        example_id = str(record.get("example_id", record.get("idx", "")))
        token_count = id_to_tokens.get(example_id, -1)
        pred_raw = record.get("pred_raw", "")

        # Must have pred_raw
        if not pred_raw:
            continue
        # Token filter (if specified)
        if args.max_tokens and token_count >= args.max_tokens:
            continue
        if token_count <= 0:
            continue

        filtered_records.append(record)

    if args.max_tokens:
        print(f"\nFiltered: {len(filtered_records)}/{len(records)} records (< {args.max_tokens} tokens)")
    else:
        print(f"\nFiltered: {len(filtered_records)}/{len(records)} records (with pred_raw)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in filtered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")

    # Print stats
    kept_tokens = [id_to_tokens[str(r.get("example_id", r.get("idx", "")))] for r in filtered_records]
    if kept_tokens:
        print(f"Token range: {min(kept_tokens)} - {max(kept_tokens)}")
        print(f"Mean tokens: {sum(kept_tokens) / len(kept_tokens):.0f}")


if __name__ == "__main__":
    main()
