#!/usr/bin/env python3
"""Collect a tool-use trajectory for testing API consistency.

This script runs an agent with search/get_document tools and collects
the full message trajectory, which can be used as example messages
in test_api_consistency.py.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer

from camel.toolkits import FunctionTool

from rosetta.workflow.basic_utils import ContentMode, HistoryConfig, msg_system, msg_user
from rosetta.workflow.track import InteractionTracker
from rosetta.workflow.display import ConvLogger
from rosetta.workflow.singletool import run_with_tools
from rosetta.workflow.browse_searcher import configure_search, search, get_document
from rosetta.workflow.camel_utils import create_model

load_dotenv(find_dotenv())


def collect_trajectory(max_iterations: int = 3):
    """Run agent with tools and collect the trajectory."""

    # Use gpt-oss model for testing
    model = create_model(
        "fireworks",
        model_type="accounts/fireworks/models/gpt-oss-20b",
        stream=True,
        temperature=0.0,
        max_tokens=4096,
    )
    tokenizer_model_name = "openai/gpt-oss-20b"

    # Configure search (BrowseCompPlus)
    configure_search(
        index_path="local/data/BrowseCompPlus/indexes/qwen3-embedding-8b/corpus.*.pkl",
        dataset_name="Tevatron/browsecomp-plus-corpus",
        sglang_url="http://localhost:30001",
        sglang_model="Qwen/Qwen3-Embedding-8B",
        task_prefix="Query: ",
    )

    tools = [FunctionTool(search), FunctionTool(get_document)]

    # Question that should trigger tool use (requires searching)
    question = "Search for information about the Eiffel Tower and tell me when it was built."

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)
    tracker = InteractionTracker(tokenizer=tokenizer)
    logger = ConvLogger(tokenizer=tokenizer)

    config = HistoryConfig(
        assistant=ContentMode.FULL,
        tool=ContentMode.FULL,
        reasoning=ContentMode.FULL,
        delay=0,
    )

    print(f"Running agent with max_iterations={max_iterations}...")
    print("=" * 60)

    messages = [msg_system("You are a helpful assistant."), msg_user(question)]
    answer, messages, tracker = run_with_tools(
        messages, model, tools,
        tracker=tracker, logger=logger, ctx_manager=None,
        max_iterations=max_iterations
    )

    print("\n" + "=" * 60)
    print("Final Answer:", answer)
    print("=" * 60)

    # Collect the trajectory
    messages = tracker.final_messages

    print(f"\nCollected {len(messages)} messages:")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        reasoning = msg.get("reasoning_content", "")

        print(f"\n[{i}] {role}:")
        if reasoning:
            print(f"  reasoning: {reasoning[:100]}...")
        if content:
            print(f"  content: {content[:100]}..." if len(str(content)) > 100 else f"  content: {content}")
        if tool_calls:
            print(f"  tool_calls: {len(tool_calls)} call(s)")
            for tc in tool_calls:
                if isinstance(tc, dict):
                    print(f"    - {tc.get('function', {}).get('name', 'unknown')}")

    # Get tool schemas
    tool_schemas = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    # Output as JSON for use in tests
    output = {
        "messages": messages,
        "tools": tool_schemas,
    }

    output_path = Path(__file__).parent / "example_trajectory.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nTrajectory saved to: {output_path}")

    return messages, tool_schemas


if __name__ == "__main__":
    collect_trajectory(max_iterations=3)
