"""Summary caching for context transformations.

This module provides:
- Pre-computation and caching of summaries for messages
- Loading cached summaries and applying them with different delay values
- Separation of summarization (expensive LLM calls) from transformation (cheap)

The cache file name encodes modes but NOT delay, allowing reuse across delays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rosetta.workflow.basic_utils import ContentMode, HistoryConfig

try:
    from rich.progress import track
except ImportError:
    track = None


@dataclass
class SummaryEntry:
    """A cached summary for a single message field."""

    conversation_id: str
    message_uid: int
    role: str  # "assistant" or "tool"
    field: str  # "reasoning", "content"
    original_content: str
    summarized_content: str
    original_char_count: int = 0
    summarized_char_count: int = 0

    def __post_init__(self):
        if self.original_char_count == 0:
            self.original_char_count = len(self.original_content)
        if self.summarized_char_count == 0:
            self.summarized_char_count = len(self.summarized_content)


@dataclass
class SummaryCache:
    """Cache of all summaries for a set of conversations."""

    # Modes used for this cache (delay is ignored)
    reasoning_mode: ContentMode
    assistant_mode: ContentMode
    tool_mode: ContentMode

    # Context model used for summarization
    context_model: Optional[str]

    # Entries indexed by (conversation_id, message_uid, field)
    entries: Dict[Tuple[str, int, str], SummaryEntry]

    # Metadata
    num_conversations: int = 0
    num_entries: int = 0

    def get(
        self, conversation_id: str, message_uid: int, field: str
    ) -> Optional[SummaryEntry]:
        """Get a cached summary entry."""
        return self.entries.get((conversation_id, message_uid, field))

    def add(self, entry: SummaryEntry):
        """Add an entry to the cache."""
        key = (entry.conversation_id, entry.message_uid, entry.field)
        self.entries[key] = entry
        self.num_entries = len(self.entries)


def get_cache_key(config: HistoryConfig) -> str:
    """Get cache key from config (excludes delay).

    Args:
        config: HistoryConfig object.

    Returns:
        String like "sum_full_sum" (reasoning_assistant_tool, no delay).
    """
    mode_str = {
        ContentMode.FULL: "full",
        ContentMode.NONE: "none",
        ContentMode.SUMMARIZED: "sum",
    }
    return f"{mode_str[config.reasoning]}_{mode_str[config.assistant]}_{mode_str[config.tool]}"


def get_cache_path(
    output_dir: Path,
    input_path: Path,
    config: HistoryConfig,
) -> Path:
    """Get the cache file path for a given config.

    Args:
        output_dir: Directory for output files.
        input_path: Input file path (used for naming).
        config: HistoryConfig object.

    Returns:
        Path to cache file.
    """
    input_stem = input_path.stem
    cache_key = get_cache_key(config)
    return output_dir / f"{input_stem}_summaries_{cache_key}.jsonl"


def needs_summarization(config: HistoryConfig) -> bool:
    """Check if config requires any summarization.

    Args:
        config: HistoryConfig object.

    Returns:
        True if any mode is SUMMARIZED.
    """
    return (
        config.reasoning == ContentMode.SUMMARIZED
        or config.assistant == ContentMode.SUMMARIZED
        or config.tool == ContentMode.SUMMARIZED
    )


def save_summary_cache(cache: SummaryCache, path: Path):
    """Save summary cache to JSONL file.

    Args:
        cache: SummaryCache object.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        # Write header with metadata
        header = {
            "_type": "header",
            "reasoning_mode": cache.reasoning_mode.value,
            "assistant_mode": cache.assistant_mode.value,
            "tool_mode": cache.tool_mode.value,
            "context_model": cache.context_model,
            "num_conversations": cache.num_conversations,
            "num_entries": cache.num_entries,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Write entries
        for entry in cache.entries.values():
            data = asdict(entry)
            data["_type"] = "entry"
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Saved {cache.num_entries} summary entries to {path}")


def load_summary_cache(path: Path) -> Optional[SummaryCache]:
    """Load summary cache from JSONL file.

    Args:
        path: Path to cache file.

    Returns:
        SummaryCache object, or None if file doesn't exist.
    """
    if not path.exists():
        return None

    entries = {}
    header = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("_type") == "header":
                header = data
            elif data.get("_type") == "entry":
                data.pop("_type")
                entry = SummaryEntry(**data)
                key = (entry.conversation_id, entry.message_uid, entry.field)
                entries[key] = entry

    if header is None:
        return None

    cache = SummaryCache(
        reasoning_mode=ContentMode(header["reasoning_mode"]),
        assistant_mode=ContentMode(header["assistant_mode"]),
        tool_mode=ContentMode(header["tool_mode"]),
        context_model=header.get("context_model"),
        entries=entries,
        num_conversations=header.get("num_conversations", 0),
        num_entries=len(entries),
    )

    print(f"Loaded {cache.num_entries} summary entries from {path}")
    return cache


def _is_tool_message(msg: Dict) -> bool:
    """Check if message is a tool response."""
    return msg.get("role") == "tool" or "tool_call_id" in msg


@dataclass
class _SummarizeTask:
    """A single summarization task."""
    conv_id: str
    uid: int
    role: str
    field: str
    original_content: str
    msg: Dict[str, Any]  # Full message for context


def build_summary_cache(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    config: HistoryConfig,
    context_model,
    show_progress: bool = True,
    max_concurrency: int = 25,
) -> SummaryCache:
    """Build summary cache by summarizing all messages that need it.

    This function iterates through ALL messages in ALL conversations and
    summarizes those that match the config's SUMMARIZED modes. The delay
    is ignored - all eligible messages are summarized.

    Args:
        conversations: List of (conversation_id, messages) tuples.
                      Messages should have _uid assigned.
        config: HistoryConfig specifying which fields to summarize.
        context_model: CAMEL model backend for summarization.
        show_progress: Whether to show progress bar.
        max_concurrency: Maximum number of concurrent summarization calls.

    Returns:
        SummaryCache with all summaries.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rosetta.workflow.contextManage import (
        inject_call_context,
        summarize_content,
        summarize_reasoning,
        summarize_tool_resp,
    )

    cache = SummaryCache(
        reasoning_mode=config.reasoning,
        assistant_mode=config.assistant,
        tool_mode=config.tool,
        context_model=str(context_model) if context_model else None,
        entries={},
        num_conversations=len(conversations),
    )

    # Nothing to summarize
    if not needs_summarization(config):
        return cache

    # Collect all tasks first
    tasks: List[_SummarizeTask] = []

    for conv_id, messages in conversations:
        # Inject call context for tool messages
        messages = [dict(m) for m in messages]  # Copy to avoid modifying original
        inject_call_context(messages)

        for msg in messages:
            uid = msg.get("_uid", -1)
            role = msg.get("role", "")

            # Handle assistant messages
            if role == "assistant":
                # Summarize reasoning if needed
                reasoning = msg.get("_reasoning", "") or msg.get("reasoning_content", "")
                if reasoning and config.reasoning == ContentMode.SUMMARIZED:
                    tasks.append(_SummarizeTask(
                        conv_id=conv_id,
                        uid=uid,
                        role="assistant",
                        field="reasoning",
                        original_content=reasoning,
                        msg={"content": reasoning},
                    ))

                # Summarize content if needed
                content = msg.get("content", "")
                if content and config.assistant == ContentMode.SUMMARIZED:
                    tasks.append(_SummarizeTask(
                        conv_id=conv_id,
                        uid=uid,
                        role="assistant",
                        field="content",
                        original_content=content,
                        msg=msg,
                    ))

            # Handle tool messages
            elif _is_tool_message(msg):
                content = msg.get("content", "")
                if content and config.tool == ContentMode.SUMMARIZED:
                    tasks.append(_SummarizeTask(
                        conv_id=conv_id,
                        uid=uid,
                        role="tool",
                        field="content",
                        original_content=content,
                        msg=msg,
                    ))

    if not tasks:
        return cache

    def execute_task(task: _SummarizeTask) -> SummaryEntry:
        """Execute a single summarization task."""
        if task.role == "assistant" and task.field == "reasoning":
            result = summarize_reasoning([task.msg], context_model)
        elif task.role == "assistant" and task.field == "content":
            result = summarize_content([task.msg], context_model)
        else:  # tool
            result = summarize_tool_resp([task.msg], context_model)

        summarized = result[0].get("content", "")
        return SummaryEntry(
            conversation_id=task.conv_id,
            message_uid=task.uid,
            role=task.role,
            field=task.field,
            original_content=task.original_content,
            summarized_content=summarized,
        )

    # Execute tasks concurrently
    print(f"Summarizing {len(tasks)} messages with concurrency={max_concurrency}...")

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(execute_task, task): task for task in tasks}

        if show_progress and track:
            iterator = track(
                as_completed(futures),
                total=len(futures),
                description="Building summary cache...",
            )
        else:
            iterator = as_completed(futures)

        for future in iterator:
            try:
                entry = future.result()
                cache.add(entry)
            except Exception as e:
                task = futures[future]
                print(f"Warning: Failed to summarize {task.conv_id}:{task.uid}: {e}")

    return cache


def get_or_build_summary_cache(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    config: HistoryConfig,
    context_model,
    cache_path: Path,
    show_progress: bool = True,
    force_rebuild: bool = False,
    max_concurrency: int = 25,
) -> SummaryCache:
    """Get summary cache from file or build it.

    Supports additive rebuilding: if existing cache is missing some conversations
    (e.g., from a previous run with --limit), only the missing conversations are
    summarized and added to the cache.

    Args:
        conversations: List of (conversation_id, messages) tuples.
        config: HistoryConfig specifying modes.
        context_model: CAMEL model for summarization.
        cache_path: Path to cache file.
        show_progress: Whether to show progress bar.
        force_rebuild: If True, rebuild from scratch even if cache exists.
        max_concurrency: Maximum number of concurrent summarization calls.

    Returns:
        SummaryCache object.
    """
    # Try to load existing cache
    existing_cache = None
    if not force_rebuild:
        existing_cache = load_summary_cache(cache_path)
        if existing_cache is not None:
            # Verify modes match
            if not (
                existing_cache.reasoning_mode == config.reasoning
                and existing_cache.assistant_mode == config.assistant
                and existing_cache.tool_mode == config.tool
            ):
                print("Cache modes don't match, rebuilding from scratch...")
                existing_cache = None

    # Find which conversations are missing from existing cache
    if existing_cache is not None:
        cached_conv_ids = set(
            entry.conversation_id for entry in existing_cache.entries.values()
        )
        missing_conversations = [
            (conv_id, msgs)
            for conv_id, msgs in conversations
            if conv_id not in cached_conv_ids
        ]

        if not missing_conversations:
            print(f"Using existing summary cache: {cache_path} (all {len(conversations)} conversations covered)")
            return existing_cache
        else:
            print(
                f"Cache has {len(cached_conv_ids)} conversations, "
                f"need {len(missing_conversations)} more (additive rebuild)"
            )
    else:
        missing_conversations = conversations

    # Build summaries for missing conversations
    new_cache = build_summary_cache(
        missing_conversations, config, context_model, show_progress, max_concurrency
    )

    # Merge with existing cache if applicable
    if existing_cache is not None:
        # Add new entries to existing cache
        for entry in new_cache.entries.values():
            existing_cache.add(entry)
        existing_cache.num_conversations = len(conversations)
        cache = existing_cache
        print(f"Merged: cache now has {cache.num_entries} entries")
    else:
        new_cache.num_conversations = len(conversations)
        cache = new_cache

    save_summary_cache(cache, cache_path)
    return cache


def apply_cached_transforms(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    config: HistoryConfig,
    cache: SummaryCache,
    show_progress: bool = True,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[Tuple[str, List[Any]]]]:
    """Apply context transformations using cached summaries.

    This function applies transformations based on the config (including delay)
    but uses pre-computed summaries from the cache instead of calling the LLM.

    Args:
        conversations: List of (conversation_id, messages) tuples.
                      Messages should have _uid assigned.
        config: HistoryConfig specifying transformations AND delay.
        cache: SummaryCache with pre-computed summaries.
        show_progress: Whether to show progress bar.

    Returns:
        Tuple of:
        - List of (conversation_id, transformed_messages) tuples
        - List of (conversation_id, transform_records) tuples
    """
    from rosetta.workflow.analysis.uid_tracking import TransformRecord
    from rosetta.workflow.contextManage import _find_round_boundaries, inject_call_context

    # Check if any transformation is needed
    is_default = (
        config.reasoning == ContentMode.FULL
        and config.assistant == ContentMode.FULL
        and config.tool == ContentMode.FULL
    )

    if is_default:
        # No transformation needed
        results = []
        transform_logs = []
        for conv_id, messages in conversations:
            results.append((conv_id, messages))
            transform_logs.append((conv_id, []))
        return results, transform_logs

    results = []
    transform_logs = []

    iterator = conversations
    if show_progress and track:
        iterator = track(conversations, description="Applying cached transforms...")

    for conv_id, messages in iterator:
        # Deep copy messages
        messages = [dict(m) for m in messages]
        inject_call_context(messages)

        # Find round boundaries
        round_boundaries = _find_round_boundaries(messages)
        if not round_boundaries:
            results.append((conv_id, messages))
            transform_logs.append((conv_id, []))
            continue

        # Determine which rounds to transform based on delay
        rounds_to_transform = len(round_boundaries) - config.delay

        if rounds_to_transform <= 0:
            results.append((conv_id, messages))
            transform_logs.append((conv_id, []))
            continue

        transform_log = []
        uid_counter = max((m.get("_uid", 0) for m in messages), default=0) + 1

        # Process each round that should be transformed
        for round_idx in range(rounds_to_transform):
            start, end = round_boundaries[round_idx]

            for msg_idx in range(start, end):
                msg = messages[msg_idx]
                original_uid = msg.get("_uid", -1)
                role = msg.get("role", "unknown")

                transformed = False
                new_msg = dict(msg)
                original_len = len(msg.get("content", "")) + len(
                    msg.get("_reasoning", "") or msg.get("reasoning_content", "")
                )

                # Apply assistant transformations
                if role == "assistant":
                    # Handle reasoning
                    reasoning = msg.get("_reasoning", "") or msg.get("reasoning_content", "")
                    if reasoning:
                        if config.reasoning == ContentMode.NONE:
                            new_msg.pop("reasoning_content", None)
                            new_msg["_reasoning"] = ""
                            transformed = True
                        elif config.reasoning == ContentMode.SUMMARIZED:
                            entry = cache.get(conv_id, original_uid, "reasoning")
                            if entry:
                                if "reasoning_content" in msg:
                                    new_msg["reasoning_content"] = entry.summarized_content
                                new_msg["_reasoning"] = entry.summarized_content
                                transformed = True

                    # Handle assistant content
                    content = msg.get("content", "")
                    if content:
                        if config.assistant == ContentMode.NONE:
                            new_msg["content"] = ""
                            transformed = True
                        elif config.assistant == ContentMode.SUMMARIZED:
                            entry = cache.get(conv_id, original_uid, "content")
                            if entry:
                                new_msg["content"] = entry.summarized_content
                                transformed = True

                # Apply tool transformations
                elif _is_tool_message(msg):
                    content = msg.get("content", "")
                    if content:
                        if config.tool == ContentMode.NONE:
                            new_msg["content"] = "[executed]"
                            transformed = True
                        elif config.tool == ContentMode.SUMMARIZED:
                            entry = cache.get(conv_id, original_uid, "content")
                            if entry:
                                new_msg["content"] = entry.summarized_content
                                transformed = True

                # If transformed, assign new UID and record
                if transformed:
                    new_uid = uid_counter
                    uid_counter += 1

                    new_msg["_uid"] = new_uid
                    new_msg["_original_uid"] = msg.get("_original_uid", original_uid)

                    new_len = len(new_msg.get("content", "")) + len(
                        new_msg.get("_reasoning", "") or new_msg.get("reasoning_content", "")
                    )

                    # Determine transform type based on what was actually applied
                    if role == "assistant":
                        # Check if any assistant field was set to NONE
                        if config.reasoning == ContentMode.NONE or config.assistant == ContentMode.NONE:
                            transform_type = "none"
                        else:
                            transform_type = "summarized"
                    elif _is_tool_message(msg):
                        transform_type = "none" if config.tool == ContentMode.NONE else "summarized"
                    else:
                        transform_type = "summarized"

                    transform_log.append(
                        TransformRecord(
                            original_uid=original_uid,
                            new_uid=new_uid,
                            transform_type=transform_type,
                            role=role,
                            original_char_count=original_len,
                            new_char_count=new_len,
                        )
                    )

                    messages[msg_idx] = new_msg

        results.append((conv_id, messages))
        transform_logs.append((conv_id, transform_log))

    return results, transform_logs


def apply_context_transformations(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    config_str: str,
    context_model=None,
    show_progress: bool = True,
    cache_dir: Optional[Path] = None,
    input_path: Optional[Path] = None,
    force_rebuild_cache: bool = False,
    max_concurrency: int = 25,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[Tuple[str, List[Any]]]]:
    """Apply context transformations with UID tracking and caching.

    High-level function that handles the full transformation pipeline:
    1. Assigns UIDs to all messages
    2. Builds or loads summary cache (if summarization is needed)
    3. Applies transformations based on config (including delay)

    If summarization is needed and cache_dir/input_path are provided, summaries
    are cached to disk. When re-running with different delay values, cached
    summaries are reused (no LLM calls needed).

    Args:
        conversations: List of (conversation_id, messages) tuples.
        config_str: Context config string (e.g., "full_full_sum_0").
        context_model: CAMEL model for summarization (required if using SUMMARIZED).
        show_progress: Whether to show progress bar.
        cache_dir: Directory for caching summaries (enables caching if provided).
        input_path: Input file path (used for cache naming).
        force_rebuild_cache: If True, rebuild cache even if it exists.
        max_concurrency: Maximum number of concurrent summarization calls.

    Returns:
        Tuple of:
        - List of (conversation_id, transformed_messages) tuples
        - List of (conversation_id, transform_records) tuples
    """
    from rosetta.workflow.analysis.uid_tracking import (
        ConversationWithUID,
        apply_transform_with_tracking,
        assign_message_uids,
        parse_context_config,
    )

    config = parse_context_config(config_str)

    # First, assign UIDs to all messages
    conversations_with_uids = []
    for conv_id, messages in conversations:
        conv = assign_message_uids(messages, conv_id)
        conversations_with_uids.append((conv_id, conv.messages))

    # Check if any transformation is needed
    is_default = (
        config.reasoning == ContentMode.FULL
        and config.assistant == ContentMode.FULL
        and config.tool == ContentMode.FULL
    )

    if is_default:
        # No transformation needed, just return with UIDs
        transform_logs = [(conv_id, []) for conv_id, _ in conversations_with_uids]
        return conversations_with_uids, transform_logs

    # Check if summarization is needed
    requires_summarization = needs_summarization(config)

    if requires_summarization and context_model is None:
        raise ValueError(
            f"Context config '{config_str}' requires SUMMARIZED mode but no context_model provided"
        )

    # Use caching if summarization is needed and cache_dir is provided
    if requires_summarization and cache_dir is not None and input_path is not None:
        cache_path = get_cache_path(cache_dir, input_path, config)
        print(f"Summary cache path: {cache_path}")

        # Get or build summary cache
        cache = get_or_build_summary_cache(
            conversations_with_uids,
            config,
            context_model,
            cache_path,
            show_progress=show_progress,
            force_rebuild=force_rebuild_cache,
            max_concurrency=max_concurrency,
        )

        # Apply transformations using cached summaries
        return apply_cached_transforms(
            conversations_with_uids,
            config,
            cache,
            show_progress=show_progress,
        )

    # Fall back to direct transformation (no caching)
    # This path is used for NONE mode or when caching is disabled
    results = []
    transform_logs = []

    iterator = conversations_with_uids
    if show_progress and track:
        iterator = track(conversations_with_uids, description="Applying transformations...")

    for conv_id, messages in iterator:
        # Re-wrap in ConversationWithUID for apply_transform_with_tracking
        max_uid = max((m.get("_uid", 0) for m in messages), default=0)
        conv = ConversationWithUID(
            conversation_id=conv_id,
            messages=messages,
            uid_counter=max_uid + 1,
            transform_log=[],
        )

        # Apply transformations
        transformed = apply_transform_with_tracking(conv, config, context_model)

        results.append((conv_id, transformed.messages))
        transform_logs.append((conv_id, transformed.transform_log))

    return results, transform_logs
