"""UID tracking for message transformations.

This module provides:
- UID assignment to messages in a conversation
- Transform tracking with UID mapping
- Config string parsing for context configurations
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rosetta.workflow.basic_utils import ContentMode, HistoryConfig


@dataclass
class TransformRecord:
    """Records a single message transformation."""

    original_uid: int
    new_uid: int
    transform_type: str  # "original", "summarized", "dropped"
    role: str
    original_char_count: int = 0
    new_char_count: int = 0


@dataclass
class ConversationWithUID:
    """Conversation with UID tracking."""

    conversation_id: str
    messages: List[Dict[str, Any]]
    uid_counter: int  # Next available UID
    transform_log: List[TransformRecord] = field(default_factory=list)

    def get_uid_to_original(self) -> Dict[int, int]:
        """Get mapping from current UIDs to original UIDs."""
        result = {}
        for msg in self.messages:
            uid = msg.get("_uid")
            original = msg.get("_original_uid", uid)
            if uid is not None:
                result[uid] = original
        return result

    def get_transform_type(self, uid: int) -> str:
        """Get the transform type for a given UID."""
        for record in self.transform_log:
            if record.new_uid == uid:
                return record.transform_type
        return "original"


def assign_message_uids(
    messages: List[Dict[str, Any]],
    conversation_id: str = "unknown",
    start_uid: int = 0,
) -> ConversationWithUID:
    """Assign UIDs to messages in a conversation.

    Each message gets:
    - _uid: Current unique identifier
    - _original_uid: Original UID (same as _uid for unmodified messages)

    Args:
        messages: List of message dicts.
        conversation_id: Identifier for the conversation.
        start_uid: Starting UID value.

    Returns:
        ConversationWithUID with messages containing _uid fields.
    """
    result_messages = []
    uid = start_uid

    for msg in messages:
        new_msg = dict(msg)
        new_msg["_uid"] = uid
        new_msg["_original_uid"] = uid
        result_messages.append(new_msg)
        uid += 1

    return ConversationWithUID(
        conversation_id=conversation_id,
        messages=result_messages,
        uid_counter=uid,
        transform_log=[],
    )


def _is_tool_message(msg: Dict) -> bool:
    """Check if message is a tool response."""
    return msg.get("role") == "tool" or "tool_call_id" in msg


def _get_content_length(msg: Dict) -> int:
    """Get character count of message content."""
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "") or msg.get("_reasoning", "")
    return len(content) + len(reasoning)


def apply_transform_with_tracking(
    conv: ConversationWithUID,
    config: HistoryConfig,
    model=None,
    delay_offset: int = 0,
) -> ConversationWithUID:
    """Apply context transformation with UID tracking.

    This function applies transformations based on HistoryConfig and tracks
    all UID changes in the transform_log.

    Args:
        conv: Conversation with UIDs assigned.
        config: HistoryConfig specifying transformations.
        model: CAMEL model backend for summarization (required if SUMMARIZED mode).
        delay_offset: Additional delay offset (for batch processing).

    Returns:
        New ConversationWithUID with transformed messages and updated transform_log.
    """
    from rosetta.workflow.contextManage import (
        _find_round_boundaries,
        inject_call_context,
        summarize_content,
        summarize_reasoning,
        summarize_tool_resp,
    )

    # Deep copy messages to avoid modifying original
    messages = [dict(m) for m in conv.messages]
    transform_log = list(conv.transform_log)
    uid_counter = conv.uid_counter

    # Inject call context for tool messages
    inject_call_context(messages)

    # Find round boundaries
    round_boundaries = _find_round_boundaries(messages)
    if not round_boundaries:
        return ConversationWithUID(
            conversation_id=conv.conversation_id,
            messages=messages,
            uid_counter=uid_counter,
            transform_log=transform_log,
        )

    # Determine which rounds to transform based on delay
    # delay=0 means transform all, delay=1 means skip last round, etc.
    effective_delay = config.delay + delay_offset
    rounds_to_transform = len(round_boundaries) - effective_delay

    # Check if anything to transform
    is_default = (
        config.reasoning == ContentMode.FULL
        and config.assistant == ContentMode.FULL
        and config.tool == ContentMode.FULL
    )

    if is_default or rounds_to_transform <= 0:
        return ConversationWithUID(
            conversation_id=conv.conversation_id,
            messages=messages,
            uid_counter=uid_counter,
            transform_log=transform_log,
        )

    # Process each round that should be transformed
    for round_idx in range(rounds_to_transform):
        start, end = round_boundaries[round_idx]

        for msg_idx in range(start, end):
            msg = messages[msg_idx]
            original_uid = msg["_uid"]
            original_len = _get_content_length(msg)
            role = msg.get("role", "unknown")

            transformed = False
            new_msg = dict(msg)

            # Apply assistant transformations
            if role == "assistant":
                # Handle reasoning
                reasoning = msg.get("_reasoning", "") or msg.get("reasoning_content", "")
                if reasoning:
                    if config.reasoning == ContentMode.NONE:
                        new_msg.pop("reasoning_content", None)
                        new_msg["_reasoning"] = ""  # Clear but keep key
                        transformed = True
                    elif config.reasoning == ContentMode.SUMMARIZED and model:
                        result = summarize_reasoning([{"content": reasoning}], model)
                        summarized = result[0].get("content", "")
                        if "reasoning_content" in msg:
                            new_msg["reasoning_content"] = summarized
                        new_msg["_reasoning"] = summarized
                        transformed = True

                # Handle assistant content
                content = msg.get("content", "")
                if content:
                    if config.assistant == ContentMode.NONE:
                        new_msg["content"] = ""
                        transformed = True
                    elif config.assistant == ContentMode.SUMMARIZED and model:
                        result = summarize_content([msg], model)
                        new_msg["content"] = result[0].get("content", "")
                        transformed = True

            # Apply tool transformations
            elif _is_tool_message(msg):
                content = msg.get("content", "")
                if content:
                    if config.tool == ContentMode.NONE:
                        new_msg["content"] = "[executed]"
                        transformed = True
                    elif config.tool == ContentMode.SUMMARIZED and model:
                        result = summarize_tool_resp([msg], model)
                        new_msg["content"] = result[0].get("content", "")
                        transformed = True

            # If transformed, assign new UID and record
            if transformed:
                new_uid = uid_counter
                uid_counter += 1

                new_msg["_uid"] = new_uid
                new_msg["_original_uid"] = msg.get("_original_uid", original_uid)

                new_len = _get_content_length(new_msg)

                transform_log.append(
                    TransformRecord(
                        original_uid=original_uid,
                        new_uid=new_uid,
                        transform_type="summarized",
                        role=role,
                        original_char_count=original_len,
                        new_char_count=new_len,
                    )
                )

                messages[msg_idx] = new_msg

    return ConversationWithUID(
        conversation_id=conv.conversation_id,
        messages=messages,
        uid_counter=uid_counter,
        transform_log=transform_log,
    )


def parse_context_config(config_str: str) -> HistoryConfig:
    """Parse a context config string into HistoryConfig.

    Format: "{reasoning}_{assistant}_{tool}_{delay}"
    Example: "full_full_sum_0" -> reasoning=FULL, assistant=FULL, tool=SUMMARIZED, delay=0

    Abbreviations:
        full -> ContentMode.FULL
        none -> ContentMode.NONE
        sum  -> ContentMode.SUMMARIZED

    Args:
        config_str: Config string like "full_full_sum_2"

    Returns:
        HistoryConfig object.

    Raises:
        ValueError: If config string is malformed.
    """
    mode_map = {
        "full": ContentMode.FULL,
        "none": ContentMode.NONE,
        "sum": ContentMode.SUMMARIZED,
        "summarized": ContentMode.SUMMARIZED,
    }

    parts = config_str.lower().split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Config string must have 4 parts separated by '_': {config_str}"
        )

    reasoning_str, assistant_str, tool_str, delay_str = parts

    if reasoning_str not in mode_map:
        raise ValueError(f"Invalid reasoning mode: {reasoning_str}")
    if assistant_str not in mode_map:
        raise ValueError(f"Invalid assistant mode: {assistant_str}")
    if tool_str not in mode_map:
        raise ValueError(f"Invalid tool mode: {tool_str}")

    try:
        delay = int(delay_str)
    except ValueError:
        raise ValueError(f"Invalid delay value: {delay_str}")

    return HistoryConfig(
        reasoning=mode_map[reasoning_str],
        assistant=mode_map[assistant_str],
        tool=mode_map[tool_str],
        delay=delay,
    )


def config_to_string(config: HistoryConfig) -> str:
    """Convert HistoryConfig to a string representation.

    Args:
        config: HistoryConfig object.

    Returns:
        String like "full_full_sum_0"
    """
    mode_str = {
        ContentMode.FULL: "full",
        ContentMode.NONE: "none",
        ContentMode.SUMMARIZED: "sum",
    }

    return (
        f"{mode_str[config.reasoning]}_"
        f"{mode_str[config.assistant]}_"
        f"{mode_str[config.tool]}_"
        f"{config.delay}"
    )


def get_message_uid(msg: Dict[str, Any]) -> int:
    """Get the UID of a message, or -1 if not assigned."""
    return msg.get("_uid", -1)


def get_original_uid(msg: Dict[str, Any]) -> int:
    """Get the original UID of a message."""
    return msg.get("_original_uid", msg.get("_uid", -1))


def get_transform_type_from_msg(msg: Dict[str, Any]) -> str:
    """Determine transform type from message fields."""
    uid = msg.get("_uid", -1)
    original = msg.get("_original_uid", uid)
    if uid == original:
        return "original"
    return "summarized"
