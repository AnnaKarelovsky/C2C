"""Section-aware tokenization for Kimi-K2-Thinking model family.

This module provides precise section detection within tokenized messages,
identifying system, user, tool, and assistant sections with sub-types
(reasoning, tool_call, text).

The KIMI chat template has similar lookahead behavior to GPT-OSS:
- It finds the last non-tool-call assistant message
- Messages before (and including) this are "history" - reasoning is dropped
- Messages after are "suffix" - reasoning is preserved

For perplexity analysis, use exclude_final=True to include all reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .interface import TokenSection, TokenizedConversation

# =============================================================================
# Token Marker IDs for Kimi-K2-Thinking
# =============================================================================

KIMI_MARKERS = {
    "IM_END": 163586,           # <|im_end|>
    "IM_USER": 163587,          # <|im_user|>
    "IM_ASSISTANT": 163588,     # <|im_assistant|>
    "IM_SYSTEM": 163594,        # <|im_system|>
    "IM_MIDDLE": 163601,        # <|im_middle|>
    "THINK_START": 163606,      # <think>
    "THINK_END": 163607,        # </think>
    "TOOL_CALLS_BEGIN": 163595, # <|tool_calls_section_begin|>
    "TOOL_CALLS_END": 163596,   # <|tool_calls_section_end|>
    "TOOL_CALL_BEGIN": 163597,  # <|tool_call_begin|>
    "TOOL_CALL_ARG": 163598,    # <|tool_call_argument_begin|>
    "TOOL_CALL_END": 163599,    # <|tool_call_end|>
}


def _is_kimi_tokenizer(tokenizer) -> bool:
    """Check if tokenizer is from Kimi model family."""
    name = getattr(tokenizer, "name_or_path", "").lower()
    return "kimi" in name or "moonshot" in name


def tokenize_conversation_kimi(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    exclude_final: bool = False,
    conversation_id: Optional[str] = None,
) -> TokenizedConversation:
    """Tokenize conversation with precise section detection for Kimi models.

    Args:
        messages: List of message dicts with "role" and "content".
        tokenizer: Kimi tokenizer (moonshotai/Kimi-K2-Thinking).
        tools: Optional list of tool schemas (OpenAI format).
        exclude_final: If True, exclude final message to include all reasoning.
        conversation_id: Optional identifier for the conversation.

    Returns:
        TokenizedConversation with precise section boundaries.

    Raises:
        NotImplementedError: If tokenizer is not from Kimi family.
    """
    if not _is_kimi_tokenizer(tokenizer):
        raise NotImplementedError(
            f"tokenize_conversation_kimi only supports Kimi models. "
            f"Got: {getattr(tokenizer, 'name_or_path', 'unknown')}"
        )

    # Preprocess messages - KIMI uses reasoning_content directly
    # Note: The KIMI-K2-Thinking template has lookahead behavior similar to GPT-OSS:
    # - Messages before/including last non-tool-call assistant: reasoning dropped
    # - Messages after: reasoning preserved
    # Use exclude_final=True to preserve all intermediate reasoning.
    processed_messages = [dict(msg) for msg in messages]

    # Optionally exclude final message (for tool-call analysis)
    if exclude_final:
        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                processed_messages = processed_messages[:i]
                break

    # Convert tools if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, "get_openai_tool_schema"):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

    # Render and tokenize
    rendered = tokenizer.apply_chat_template(
        processed_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tool_schemas,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Find sections by scanning for markers
    sections = _find_sections_kimi(tokens, tokenizer, processed_messages)

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=processed_messages,
        conversation_id=conversation_id,
    )


def _find_sections_kimi(
    tokens: List[int],
    tokenizer,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[TokenSection]:
    """Find all sections in tokenized KIMI conversation.

    KIMI format:
    - <|im_system|>name<|im_middle|>content<|im_end|>
    - <|im_user|>name<|im_middle|>content<|im_end|>
    - <|im_assistant|>name<|im_middle|><think>reasoning</think>content<|tool_calls_section_begin|>...<|tool_calls_section_end|><|im_end|>
    - Tool responses: <|im_system|>tool_call_id<|im_middle|>## Return of ...\\ncontent<|im_end|>

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer for decoding headers
        messages: Optional list of messages with UID info (_uid, _original_uid)

    Returns:
        List of TokenSection objects
    """
    IM_END = KIMI_MARKERS["IM_END"]
    IM_USER = KIMI_MARKERS["IM_USER"]
    IM_ASSISTANT = KIMI_MARKERS["IM_ASSISTANT"]
    IM_SYSTEM = KIMI_MARKERS["IM_SYSTEM"]
    IM_MIDDLE = KIMI_MARKERS["IM_MIDDLE"]
    THINK_START = KIMI_MARKERS["THINK_START"]
    THINK_END = KIMI_MARKERS["THINK_END"]
    TOOL_CALLS_BEGIN = KIMI_MARKERS["TOOL_CALLS_BEGIN"]
    TOOL_CALLS_END = KIMI_MARKERS["TOOL_CALLS_END"]

    ROLE_MARKERS = {IM_USER, IM_ASSISTANT, IM_SYSTEM}

    # Build message info for UID lookup
    message_info: List[Dict[str, Any]] = []
    if messages:
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            uid = msg.get("_uid", idx)
            original_uid = msg.get("_original_uid", uid)
            transform_type = "original" if uid == original_uid else "summarized"
            # Extract tool name for tool messages
            tool_name = None
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and idx > 0:
                    for prev_msg in reversed(messages[:idx]):
                        if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                            for tc in prev_msg["tool_calls"]:
                                if tc.get("id") == tool_call_id:
                                    tool_name = tc.get("function", {}).get("name")
                                    break
                            break
            message_info.append({
                "role": role,
                "uid": uid,
                "original_uid": original_uid,
                "transform_type": transform_type,
                "tool_name": tool_name,
            })

    sections = []
    message_idx = -1  # Will be incremented when we see a role marker
    i = 0

    while i < len(tokens):
        # Look for role markers
        if tokens[i] not in ROLE_MARKERS:
            i += 1
            continue

        role_marker = tokens[i]

        # Determine role from marker
        if role_marker == IM_USER:
            current_role = "user"
        elif role_marker == IM_ASSISTANT:
            current_role = "assistant"
        elif role_marker == IM_SYSTEM:
            current_role = "system"  # Could be tool or actual system
        else:
            i += 1
            continue

        # Find <|im_middle|> after role marker
        middle_pos = None
        for j in range(i + 1, min(i + 100, len(tokens))):
            if tokens[j] == IM_MIDDLE:
                middle_pos = j
                break

        if middle_pos is None:
            i += 1
            continue

        # Check header to identify message type
        header_tokens = tokens[i + 1 : middle_pos]
        header_text = ""
        is_tool_response = False
        is_tool_declare = False
        if header_tokens:
            header_text = tokenizer.decode(header_tokens, skip_special_tokens=False)
            # Skip tool_declare section (tool definitions, not a user message)
            if header_text == "tool_declare":
                is_tool_declare = True
            # Tool responses have header name "tool" in KIMI template
            elif current_role == "system" and header_text == "tool":
                is_tool_response = True

        # Find <|im_end|> for this message
        end_pos = None
        for k in range(middle_pos + 1, len(tokens)):
            if tokens[k] == IM_END:
                end_pos = k
                break

        if end_pos is None:
            end_pos = len(tokens)

        # Content is between middle and end
        content_start = middle_pos + 1
        content_end = end_pos

        # Skip tool_declare section (not a user message)
        if is_tool_declare:
            i = end_pos + 1
            continue

        # Advance message index
        message_idx += 1

        # Get UID info for this message
        if 0 <= message_idx < len(message_info):
            info = message_info[message_idx]
            msg_uid = info.get("uid", -1)
            orig_uid = info.get("original_uid", msg_uid)
            transform = info.get("transform_type", "original")
            # Override role detection with actual message role
            if is_tool_response or info.get("role") == "tool":
                current_role = "tool"
                is_tool_response = True
        else:
            msg_uid = -1
            orig_uid = -1
            transform = "original"

        if current_role == "assistant":
            # Parse assistant message for reasoning, text, and tool_calls
            _parse_assistant_sections(
                tokens,
                content_start,
                content_end,
                sections,
                message_idx,
                msg_uid,
                orig_uid,
                transform,
            )
        else:
            # Simple section for user/system/tool
            if content_end > content_start:
                role_for_section = "tool" if is_tool_response else current_role
                if is_tool_response:
                    info = message_info[message_idx] if message_idx < len(message_info) else {}
                    content_type = info.get("tool_name") or "tool_response"
                else:
                    content_type = "text"
                sections.append(
                    TokenSection(
                        start_idx=content_start,
                        end_idx=content_end,
                        role=role_for_section,
                        message_idx=message_idx,
                        content_type=content_type,
                        message_uid=msg_uid,
                        original_uid=orig_uid,
                        transform_type=transform,
                    )
                )

        # Move past this message
        i = end_pos + 1

    return sections


def _parse_assistant_sections(
    tokens: List[int],
    content_start: int,
    content_end: int,
    sections: List[TokenSection],
    message_idx: int,
    msg_uid: int,
    orig_uid: int,
    transform: str,
):
    """Parse assistant message content into reasoning, text, and tool_call sections.

    KIMI assistant format:
    <think>reasoning</think>text content<|tool_calls_section_begin|>...<|tool_calls_section_end|>

    Args:
        tokens: Full token list
        content_start: Start of content (after <|im_middle|>)
        content_end: End of content (before <|im_end|>)
        sections: List to append sections to
        message_idx: Message index
        msg_uid: Message UID
        orig_uid: Original UID
        transform: Transform type
    """
    THINK_START = KIMI_MARKERS["THINK_START"]
    THINK_END = KIMI_MARKERS["THINK_END"]
    TOOL_CALLS_BEGIN = KIMI_MARKERS["TOOL_CALLS_BEGIN"]
    TOOL_CALLS_END = KIMI_MARKERS["TOOL_CALLS_END"]

    current_pos = content_start

    # Look for <think> tag
    if current_pos < content_end and tokens[current_pos] == THINK_START:
        # Find </think>
        think_end_pos = None
        for j in range(current_pos + 1, content_end):
            if tokens[j] == THINK_END:
                think_end_pos = j
                break

        if think_end_pos is not None:
            # Reasoning content is between <think> and </think>
            reasoning_start = current_pos + 1
            reasoning_end = think_end_pos

            if reasoning_end > reasoning_start:
                sections.append(
                    TokenSection(
                        start_idx=reasoning_start,
                        end_idx=reasoning_end,
                        role="assistant",
                        message_idx=message_idx,
                        content_type="reasoning",
                        message_uid=msg_uid,
                        original_uid=orig_uid,
                        transform_type=transform,
                    )
                )

            current_pos = think_end_pos + 1
        else:
            # No closing </think>, treat rest as reasoning
            if content_end > current_pos + 1:
                sections.append(
                    TokenSection(
                        start_idx=current_pos + 1,
                        end_idx=content_end,
                        role="assistant",
                        message_idx=message_idx,
                        content_type="reasoning",
                        message_uid=msg_uid,
                        original_uid=orig_uid,
                        transform_type=transform,
                    )
                )
            return

    # Look for tool calls section
    tool_calls_start = None
    tool_calls_end = None
    for j in range(current_pos, content_end):
        if tokens[j] == TOOL_CALLS_BEGIN:
            tool_calls_start = j
        elif tokens[j] == TOOL_CALLS_END:
            tool_calls_end = j + 1
            break

    # Text content is between current_pos and tool_calls_start (or content_end)
    text_end = tool_calls_start if tool_calls_start else content_end
    if text_end > current_pos:
        sections.append(
            TokenSection(
                start_idx=current_pos,
                end_idx=text_end,
                role="assistant",
                message_idx=message_idx,
                content_type="text",
                message_uid=msg_uid,
                original_uid=orig_uid,
                transform_type=transform,
            )
        )

    # Tool calls section
    if tool_calls_start is not None and tool_calls_end is not None:
        # Content inside tool_calls section
        tc_content_start = tool_calls_start + 1
        tc_content_end = tool_calls_end - 1
        if tc_content_end > tc_content_start:
            sections.append(
                TokenSection(
                    start_idx=tc_content_start,
                    end_idx=tc_content_end,
                    role="assistant",
                    message_idx=message_idx,
                    content_type="tool_call",
                    message_uid=msg_uid,
                    original_uid=orig_uid,
                    transform_type=transform,
                )
            )
