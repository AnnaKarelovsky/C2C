"""Section-aware tokenization for gpt-oss-20b model family.

This module provides precise section detection within tokenized messages,
identifying system, user, tool, and assistant sections with sub-types
(reasoning, tool_call, text).

Handles two gpt-oss template quirks:
  1. tojson double-escaping: tool content and tool-call args are JSON strings,
     but tojson treats them as plain strings and re-escapes. Fix: parse to
     dicts before rendering.
  2. Tool-call arg formatting: tojson emits compact JSON, but the model may
     have generated pretty-printed JSON. Fix: post-render replacement.

Known differences vs Fireworks server-side tokenization (system/developer
preamble only — user/assistant/tool sections match exactly):
  - Developer "# Instructions" header: Fireworks renders it even when empty;
    HF skips it. ~5 tokens difference.
  - Tool description "//" comment continuations: minor formatting difference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from rosetta.workflow.analysis.interface import TokenSection, TokenizedConversation
from rosetta.workflow.camel_utils import read_jsonl

# =============================================================================
# Token Marker IDs for gpt-oss-20b
# =============================================================================

OSS_MARKERS = {
    "START": 200006,    # <|start|> - Section start
    "END": 200007,      # <|end|> - Section end (normal)
    "CALL": 200012,     # <|call|> - Section end (tool call)
    "RETURN": 200002,   # <|return|> - Section end (final)
    "CHANNEL": 200005,  # <|channel|> - Channel type marker
    "MESSAGE": 200008,  # <|message|> - Content start
}


def _is_oss_tokenizer(tokenizer) -> bool:
    """Check if tokenizer is from gpt-oss model family."""
    name = getattr(tokenizer, "name_or_path", "")
    return "gpt-oss" in name.lower() or "openai/gpt-oss" in name.lower()


def _parse_section_header(
    tokens: List[int],
    start_idx: int,
    message_idx: int,
    end_marker_id: int,
    tokenizer,
) -> tuple[str, str]:
    """Parse role and content_type from section header tokens.

    Args:
        tokens: Full token list
        start_idx: Index of <|start|> token
        message_idx: Index of <|message|> token
        end_marker_id: Token ID of the section end marker
        tokenizer: Tokenizer for decoding

    Returns:
        (role, content_type) tuple
    """
    # Decode header text (between <|start|> and <|message|>)
    header_tokens = tokens[start_idx + 1 : message_idx]
    header_text = tokenizer.decode(header_tokens, skip_special_tokens=False)

    # Default values
    role = "unknown"
    content_type = "text"

    # Parse role from header
    header_lower = header_text.lower()

    # Tool response starts with "functions." (e.g., "functions.search to=assistant...")
    # Note: Tool CALL starts with "assistant to=functions..." which also contains "functions."
    # So we must check if it STARTS with "functions." to detect tool response
    if header_text.startswith("functions."):
        role = "tool"
        # Extract tool name: "functions.search to=..." -> "search"
        # Strip <|channel|> first since it's not whitespace and may appear
        # directly after the tool name (e.g., "get_document<|channel|>commentary")
        func_part = header_text[len("functions."):].split("<|channel|>")[0]
        tool_name = func_part.split()[0] if func_part else "tool_response"
        content_type = tool_name

    elif header_lower.startswith("assistant") or "assistant" in header_lower:
        role = "assistant"

        # Determine content_type from channel
        # Check for tool call first: header contains "to=functions.NAME"
        # (model may use either "commentary" or "analysis" channel for tool calls)
        to_idx = header_text.find("to=functions.")
        if to_idx >= 0 and end_marker_id == OSS_MARKERS["CALL"]:
            func_name = header_text[to_idx + len("to=functions."):].split()[0].split("<")[0]
            content_type = f"tool_call:{func_name}" if func_name else "tool_call"
        elif "analysis" in header_lower:
            content_type = "reasoning"
        elif "final" in header_lower:
            content_type = "text"
        else:
            content_type = "text"

    elif header_lower.startswith("system"):
        role = "system"
        content_type = "text"

    elif header_lower.startswith("developer"):
        # Developer section contains tool definitions
        role = "developer"
        content_type = "tool_definitions"

    elif header_lower.startswith("user"):
        role = "user"
        content_type = "text"

    return role, content_type


def _try_parse_json(s: Any) -> Any:
    """Parse a JSON string to dict/list if possible, else return as-is."""
    if isinstance(s, str) and s.strip()[:1] in ("{", "["):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
    return s


def preprocess_messages(
    messages: List[Dict[str, Any]],
    convert_reasoning: bool = True,
    fix_tool_escaping: bool = True,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Prepare messages for gpt-oss apply_chat_template.

    Applies three fixes:
      1. reasoning_content → thinking (gpt-oss template field name)
      2. Tool content: parse JSON strings → dicts (avoid tojson double-escape)
      3. Tool-call args: parse JSON strings → dicts, collect originals for
         post-render fixup via ``_fixup_tool_call_args``.

    Returns:
        (processed_messages, original_arg_strings)
    """
    processed = []
    original_args: List[str] = []

    for msg in messages:
        m = dict(msg)

        if convert_reasoning and msg.get("reasoning_content"):
            m["thinking"] = msg["reasoning_content"]

        if fix_tool_escaping and msg.get("role") == "tool" and "content" in msg:
            m["content"] = _try_parse_json(msg["content"])

        if fix_tool_escaping and msg.get("tool_calls"):
            new_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                func = dict(tc.get("function", {}))
                raw_args = func.get("arguments", "")
                if isinstance(raw_args, str):
                    original_args.append(raw_args)
                    parsed = _try_parse_json(raw_args)
                    if parsed is not raw_args:
                        func["arguments"] = parsed
                tc["function"] = func
                new_calls.append(tc)
            m["tool_calls"] = new_calls

        processed.append(m)

    return processed, original_args


def _fixup_tool_call_args(rendered: str, original_args: List[str]) -> str:
    """Replace tojson compact args with the original model-generated format.

    The template's ``tojson`` filter always produces compact JSON, but the
    model may have generated pretty-printed JSON.  This replaces each
    occurrence so the tokenization matches what the API actually saw.
    """
    idx = 0
    for raw_arg in original_args:
        try:
            compact = json.dumps(
                json.loads(raw_arg), ensure_ascii=False, separators=(", ", ": ")
            )
        except (json.JSONDecodeError, TypeError):
            continue
        target = f"<|message|>{compact}<|call|>"
        replacement = f"<|message|>{raw_arg}<|call|>"
        pos = rendered.find(target, idx)
        if pos >= 0:
            rendered = rendered[:pos] + replacement + rendered[pos + len(target):]
            idx = pos + len(replacement)
    return rendered


def _render_and_encode(
    tokenizer,
    processed_messages: List[Dict[str, Any]],
    original_args: List[str],
    tools: Optional[List[Dict[str, Any]]] = None,
    model_identity: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> List[int]:
    """Render messages via apply_chat_template and encode to token IDs."""
    kwargs: Dict[str, Any] = {}
    if model_identity:
        kwargs["model_identity"] = model_identity
    if tools:
        kwargs["tools"] = tools

    rendered = tokenizer.apply_chat_template(
        processed_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    rendered = _fixup_tool_call_args(rendered, original_args)
    return tokenizer.encode(rendered, add_special_tokens=False)


def tokenize_conversation_oss(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    exclude_final: bool = False,
    conversation_id: Optional[str] = None,
    convert_reasoning: bool = True,
    fix_tool_escaping: bool = True,
    model_identity: Optional[str] = None,
) -> TokenizedConversation:
    """Tokenize conversation with precise section detection for gpt-oss models.

    Args:
        messages: List of message dicts with "role" and "content".
        tokenizer: gpt-oss tokenizer (openai/gpt-oss-*).
        tools: Optional list of tool schemas (OpenAI format).
        exclude_final: If True, exclude final message to include all reasoning.
        conversation_id: Optional identifier for the conversation.
        convert_reasoning: If True, convert reasoning_content to thinking field.
        fix_tool_escaping: If True, fix JSON double-escaping in tool messages.
        model_identity: System prompt text passed to apply_chat_template.

    Returns:
        TokenizedConversation with precise section boundaries.

    Raises:
        NotImplementedError: If tokenizer is not from gpt-oss family.
    """
    if not _is_oss_tokenizer(tokenizer):
        raise NotImplementedError(
            f"tokenize_conversation_oss only supports gpt-oss models. "
            f"Got: {getattr(tokenizer, 'name_or_path', 'unknown')}"
        )

    processed_messages, original_args = preprocess_messages(
        messages, convert_reasoning=convert_reasoning,
        fix_tool_escaping=fix_tool_escaping,
    )

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

    tokens = _render_and_encode(
        tokenizer, processed_messages, original_args,
        tools=tool_schemas, model_identity=model_identity,
    )
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Find sections by scanning for markers, passing messages for UID info
    sections = _find_sections(tokens, tokenizer, processed_messages)

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=processed_messages,
        conversation_id=conversation_id,
    )


def _find_sections(
    tokens: List[int],
    tokenizer,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[TokenSection]:
    """Find all sections in tokenized conversation.

    Scans for <|start|> markers and parses section boundaries.
    Correctly maps sections to messages, handling cases where one message
    produces multiple sections (e.g., assistant with reasoning + text + tool_call).

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer for decoding headers
        messages: Optional list of messages with UID info (_uid, _original_uid)

    Returns:
        List of TokenSection objects
    """
    START = OSS_MARKERS["START"]
    END = OSS_MARKERS["END"]
    CALL = OSS_MARKERS["CALL"]
    RETURN = OSS_MARKERS["RETURN"]
    MESSAGE = OSS_MARKERS["MESSAGE"]
    END_MARKERS = {END, CALL, RETURN}

    # Build a list of message info for UID lookup
    # Each entry: (role, uid, original_uid, transform_type)
    message_info: List[Dict[str, Any]] = []
    if messages:
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            uid = msg.get("_uid", idx)  # Default to index if no UID
            original_uid = msg.get("_original_uid", uid)
            transform_type = "original" if uid == original_uid else "summarized"
            message_info.append({
                "role": role,
                "uid": uid,
                "original_uid": original_uid,
                "transform_type": transform_type,
            })

    sections = []
    section_idx_counter = 0
    message_idx = 0  # Index into message_info
    last_role = None  # Track role transitions
    assistant_section_count = 0  # Track sections within an assistant message
    i = 0

    while i < len(tokens):
        # Find next <|start|>
        if tokens[i] != START:
            i += 1
            continue

        start_idx = i

        # Find <|message|> after <|start|>
        message_pos = None
        for j in range(start_idx + 1, min(start_idx + 50, len(tokens))):
            if tokens[j] == MESSAGE:
                message_pos = j
                break

        if message_pos is None:
            i += 1
            continue

        # Find end marker after <|message|>
        end_pos = None
        end_marker_id = None
        for k in range(message_pos + 1, len(tokens)):
            if tokens[k] in END_MARKERS:
                end_pos = k
                end_marker_id = tokens[k]
                break

        if end_pos is None:
            # No end marker found, section extends to end
            end_pos = len(tokens)
            end_marker_id = END

        # Parse header to get role and content_type
        role, content_type = _parse_section_header(
            tokens, start_idx, message_pos, end_marker_id, tokenizer
        )

        # Section boundaries: content is between <|message|> and end marker,
        # but tool_call sections include the full header (from <|start|>)
        # so that function selection tokens are captured.
        if content_type.startswith("tool_call"):
            content_start = start_idx
            content_end = end_pos + 1  # include <|call|> end marker
        else:
            content_start = message_pos + 1
            content_end = end_pos

        # Determine which message this section belongs to
        # Logic:
        # - "developer" sections are for tool definitions, skip message advancement
        # - Multiple assistant sections (reasoning, text, tool_call) belong to same message
        # - Each tool section belongs to a separate tool message
        # - system/user each have their own message

        current_message_idx = message_idx

        if role == "developer":
            # Developer section is for tool definitions, not a user message
            current_message_idx = -1  # No message association
        elif role == "assistant":
            if last_role != "assistant":
                # First assistant section after non-assistant, new message
                if last_role is not None and last_role != "developer":
                    message_idx += 1
                current_message_idx = message_idx
                assistant_section_count = 1
            else:
                # Continuing assistant sections, same message
                current_message_idx = message_idx
                assistant_section_count += 1
        elif role == "tool":
            # Each tool section is a separate message
            if last_role is not None and last_role != "developer":
                message_idx += 1
            current_message_idx = message_idx
        else:
            # system, user - each is separate message
            if last_role is not None and last_role != "developer":
                message_idx += 1
            current_message_idx = message_idx

        # Get UID info for this message
        if current_message_idx >= 0 and current_message_idx < len(message_info):
            info = message_info[current_message_idx]
            msg_uid = info.get("uid", -1)
            orig_uid = info.get("original_uid", msg_uid)
            transform = info.get("transform_type", "original")
        else:
            msg_uid = -1
            orig_uid = -1
            transform = "original"

        # Create section if it has content
        if content_end > content_start:
            section = TokenSection(
                start_idx=content_start,
                end_idx=content_end,
                role=role,
                message_idx=current_message_idx,
                content_type=content_type,
                message_uid=msg_uid,
                original_uid=orig_uid,
                transform_type=transform,
            )
            sections.append(section)
            section_idx_counter += 1

        # Update last_role (skip developer for transition logic)
        if role != "developer":
            last_role = role

        # Move past end marker
        i = end_pos + 1

    return sections


def tokenize_conversation_with_sections(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> TokenizedConversation:
    """Auto-detect model family and tokenize with section detection.

    This is the main dispatcher function that selects the appropriate
    tokenizer based on the model family.

    Args:
        messages: List of message dicts.
        tokenizer: HuggingFace tokenizer.
        tools: Optional list of tool schemas.
        **kwargs: Additional arguments passed to model-specific tokenizer.

    Returns:
        TokenizedConversation with precise section boundaries.

    Raises:
        NotImplementedError: If model family is not supported.

    Supported models:
        - gpt-oss-* (openai/gpt-oss-20b, etc.)
        - Kimi-K2-* (moonshotai/Kimi-K2-Thinking, etc.)
    """
    from .kimi_tokenizer import _is_kimi_tokenizer, tokenize_conversation_kimi

    model_name = getattr(tokenizer, "name_or_path", "")

    if _is_oss_tokenizer(tokenizer):
        return tokenize_conversation_oss(messages, tokenizer, tools, **kwargs)
    elif _is_kimi_tokenizer(tokenizer):
        # KIMI doesn't need convert_reasoning - it uses reasoning_content directly
        kimi_kwargs = {k: v for k, v in kwargs.items() if k != "convert_reasoning"}
        return tokenize_conversation_kimi(messages, tokenizer, tools, **kimi_kwargs)
    else:
        raise NotImplementedError(
            f"Section-aware tokenization not supported for model: {model_name}. "
            f"Supported model families: gpt-oss-*, Kimi-K2-*"
        )


def batch_tokenize_with_sections(
    conversations: List[tuple[str, List[Dict[str, Any]]]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_length: Optional[int] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[TokenizedConversation]:
    """Batch tokenize conversations with section detection.

    Args:
        conversations: List of (id, messages) tuples.
        tokenizer: HuggingFace tokenizer.
        tools: Optional list of tool schemas.
        max_length: Optional maximum sequence length (skip longer).
        show_progress: Whether to show progress bar.
        **kwargs: Additional arguments passed to tokenizer.

    Returns:
        List of TokenizedConversation objects with section boundaries.
    """
    results = []

    iterator = conversations
    if show_progress:
        try:
            from rich.progress import track

            iterator = track(conversations, description="Tokenizing with sections...")
        except ImportError:
            pass

    for conv_id, messages in iterator:
        try:
            conv = tokenize_conversation_with_sections(
                messages,
                tokenizer,
                tools=tools,
                conversation_id=conv_id,
                **kwargs,
            )
            if max_length is not None and conv.seq_len > max_length:
                continue
            results.append(conv)
        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {conv_id}: {e}")
            continue

    return results


# =============================================================================
# Per-interaction trajectory tokenization
# =============================================================================

def tokenize_trajectory_interaction(
    tokenizer,
    messages: List[Dict[str, Any]],
    asst_idx: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    model_identity: Optional[str] = None,
) -> tuple[List[int], int]:
    """Tokenize one interaction from a trajectory.

    Slices messages per-interaction so intermediate reasoning is preserved
    (no future final message to trigger the template's lookahead that drops
    thinking from earlier assistant turns).

    Args:
        tokenizer:      HF tokenizer (e.g. openai/gpt-oss-20b).
        messages:       Full message list from trajectory (including system).
        asst_idx:       Index of the assistant message for this interaction.
        tools:          Tool schemas (OpenAI format).
        model_identity: System prompt text, passed to apply_chat_template.

    Returns:
        (input_ids, prompt_len) — full token sequence (prompt + response)
        and the index where the response starts. Use as::

            prompt_ids = input_ids[:prompt_len]
            response_ids = input_ids[prompt_len:]
    """
    # System message (index 0) is handled via model_identity kwarg
    prompt_msgs = messages[1:asst_idx]
    full_msgs = messages[1:asst_idx + 1]

    prompt_processed, prompt_args = preprocess_messages(prompt_msgs)
    full_processed, full_args = preprocess_messages(full_msgs)

    prompt_ids = _render_and_encode(
        tokenizer, prompt_processed, prompt_args,
        tools=tools, model_identity=model_identity,
        add_generation_prompt=True,
    )
    full_ids = _render_and_encode(
        tokenizer, full_processed, full_args,
        tools=tools, model_identity=model_identity,
    )
    return full_ids, len(prompt_ids)


@dataclass
class TokenizedTrajectory:
    """Tokenized conversation with section boundaries.

    Attributes:
        example_id: Trajectory identifier.
        input_ids: Flat token ID sequence for the conversation.
        section_indices: Section boundaries with role and content_type.
            Each :class:`TokenSection` has ``start_idx``, ``end_idx``,
            ``role`` (system/user/assistant/tool/developer), and
            ``content_type`` (text/reasoning/tool_call/tool_definitions).

    Example::

        t = results[0]
        t.input_ids                              # List[int]
        t.section_indices                        # List[TokenSection]
        t.input_ids[s.start_idx:s.end_idx]       # tokens for section s
        [s for s in t.section_indices if s.content_type == "reasoning"]
    """
    example_id: str
    input_ids: List[int] = field(default_factory=list)
    section_indices: List[TokenSection] = field(default_factory=list)


def _tokenize_final_assistant(
    tokenizer,
    final_msg: Dict[str, Any],
    model_identity: Optional[str] = None,
) -> tuple[List[int], List[TokenSection]]:
    """Tokenize a final assistant message in isolation.

    Renders the message with a dummy user prefix so the assistant turn
    tokens are context-independent.  Returns (token_ids, sections).
    """
    dummy = [{"role": "user", "content": "."}]
    processed_dummy, args_dummy = preprocess_messages(dummy)
    processed_full, args_full = preprocess_messages(dummy + [final_msg])

    prefix = _render_and_encode(
        tokenizer, processed_dummy, args_dummy, model_identity=model_identity,
    )
    full = _render_and_encode(
        tokenizer, processed_full, args_full, model_identity=model_identity,
    )

    final_ids = full[len(prefix):]
    final_sections = _find_sections(final_ids, tokenizer)
    return final_ids, final_sections


def load_and_tokenize_with_reasoning(
    trajectory_path: str,
    tokenizer: AutoTokenizer,
) -> List[TokenizedTrajectory]:
    """Load trajectories and tokenize with section detection.

    Preserves intermediate reasoning by tokenizing with
    ``exclude_final=True``, then separately tokenizing the final
    assistant message and appending it.  This avoids the template's
    lookahead that drops intermediate thinking.

    Args:
        trajectory_path: Path to ``*_trajectories.jsonl``.
        tokenizer:       HuggingFace tokenizer.

    Returns:
        List of TokenizedTrajectory, one per example.
    """
    trajectories = read_jsonl(Path(trajectory_path))

    results = []
    for traj in trajectories:
        messages = traj["messages"]
        tools = traj.get("tools")
        model_identity = traj.get("model_identity")
        if model_identity is None and messages and messages[0].get("role") == "system":
            model_identity = messages[0]["content"]

        # Tokenize with exclude_final=True to preserve intermediate reasoning
        conv = tokenize_conversation_oss(
            messages[1:], tokenizer, tools=tools,
            exclude_final=True,
            conversation_id=str(traj.get("example_id", "")),
            model_identity=model_identity,
        )
        result_ids = conv.input_ids.tolist()
        result_sections = list(conv.sections)

        # Append the final assistant message (tokenized in isolation)
        final_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant" and not messages[i].get("tool_calls"):
                final_msg = messages[i]
                break

        if final_msg is not None:
            final_ids, final_sections = _tokenize_final_assistant(
                tokenizer, final_msg, model_identity,
            )
            offset = len(result_ids)
            for s in final_sections:
                s.start_idx += offset
                s.end_idx += offset
            result_ids.extend(final_ids)
            result_sections.extend(final_sections)

        results.append(TokenizedTrajectory(
            example_id=str(traj["example_id"]),
            input_ids=result_ids,
            section_indices=result_sections,
        ))
    return results
