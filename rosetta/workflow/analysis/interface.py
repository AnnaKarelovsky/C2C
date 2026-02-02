"""Interface for loading evaluation results and tokenizing with section tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class TokenSection:
    """Tracks a contiguous section of tokens from one message.

    Attributes:
        start_idx: Start position in token sequence (inclusive).
        end_idx: End position in token sequence (exclusive).
        role: Message role ("system", "user", "assistant", "tool").
        message_idx: Index of source message in the conversation.
        content_type: Type of content ("text", "tool_call", "reasoning").
    """

    start_idx: int
    end_idx: int
    role: str
    message_idx: int
    content_type: str = "text"

    @property
    def length(self) -> int:
        """Number of tokens in this section."""
        return self.end_idx - self.start_idx

    def __repr__(self) -> str:
        return (
            f"TokenSection({self.role}:{self.content_type}, "
            f"[{self.start_idx}:{self.end_idx}], msg={self.message_idx})"
        )


@dataclass
class TokenizedConversation:
    """A conversation converted to tokens with section tracking.

    Attributes:
        input_ids: Token IDs as tensor [seq_len].
        sections: List of TokenSection metadata.
        messages: Original messages (for reference).
        conversation_id: Optional identifier for the conversation.
    """

    input_ids: torch.Tensor
    sections: List[TokenSection]
    messages: List[Dict[str, Any]]
    conversation_id: Optional[str] = None

    @property
    def seq_len(self) -> int:
        """Total sequence length."""
        return len(self.input_ids)

    def get_section_mask(self, role: Optional[str] = None, content_type: Optional[str] = None) -> torch.Tensor:
        """Create a boolean mask for tokens matching the filter criteria.

        Args:
            role: Filter by role (e.g., "assistant", "tool").
            content_type: Filter by content type (e.g., "text", "reasoning").

        Returns:
            Boolean tensor of shape [seq_len].
        """
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        for section in self.sections:
            if role is not None and section.role != role:
                continue
            if content_type is not None and section.content_type != content_type:
                continue
            mask[section.start_idx : section.end_idx] = True
        return mask

    def get_role_labels(self) -> torch.Tensor:
        """Create a tensor with role labels for each token.

        Returns:
            Integer tensor [seq_len] with role indices:
            0=system, 1=user, 2=assistant, 3=tool, -1=unknown
        """
        role_map = {"system": 0, "user": 1, "assistant": 2, "tool": 3}
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for section in self.sections:
            labels[section.start_idx : section.end_idx] = role_map.get(section.role, -1)
        return labels


@dataclass
class TransformResult:
    """Result of applying a context transformation.

    Attributes:
        original: The original tokenized conversation.
        transformed: The transformed tokenized conversation.
        transform_name: Name of the transformation applied.
        token_mapping: Optional mapping from transformed token indices to original.
    """

    original: TokenizedConversation
    transformed: TokenizedConversation
    transform_name: str
    token_mapping: Optional[Dict[int, int]] = None


# =============================================================================
# Data Loading
# =============================================================================


def load_evaluation_results(path: Path | str) -> List[Dict[str, Any]]:
    """Load evaluation records from a JSONL file.

    Args:
        path: Path to the JSONL file (e.g., results.jsonl).

    Returns:
        List of record dictionaries.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
    return records


def extract_conversations(
    records: List[Dict[str, Any]],
    message_field: str = "llm0_messages",
    id_field: str = "example_id",
    filter_errors: bool = True,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Extract conversations from evaluation records.

    Args:
        records: List of evaluation record dicts.
        message_field: Field name containing the message list.
        id_field: Field name for the conversation ID.
        filter_errors: If True, skip records with non-null error field.

    Returns:
        List of (conversation_id, messages) tuples.
    """
    conversations = []
    for rec in records:
        if filter_errors and rec.get("error"):
            continue
        messages = rec.get(message_field)
        if not messages or not isinstance(messages, list):
            continue
        conv_id = str(rec.get(id_field, rec.get("idx", len(conversations))))
        conversations.append((conv_id, messages))
    return conversations


# =============================================================================
# Tokenization with Section Tracking
# =============================================================================


def _get_message_parts(msg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract content parts from a message.

    Returns a list of (content_type, text) tuples for the message.
    """
    parts = []
    role = msg.get("role", "unknown")

    # Handle reasoning content (assistant only)
    if role == "assistant":
        reasoning = msg.get("reasoning_content") or msg.get("_reasoning")
        if reasoning:
            parts.append(("reasoning", str(reasoning)))

    # Main content
    content = msg.get("content", "")
    if content:
        parts.append(("text", str(content)))

    # Tool calls (assistant only)
    if role == "assistant":
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc_text = json.dumps(tool_calls, ensure_ascii=False)
            parts.append(("tool_call", tc_text))

    # If no parts extracted, add empty text
    if not parts:
        parts.append(("text", ""))

    return parts


def tokenize_conversation(
    messages: List[Dict[str, Any]],
    tokenizer,
    add_generation_prompt: bool = False,
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> TokenizedConversation:
    """Convert a conversation to tokens with section tracking.

    This tokenizes each message individually to track section boundaries,
    then concatenates them to form the full sequence.

    Args:
        messages: List of message dicts with "role" and "content".
        tokenizer: HuggingFace tokenizer with chat template support.
        add_generation_prompt: If True, add generation prompt at the end.
        conversation_id: Optional identifier for the conversation.
        tools: Optional list of tool schemas (OpenAI format). Can be:
            - List of dicts with OpenAI tool schema format
            - List of FunctionTool objects (will call get_openai_tool_schema())

    Returns:
        TokenizedConversation with input_ids and section metadata.
    """
    # Convert FunctionTool objects to schemas if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, 'get_openai_tool_schema'):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
    sections = []
    all_input_ids = []
    current_pos = 0

    # Try to use chat template for full conversation first to get proper formatting
    # But we need per-message boundaries, so we tokenize incrementally
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        # Build a minimal conversation up to this message to get proper formatting
        # This handles chat templates that add special tokens between messages
        prefix_msgs = messages[:msg_idx]
        current_msgs = messages[: msg_idx + 1]

        # Tokenize prefix (all messages before current)
        if prefix_msgs:
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=False,
                tools=tool_schemas,
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        else:
            prefix_ids = []

        # Tokenize up to and including current message
        current_text = tokenizer.apply_chat_template(
            current_msgs, tokenize=False, add_generation_prompt=False,
            tools=tool_schemas,
        )
        current_ids = tokenizer.encode(current_text, add_special_tokens=False)

        # The tokens for this message are the difference
        msg_start = len(prefix_ids)
        msg_end = len(current_ids)
        msg_ids = current_ids[msg_start:]

        if msg_ids:
            # Create section for this message
            # For now, treat entire message as single section
            # Future: split by content parts (reasoning, text, tool_call)
            section = TokenSection(
                start_idx=current_pos,
                end_idx=current_pos + len(msg_ids),
                role=role,
                message_idx=msg_idx,
                content_type="text",  # TODO: split by parts
            )
            sections.append(section)
            all_input_ids.extend(msg_ids)
            current_pos += len(msg_ids)

    # Add generation prompt if requested
    if add_generation_prompt and messages:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tool_schemas,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > current_pos:
            # There's a generation prompt suffix
            gen_ids = full_ids[current_pos:]
            section = TokenSection(
                start_idx=current_pos,
                end_idx=current_pos + len(gen_ids),
                role="assistant",
                message_idx=len(messages),  # Virtual message index
                content_type="generation_prompt",
            )
            sections.append(section)
            all_input_ids.extend(gen_ids)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=messages,
        conversation_id=conversation_id,
    )


def tokenize_conversation_simple(
    messages: List[Dict[str, Any]],
    tokenizer,
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> TokenizedConversation:
    """Simplified tokenization that tokenizes the full conversation at once.

    This is faster but may have less accurate section boundaries for some
    tokenizers with complex chat templates.

    Args:
        messages: List of message dicts.
        tokenizer: HuggingFace tokenizer.
        conversation_id: Optional identifier.
        tools: Optional list of tool schemas (OpenAI format) or FunctionTool objects.

    Returns:
        TokenizedConversation with approximate section boundaries.
    """
    # Convert FunctionTool objects to schemas if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, 'get_openai_tool_schema'):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

    # Tokenize full conversation
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tool_schemas
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)

    # Estimate section boundaries by tokenizing each message's content
    sections = []
    current_pos = 0

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Tokenize just the content to estimate its length
        if content:
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            content_len = len(content_ids)
        else:
            content_len = 0

        # Account for role tokens and template formatting (approximate)
        # This is a rough estimate; exact boundaries depend on the chat template
        estimated_overhead = 10  # Typical overhead for role markers, etc.

        section_len = content_len + estimated_overhead
        section_end = min(current_pos + section_len, len(input_ids))

        if section_end > current_pos:
            section = TokenSection(
                start_idx=current_pos,
                end_idx=section_end,
                role=role,
                message_idx=msg_idx,
                content_type="text",
            )
            sections.append(section)
            current_pos = section_end

    # Adjust last section to cover remaining tokens
    if sections and current_pos < len(input_ids):
        sections[-1] = TokenSection(
            start_idx=sections[-1].start_idx,
            end_idx=len(input_ids),
            role=sections[-1].role,
            message_idx=sections[-1].message_idx,
            content_type=sections[-1].content_type,
        )

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=messages,
        conversation_id=conversation_id,
    )


# =============================================================================
# Context Transformations
# =============================================================================


def apply_context_transform(
    conversation: TokenizedConversation,
    transform_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    tokenizer,
    transform_name: str = "transform",
) -> TransformResult:
    """Apply a transformation to the conversation and re-tokenize.

    Args:
        conversation: Original tokenized conversation.
        transform_fn: Function that transforms the message list.
        tokenizer: Tokenizer for re-tokenizing after transformation.
        transform_name: Name of the transformation for logging.

    Returns:
        TransformResult with original and transformed conversations.
    """
    # Apply transformation to messages
    transformed_messages = transform_fn(conversation.messages)

    # Re-tokenize
    transformed_conv = tokenize_conversation(
        transformed_messages,
        tokenizer,
        conversation_id=f"{conversation.conversation_id}__{transform_name}",
    )

    return TransformResult(
        original=conversation,
        transformed=transformed_conv,
        transform_name=transform_name,
        token_mapping=None,  # Token alignment is complex after transforms
    )


def create_summarize_transform(
    model,
    summarize_tool: bool = True,
    summarize_assistant: bool = False,
    summarize_reasoning: bool = False,
):
    """Create a transformation function that summarizes content.

    Args:
        model: CAMEL model backend for summarization.
        summarize_tool: Whether to summarize tool responses.
        summarize_assistant: Whether to summarize assistant responses.
        summarize_reasoning: Whether to summarize reasoning content.

    Returns:
        A callable that transforms a message list.
    """
    from rosetta.workflow.contextManage import (
        inject_call_context,
        summarize_content,
        summarize_reasoning as summarize_reasoning_fn,
        summarize_tool_resp,
    )

    def transform(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Make a copy to avoid modifying original
        messages = [dict(m) for m in messages]

        # Inject call context for tool messages
        inject_call_context(messages)

        result = []
        for msg in messages:
            role = msg.get("role", "")

            if role == "tool" and summarize_tool:
                summarized = summarize_tool_resp([msg], model)
                result.append(summarized[0])
            elif role == "assistant":
                new_msg = dict(msg)
                if summarize_assistant and msg.get("content"):
                    summarized = summarize_content([msg], model)
                    new_msg["content"] = summarized[0].get("content", "")
                if summarize_reasoning and (msg.get("_reasoning") or msg.get("reasoning_content")):
                    reasoning_msg = {"content": msg.get("_reasoning") or msg.get("reasoning_content")}
                    summarized = summarize_reasoning_fn([reasoning_msg], model)
                    if "_reasoning" in msg:
                        new_msg["_reasoning"] = summarized[0].get("content", "")
                    if "reasoning_content" in msg:
                        new_msg["reasoning_content"] = summarized[0].get("content", "")
                result.append(new_msg)
            else:
                result.append(msg)

        return result

    return transform


# =============================================================================
# Batch Processing Utilities
# =============================================================================


def batch_tokenize_conversations(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    tokenizer,
    max_length: Optional[int] = None,
    show_progress: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[TokenizedConversation]:
    """Tokenize multiple conversations.

    Args:
        conversations: List of (id, messages) tuples.
        tokenizer: HuggingFace tokenizer.
        max_length: Optional maximum sequence length (skip longer).
        show_progress: Whether to show progress bar.
        tools: Optional list of tool schemas (OpenAI format) or FunctionTool objects.
            Applied to all conversations in the batch.

    Returns:
        List of TokenizedConversation objects.
    """
    results = []

    iterator = conversations
    if show_progress:
        try:
            from rich.progress import track
            iterator = track(conversations, description="Tokenizing...")
        except ImportError:
            pass

    for conv_id, messages in iterator:
        try:
            conv = tokenize_conversation(
                messages, tokenizer, conversation_id=conv_id, tools=tools
            )
            if max_length is not None and conv.seq_len > max_length:
                continue
            results.append(conv)
        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {conv_id}: {e}")
            continue

    return results
