"""Interface for loading evaluation results and tokenizing with section tracking.

This module provides:
- Data loading from evaluation JSONL files
- Tokenization with section tracking (role, content_type)
- Unified backend abstraction for computing metrics (local HF or Fireworks API)
- Analysis result structures and aggregation
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Tuple

import torch

from rosetta.utils.core import (
    DEFAULT_UNIFIED_METRICS,
    EntropyLowerBoundUnified,
    EntropyUnified,
    EstimatedEntropyUnified,
    NegLogProbUnified,
    PerplexityUnified,
    PrefillResult,
    TokenMetric,
    Top1NegLogProbUnified,
    TopKEntropyUnified,
    TopKMassUnified,
    UnifiedMetric,
    compute_unified_metrics,
    fireworks_prefill,
    fireworks_to_prefill_result,
    hf_logits_to_prefill_result,
)


@dataclass
class TokenSection:
    """Tracks a contiguous section of tokens from one message.

    Attributes:
        start_idx: Start position in token sequence (inclusive).
        end_idx: End position in token sequence (exclusive).
        role: Message role ("system", "user", "assistant", "tool").
        message_idx: Index of source message in the conversation.
        content_type: Type of content ("text", "tool_call", "reasoning").
        message_uid: UID of the message this section belongs to.
        original_uid: Original UID before any transformation.
        transform_type: Type of transformation applied ("original", "summarized", "dropped").
    """

    start_idx: int
    end_idx: int
    role: str
    message_idx: int
    content_type: str = "text"
    message_uid: int = -1
    original_uid: int = -1
    transform_type: str = "original"

    @property
    def length(self) -> int:
        """Number of tokens in this section."""
        return self.end_idx - self.start_idx

    @property
    def token_count(self) -> int:
        """Alias for length (number of tokens)."""
        return self.end_idx - self.start_idx

    def __repr__(self) -> str:
        return (
            f"TokenSection({self.role}:{self.content_type}, "
            f"[{self.start_idx}:{self.end_idx}], msg={self.message_idx}, "
            f"uid={self.message_uid})"
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


# =============================================================================
# Analysis Results
# =============================================================================


@dataclass
class SectionMetrics:
    """Metrics aggregated for a single section."""

    role: str
    content_type: str
    token_count: int
    start_idx: int
    end_idx: int
    metrics: Dict[str, float]
    message_uid: int = -1
    original_uid: int = -1
    transform_type: str = "original"


@dataclass
class AnalysisResult:
    """Result of analyzing a single conversation."""

    conversation_id: str
    token_count: int
    sections: List[SectionMetrics]
    metrics_by_position: Dict[str, List[float]]
    overall_metrics: Dict[str, float]


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple conversations."""

    num_conversations: int
    total_tokens: int
    by_role: Dict[str, Dict[str, float]]
    by_content_type: Dict[str, Dict[str, float]]
    overall: Dict[str, float]
    by_position_normalized: Dict[str, List[float]]


# =============================================================================
# Generation Logprobs Analysis (no re-prefill)
# =============================================================================

# Special tokens used by gpt-oss to delimit sections in generated output.
_OSS_CHANNEL = "<|channel|>"
_OSS_MESSAGE = "<|message|>"
_OSS_END = "<|end|>"
_OSS_CALL = "<|call|>"
_OSS_START = "<|start|>"
_OSS_CONSTRAIN = "<|constrain|>"


def _classify_channel(tokens: List[str], channel_idx: int) -> str:
    """Determine section type from the token(s) following <|channel|>.

    Returns one of: 'reasoning', 'tool_call', 'text'.

    The model may use *either* ``analysis`` or ``commentary`` as the channel
    name for tool calls (the distinguishing feature is the presence of
    ``to=functions.`` in the header).  We therefore scan all header tokens
    between ``<|channel|>`` and ``<|message|>`` for that marker.
    """
    # Collect header tokens between <|channel|> and <|message|>/<|constrain|>
    header_tokens: List[str] = []
    for j in range(channel_idx + 1, min(channel_idx + 15, len(tokens))):
        if tokens[j] in (_OSS_MESSAGE, _OSS_END, _OSS_CALL, _OSS_START):
            break
        # Include <|constrain|> as a marker but keep scanning
        if tokens[j] == _OSS_CONSTRAIN:
            break
        header_tokens.append(tokens[j])

    header = "".join(header_tokens).strip().lower()

    # Tool call: header contains "to=functions." regardless of channel name
    if "to=functions." in header or "to=functions." in header.replace(" ", ""):
        return "tool_call"

    if header.startswith("analysis"):
        return "reasoning"
    elif header.startswith("comment"):
        return "tool_call"
    elif header.startswith("final"):
        return "text"
    return "unknown"


@dataclass
class GenSection:
    """A section parsed from generation logprobs."""

    content_type: str  # 'reasoning', 'tool_call', 'text', 'header'
    start: int  # index into logprobs list (inclusive)
    end: int  # index into logprobs list (exclusive)
    token_count: int
    mean_nll: float
    mean_top1_nll: float

    @property
    def gap(self) -> float:
        return self.mean_nll - self.mean_top1_nll


def _get_section_role(tokens: List[str], channel_idx: int) -> str:
    """Determine role by looking back from <|channel|> to the preceding <|start|>.

    Returns 'tool' if the header starts with 'functions.', else 'assistant'.
    """
    # Scan backwards to find <|start|>
    for j in range(channel_idx - 1, max(channel_idx - 30, -1), -1):
        if tokens[j] == _OSS_START:
            # Collect header text between <|start|> and <|channel|>
            header = "".join(tokens[j + 1 : channel_idx])
            if header.strip().startswith("functions."):
                return "tool"
            return "assistant"
    return "assistant"


def analyze_generation_logprobs(
    logprobs: List[Dict[str, Any]],
) -> List[GenSection]:
    """Parse logprobs into sections and compute NLL / Top1-NLL.

    Identifies reasoning, tool_call, tool_response, and text sections from the
    special-token structure in gpt-oss output.  Works on both generation-only
    and combined (prefill + generation) logprobs.  Metrics are computed only on
    *content* tokens (between ``<|message|>`` and ``<|end|>``/``<|call|>``),
    excluding header / control tokens.

    Args:
        logprobs: Per-token logprob dicts (``token``, ``logprob``,
            ``top_logprobs``).

    Returns:
        List of ``GenSection`` with per-section metrics.
    """
    if not logprobs:
        return []

    tokens = [lp["token"] for lp in logprobs]
    n = len(tokens)

    sections: List[GenSection] = []

    # State machine: scan for <|channel|> → classify → skip to <|message|> →
    # collect content until <|end|>/<|call|>
    i = 0
    while i < n:
        # Look for the next <|channel|> (section start)
        if tokens[i] == _OSS_CHANNEL:
            content_type = _classify_channel(tokens, i)

            # Distinguish tool response from tool call by checking header
            if content_type == "tool_call":
                role = _get_section_role(tokens, i)
                if role == "tool":
                    content_type = "tool_resp"

            # Skip to <|message|> to find content start
            j = i + 1
            while j < n and tokens[j] != _OSS_MESSAGE:
                j += 1
            content_start = j + 1  # first content token

            # Find section end: <|end|> or <|call|>
            k = content_start
            while k < n and tokens[k] not in (_OSS_END, _OSS_CALL, _OSS_CHANNEL, _OSS_START):
                k += 1
            content_end = k  # exclusive

            # Compute metrics on content tokens only
            content_lps = logprobs[content_start:content_end]
            if content_lps:
                nlls = [-lp["logprob"] for lp in content_lps
                        if lp["logprob"] is not None]
                top1_nlls = []
                for lp in content_lps:
                    if lp.get("top_logprobs"):
                        top1_nlls.append(-lp["top_logprobs"][0]["logprob"])
                    elif lp["logprob"] is not None:
                        top1_nlls.append(-lp["logprob"])

                mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
                mean_top1 = sum(top1_nlls) / len(top1_nlls) if top1_nlls else float("nan")

                sections.append(GenSection(
                    content_type=content_type,
                    start=content_start,
                    end=content_end,
                    token_count=len(content_lps),
                    mean_nll=mean_nll,
                    mean_top1_nll=mean_top1,
                ))

            i = k  # continue after section end
        else:
            i += 1

    return sections


def print_generation_analysis(
    all_logprobs: List[Optional[List[Dict[str, Any]]]],
    prefill_counts: Optional[List[int]] = None,
    label: str = "Per-Section Analysis (NLL & Top1-NLL)",
) -> None:
    """Parse and print per-section NLL / Top1-NLL from logprobs.

    Args:
        all_logprobs: List of per-interaction logprobs. Can be generation-only
            (from ``tracker.get_all_logprobs()``) or combined prefill+generation
            (from ``tracker.get_combined_logprobs()``).
        prefill_counts: Optional list of prefill token counts per interaction.
            When provided, sections whose *end* index falls within the prefill
            range are labelled ``[P]`` (prompt) and excluded from the totals.
            Pass ``tracker.get_prefill_token_counts()`` or construct manually.
        label: Header label for the table.
    """
    print(f"\n{'=' * 60}")
    print(f"{label}:")
    src_col = "  Src" if prefill_counts else ""
    print(f"  {'Inter':<6} {'Type':<12} {'Tokens':>6}  {'NLL':>8}  {'Top1-NLL':>8}{src_col}")
    print(f"  {'-'*6} {'-'*12} {'-'*6}  {'-'*8}  {'-'*8}{'-'*5 if prefill_counts else ''}")

    total_nll_sum = 0.0
    total_top1_sum = 0.0
    total_tokens = 0

    for idx, lps in enumerate(all_logprobs):
        if not lps:
            print(f"  I{idx:<5} {'(no logprobs)':<12}")
            continue

        n_prefill = prefill_counts[idx] if prefill_counts and idx < len(prefill_counts) else 0
        sections = analyze_generation_logprobs(lps)
        for sec in sections:
            is_prefill = sec.end <= n_prefill
            src = "  [P]" if is_prefill else "  [G]" if prefill_counts else ""
            print(
                f"  I{idx:<5} {sec.content_type:<12} {sec.token_count:>6}"
                f"  {sec.mean_nll:>8.4f}  {sec.mean_top1_nll:>8.4f}{src}"
            )
            if not is_prefill:
                total_nll_sum += sec.mean_nll * sec.token_count
                total_top1_sum += sec.mean_top1_nll * sec.token_count
                total_tokens += sec.token_count

    print(f"  {'-'*6} {'-'*12} {'-'*6}  {'-'*8}  {'-'*8}{'-'*5 if prefill_counts else ''}")
    if total_tokens > 0:
        overall_nll = total_nll_sum / total_tokens
        overall_top1 = total_top1_sum / total_tokens
        gen_label = "GEN" if prefill_counts else "ALL"
        print(
            f"  {gen_label:<6} {'':<12} {total_tokens:>6}"
            f"  {overall_nll:>8.4f}  {overall_top1:>8.4f}"
        )


def print_tool_response_errors(
    all_logprobs: List[Optional[List[Dict[str, Any]]]],
    top_n: int = 20,
    label: str = "Top-1 Mispredicted Tokens in Tool Responses",
) -> None:
    """Print top-1 mispredicted tokens in tool response sections.

    For each token in tool_resp sections, checks whether the model's top-1
    predicted token differs from the actual token. Aggregates error counts
    and prints a ranked table of the most common mispredictions.

    Args:
        all_logprobs: List of per-interaction logprobs (generation-only or
            combined prefill+generation).
        top_n: Number of top mispredicted tokens to display.
        label: Header label for the table.
    """
    error_counts: Dict[str, int] = {}
    error_examples: Dict[str, List[Tuple[str, float]]] = {}
    total_tool_tokens = 0
    n_errors = 0

    for lps in all_logprobs:
        if not lps:
            continue
        for sec in analyze_generation_logprobs(lps):
            if sec.content_type != "tool_resp":
                continue
            for lp in lps[sec.start : sec.end]:
                if lp.get("logprob") is None or not lp.get("top_logprobs"):
                    continue
                total_tool_tokens += 1
                actual = lp["token"]
                top1 = lp["top_logprobs"][0]["token"]
                if top1 != actual:
                    n_errors += 1
                    error_counts[top1] = error_counts.get(top1, 0) + 1
                    if top1 not in error_examples or len(error_examples[top1]) < 3:
                        gap = lp["top_logprobs"][0]["logprob"] - lp["logprob"]
                        error_examples.setdefault(top1, []).append((actual, gap))

    print(f"\n{'=' * 60}")
    print(f"{label}:")
    if total_tool_tokens > 0:
        print(
            f"  Tool response tokens: {total_tool_tokens}, "
            f"top-1 errors: {n_errors} ({100 * n_errors / total_tool_tokens:.1f}%)"
        )
        print(
            f"\n  {'Rank':<5} {'Top-1 Predicted':<20} {'Count':>5}"
            f"  {'Example actual tokens'}"
        )
        print(f"  {'-' * 5} {'-' * 20} {'-' * 5}  {'-' * 40}")
        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:top_n]
        for rank, (tok, cnt) in enumerate(sorted_errors, 1):
            tok_repr = repr(tok)
            examples = error_examples.get(tok, [])
            ex_str = ", ".join(repr(a) for a, _ in examples[:3])
            print(f"  {rank:<5} {tok_repr:<20} {cnt:>5}  instead of: {ex_str}")
    else:
        print("  No tool response sections found.")


# =============================================================================
# Backend Abstraction
# =============================================================================


def get_prefill_result_local(
    model,
    input_ids: torch.Tensor,
    device: Optional[torch.device] = None,
) -> PrefillResult:
    """Get PrefillResult from local HuggingFace model.

    Args:
        model: HuggingFace model.
        input_ids: Token IDs [seq_len].
        device: Device to run on (None lets model handle placement).

    Returns:
        PrefillResult with precomputed entropy (logits are not stored to save memory).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if device is not None:
        input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits.squeeze(0)
    input_ids = input_ids.squeeze(0)

    return hf_logits_to_prefill_result(logits, input_ids)


def get_prefill_result_fireworks(
    input_ids: torch.Tensor,
    model: str = "accounts/fireworks/models/gpt-oss-20b",
    api_key: Optional[str] = None,
) -> PrefillResult:
    """Get PrefillResult from Fireworks API.

    Args:
        input_ids: Token IDs [seq_len].
        model: Fireworks model name.
        api_key: API key (or from env).

    Returns:
        PrefillResult (without full logits).
    """
    token_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    return fireworks_prefill(token_ids, model=model, api_key=api_key)


def logprobs_to_prefill_result(
    logprobs: List[Dict[str, Any]],
    input_ids: torch.Tensor,
) -> PrefillResult:
    """Convert tracker logprobs format to PrefillResult.

    Tracker logprobs are dicts with keys: token, logprob, top_logprobs.
    This converts them into the unified PrefillResult format used by
    the metric computation pipeline.

    Args:
        logprobs: List of per-token logprob dicts from the tracker.
            Each dict has: {"token": str, "logprob": float,
            "top_logprobs": [{"token": str, "logprob": float}, ...]}.
        input_ids: Token IDs tensor [seq_len] from the TokenizedConversation.

    Returns:
        PrefillResult aligned to input_ids length.
    """
    # Truncate to min length to handle trailing template tokens
    n = min(len(logprobs), len(input_ids))

    tokens = [logprobs[i]["token"] for i in range(n)]
    lp_values = [logprobs[i].get("logprob") for i in range(n)]
    top_lps = [logprobs[i].get("top_logprobs", []) for i in range(n)]

    return fireworks_to_prefill_result(
        tokens=tokens,
        token_ids=input_ids[:n].tolist(),
        logprobs=lp_values,
        top_logprobs=top_lps,
    )


def analyze_conversation(
    conversation: TokenizedConversation,
    backend: str = "local",
    model=None,
    device: Optional[torch.device] = None,
    fireworks_model: str = "accounts/fireworks/models/gpt-oss-20b",
    fireworks_api_key: Optional[str] = None,
    metrics: Optional[List[UnifiedMetric]] = None,
    prefill_result: Optional[PrefillResult] = None,
) -> AnalysisResult:
    """Analyze a single conversation using specified backend.

    Args:
        conversation: Tokenized conversation with section tracking.
        backend: "local" for HuggingFace model, "fireworks" for Fireworks API.
        model: HuggingFace model (required for local backend).
        device: Device for local computation.
        fireworks_model: Fireworks model name.
        fireworks_api_key: Fireworks API key.
        metrics: List of UnifiedMetric to compute. Defaults to standard set.
        prefill_result: Pre-computed PrefillResult. When provided, skips the
            backend prefill call entirely (avoids extra API calls).

    Returns:
        AnalysisResult with per-token and per-section metrics.
    """
    # Get PrefillResult from appropriate backend (skip if pre-computed)
    if prefill_result is None:
        if backend == "fireworks":
            prefill_result = get_prefill_result_fireworks(
                conversation.input_ids,
                model=fireworks_model,
                api_key=fireworks_api_key,
            )
        else:
            if model is None:
                raise ValueError("model is required for local backend")
            prefill_result = get_prefill_result_local(
                model, conversation.input_ids, device=device
            )

    # Use default metrics if not specified
    if metrics is None:
        metrics = [
            NegLogProbUnified(),
            Top1NegLogProbUnified(),
            PerplexityUnified(),
            EstimatedEntropyUnified(),  # Exact for HF, tight lower bound for API
            TopKEntropyUnified(),       # Renormalized top-k (comparable across backends)
            TopKMassUnified(),
        ]

    # Compute metrics using unified interface
    metric_values = compute_unified_metrics(prefill_result, metrics)
    token_count = prefill_result.seq_len

    # Per-section aggregation
    section_metrics = []
    for section in conversation.sections:
        section_data: Dict[str, float] = {}

        for metric_name, values in metric_values.items():
            start = section.start_idx
            end = min(len(values), section.end_idx)

            if end > start:
                section_values = values[start:end]
                valid_mask = ~torch.isnan(section_values)
                if valid_mask.any():
                    section_data[metric_name] = section_values[valid_mask].mean().item()
                else:
                    section_data[metric_name] = float("nan")
            else:
                section_data[metric_name] = float("nan")

        section_metrics.append(
            SectionMetrics(
                role=section.role,
                content_type=section.content_type,
                token_count=section.length,
                start_idx=section.start_idx,
                end_idx=section.end_idx,
                metrics=section_data,
                message_uid=section.message_uid,
                original_uid=section.original_uid,
                transform_type=section.transform_type,
            )
        )

    # Overall metrics (ignoring NaN)
    overall = {}
    metrics_by_position = {}
    for metric_name, values in metric_values.items():
        valid_mask = ~torch.isnan(values)
        if valid_mask.any():
            overall[metric_name] = values[valid_mask].mean().item()
        else:
            overall[metric_name] = float("nan")
        metrics_by_position[metric_name] = values.tolist()

    return AnalysisResult(
        conversation_id=conversation.conversation_id or "unknown",
        token_count=token_count,
        sections=section_metrics,
        metrics_by_position=metrics_by_position,
        overall_metrics=overall,
    )


def aggregate_results(results: List[AnalysisResult]) -> AggregatedMetrics:
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
            if not (isinstance(value, float) and math.isnan(value)):
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
                if isinstance(value, float) and math.isnan(value):
                    continue
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
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }

    by_content_type = {}
    for ctype, metrics in content_type_metrics.items():
        by_content_type[ctype] = {
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }

    overall = {
        name: sum(values) / len(values)
        for name, values in overall_values.items()
        if values
    }

    return AggregatedMetrics(
        num_conversations=len(results),
        total_tokens=total_tokens,
        by_role=by_role,
        by_content_type=by_content_type,
        overall=overall,
        by_position_normalized={},
    )


# =============================================================================
# Plot Source Data Export
# =============================================================================


def _open_text_file(path: Path) -> IO[str]:
    """Open a text file for writing, supporting optional gzip via .gz suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def save_token_plot_data_csv(
    results: List[Any],
    metrics: List[TokenMetric],
    output_path: Path,
    include_uid: bool = True,
):
    """Save token-level plot source data to a CSV for fast re-plotting.

    Writes one row per token with role/content_type labels derived from section
    boundaries and one column per metric. Shifted metrics are aligned to the
    predicted token position (i+1).

    Args:
        results: List of AnalysisResult-like objects (must include per-token metric arrays).
        metrics: Metric objects (used for name + is_shifted alignment).
        output_path: Output CSV path. Use ".csv.gz" to gzip.
        include_uid: Whether to include UID columns (message_uid, original_uid, transform_type).
    """
    metric_names = [m.name for m in metrics]
    metric_lookup = {m.name: m for m in metrics}

    fieldnames = [
        "conversation_id",
        "token_idx",
        "section_idx",
    ]
    if include_uid:
        fieldnames.extend(["message_uid", "original_uid", "transform_type"])
    fieldnames.extend([
        "role",
        "content_type",
        *metric_names,
    ])

    with _open_text_file(output_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            token_count = int(getattr(result, "token_count"))

            roles = ["unknown"] * token_count
            content_types = ["unknown"] * token_count
            section_indices = [-1] * token_count
            message_uids = [-1] * token_count
            original_uids = [-1] * token_count
            transform_types = ["unknown"] * token_count

            for idx, section in enumerate(getattr(result, "sections")):
                start = max(0, int(getattr(section, "start_idx")))
                end = min(token_count, int(getattr(section, "end_idx")))
                if end <= start:
                    continue
                role = str(getattr(section, "role"))
                content_type = str(getattr(section, "content_type"))
                msg_uid = int(getattr(section, "message_uid", -1))
                orig_uid = int(getattr(section, "original_uid", msg_uid))
                transform = str(getattr(section, "transform_type", "original"))

                roles[start:end] = [role] * (end - start)
                content_types[start:end] = [content_type] * (end - start)
                section_indices[start:end] = [idx] * (end - start)
                message_uids[start:end] = [msg_uid] * (end - start)
                original_uids[start:end] = [orig_uid] * (end - start)
                transform_types[start:end] = [transform] * (end - start)

            metrics_by_position = getattr(result, "metrics_by_position", {}) or {}

            # Align all metrics to token positions (0..token_count-1).
            aligned_by_metric: Dict[str, List[float]] = {}
            for name in metric_names:
                values = metrics_by_position.get(name)
                aligned = [float("nan")] * token_count

                if values is None:
                    aligned_by_metric[name] = aligned
                    continue

                is_shifted = metric_lookup.get(name).is_shifted if name in metric_lookup else False
                if is_shifted and len(values) != token_count:
                    # Shifted metrics usually have length seq_len-1; index i predicts token i+1.
                    for i, v in enumerate(values):
                        pos = i + 1
                        if pos >= token_count:
                            break
                        aligned[pos] = v
                else:
                    for i, v in enumerate(values[:token_count]):
                        aligned[i] = v

                aligned_by_metric[name] = aligned

            for token_idx in range(token_count):
                row: Dict[str, Any] = {
                    "conversation_id": getattr(result, "conversation_id", "unknown"),
                    "token_idx": token_idx,
                    "section_idx": section_indices[token_idx],
                    "role": roles[token_idx],
                    "content_type": content_types[token_idx],
                }
                if include_uid:
                    row["message_uid"] = message_uids[token_idx]
                    row["original_uid"] = original_uids[token_idx]
                    row["transform_type"] = transform_types[token_idx]
                for name in metric_names:
                    v = aligned_by_metric[name][token_idx]
                    row[name] = v if math.isfinite(v) else ""
                writer.writerow(row)

    print(f"Saved token-level plot data to {output_path}")


def save_transform_log_csv(
    transform_logs: List[Tuple[str, List[Any]]],
    output_path: Path,
):
    """Save transformation log to CSV.

    Args:
        transform_logs: List of (conversation_id, transform_records) tuples.
        output_path: Output CSV path.
    """
    fieldnames = [
        "conversation_id",
        "original_uid",
        "new_uid",
        "transform_type",
        "role",
        "original_char_count",
        "new_char_count",
    ]

    with _open_text_file(output_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for conv_id, records in transform_logs:
            for record in records:
                row = {
                    "conversation_id": conv_id,
                    "original_uid": getattr(record, "original_uid", -1),
                    "new_uid": getattr(record, "new_uid", -1),
                    "transform_type": getattr(record, "transform_type", "unknown"),
                    "role": getattr(record, "role", "unknown"),
                    "original_char_count": getattr(record, "original_char_count", 0),
                    "new_char_count": getattr(record, "new_char_count", 0),
                }
                writer.writerow(row)

    print(f"Saved transform log to {output_path}")
