"""Efficient SFT dataset with pre-tokenization and lazy tensor conversion.

Pre-tokenizes all samples once during construction (seconds for medium datasets).
Keeps data in Arrow format and converts to tensors lazily in ``__getitem__``.

Usage::

    from rosetta.optimize.dataset import PackedSFTDataset, collate_padded

    dataset = PackedSFTDataset(hf_dataset, tokenizer, max_length=4096)
    loader = DataLoader(dataset, batch_size=2,
                        collate_fn=lambda b: collate_padded(b, tokenizer.pad_token_id))
"""

from __future__ import annotations

import json
import os

# Pre-tokenization happens before DataLoader forks; workers only return
# pre-computed tensors, so tokenizer parallelism is unnecessary.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rosetta.optimize.utils import tool_meta_key


# ---------------------------------------------------------------------------
# Multi-turn splitting
# ---------------------------------------------------------------------------


def split_multi_turn(messages, tools):
    """Split a multi-turn conversation at user-message boundaries.

    Each user message starts a new "round".  The first round includes
    everything from the beginning up to (but not including) the second
    user message, the second round extends from the beginning up to (but
    not including) the third user message, and so on.

    Only the assistant turns in the **last round** of each split are
    supervised during training (the label builders mask earlier assistant
    turns automatically by supervising from the last user message).

    Returns a list of dicts, each with:

    - ``messages`` – message prefix for this split
    - ``tools`` – unchanged tool list
    - ``round`` – 0-based round index
    - ``total_rounds`` – total number of rounds produced
    """
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

    if len(user_indices) <= 1:
        return [{"messages": messages, "tools": tools, "round": 0, "total_rounds": 1}]

    splits: list = []
    for k, u_idx in enumerate(user_indices):
        end = user_indices[k + 1] if k + 1 < len(user_indices) else len(messages)
        prefix = messages[:end]

        # Skip splits with no assistant content after this user
        if not any(m["role"] == "assistant" for m in prefix[u_idx:]):
            continue

        splits.append({
            "messages": prefix,
            "tools": tools,
            "round": len(splits),
            "total_rounds": -1,  # filled below
        })

    if not splits:
        return [{"messages": messages, "tools": tools, "round": 0, "total_rounds": 1}]

    for s in splits:
        s["total_rounds"] = len(splits)
    return splits


# ---------------------------------------------------------------------------
# Supervision role configuration
# ---------------------------------------------------------------------------

_VALID_SUPERVISE_ROLES = {"assistant", "tool", "tool_call"}


def parse_supervise_roles(value: str) -> frozenset:
    """Parse a comma-separated supervision role string into a frozenset.

    Valid tokens: ``assistant``, ``tool``, ``tool_call``.
    """
    roles = frozenset(t.strip() for t in value.split(",") if t.strip())
    invalid = roles - _VALID_SUPERVISE_ROLES
    if invalid:
        raise ValueError(f"Invalid supervise role(s): {invalid}. Valid: {_VALID_SUPERVISE_ROLES}")
    if not roles:
        raise ValueError("supervise_roles must not be empty")
    return roles


def _should_supervise(msg: dict, supervise_roles: frozenset) -> bool:
    """Return True if this message should be supervised given the role config."""
    role = msg.get("role")
    if role == "assistant":
        if "assistant" in supervise_roles:
            return True
        return "tool_call" in supervise_roles and bool(msg.get("tool_calls"))
    if role == "tool":
        return "tool" in supervise_roles
    return False


# ---------------------------------------------------------------------------
# Label building: find supervised turn boundaries
# ---------------------------------------------------------------------------


def _last_user_idx(messages):
    """Return the index of the last user message, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1


def _build_labels_native(tokenizer, messages, tools, template_kwargs, supervise_roles=None):
    """Single-pass boundary detection via HF's return_assistant_tokens_mask.

    Requires the chat template to include {% generation %} tags.
    Returns None if the template doesn't support it (mask is all zeros),
    or if ``supervise_roles`` targets non-assistant roles (HF only masks
    assistant tokens natively).
    """
    # HF native method only supports assistant role supervision
    if supervise_roles is not None and supervise_roles != frozenset({"assistant"}):
        return None
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            **template_kwargs,
        )
    except TypeError:
        # Older transformers without return_assistant_tokens_mask
        return None

    input_ids = result["input_ids"]
    mask = result["assistant_masks"]

    # All-zero mask means the template lacks {% generation %} tags
    if not any(mask):
        return None

    labels = [tok if m else -100 for tok, m in zip(input_ids, mask)]

    # Mask out assistant turns before the last user message
    lu = _last_user_idx(messages)
    if lu > 0 and any(m["role"] == "assistant" for m in messages[:lu]):
        prefix_ids = tokenizer.apply_chat_template(
            messages[:lu], tools=tools, tokenize=True,
            add_generation_prompt=False, **template_kwargs,
        )
        labels[:len(prefix_ids)] = [-100] * len(prefix_ids)

    return input_ids, labels


def _build_labels_progressive(tokenizer, messages, tools, template_kwargs, supervise_roles=None):
    """Fallback: progressive tokenization to find per-turn boundaries.

    Calls apply_chat_template N+1 times (once full, once per prefix).
    Only turns matching ``supervise_roles`` after the last user message
    are supervised.
    """
    if supervise_roles is None:
        supervise_roles = frozenset({"assistant"})
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True,
        add_generation_prompt=False, **template_kwargs,
    )

    lu = _last_user_idx(messages)

    boundaries = []
    for i in range(len(messages)):
        ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
        boundaries.append(len(ids))

    labels = [-100] * len(full_ids)
    for i, msg in enumerate(messages):
        if i > lu and _should_supervise(msg, supervise_roles):
            start = boundaries[i - 1] if i > 0 else 0
            end = boundaries[i]
            labels[start:end] = full_ids[start:end]

    return full_ids, labels


def tokenize_last_turn(tokenizer, messages, tools, max_length, template_kwargs):
    """Tokenize conversation and supervise only the last message.

    Args:
        tokenizer: HuggingFace tokenizer.
        messages: Full conversation (last message = completion to supervise).
        tools: Tool schemas (list of dicts) or None.
        max_length: Truncation length.
        template_kwargs: Chat template kwargs (e.g. ``enable_thinking=False``).

    Returns:
        ``(input_ids, labels)`` where labels = -100 except for last message tokens.
    """
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True,
        add_generation_prompt=False, **template_kwargs,
    )
    prefix_ids = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, tokenize=True,
        add_generation_prompt=True, **template_kwargs,
    )
    labels = [-100] * len(full_ids)
    labels[len(prefix_ids):] = full_ids[len(prefix_ids):]

    return list(full_ids[:max_length]), list(labels[:max_length])


def fill_reasoning(messages):
    """Set ``reasoning_content='\\n'`` on all assistant messages.

    Use as a ``pre_processor`` for :class:`PackedSFTDataset` so that
    Qwen3's chat template always renders empty ``<think>`` blocks,
    matching inference-time behaviour when thinking is disabled.

    Any existing ``reasoning_content`` (e.g. from a different source
    model like kimi) is replaced with ``'\\n'`` so it doesn't leak
    into the target model's training.

    Note: pre-processors must not insert or remove messages — only modify
    existing ones — so that message indices stay consistent with the
    label builders' last-user-message detection.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            m["reasoning_content"] = "\n"
        out.append(m)
    return out


def tokenize_item(tokenizer, item, max_length, template_kwargs, pre_processor=None, supervise_roles=None):
    """Tokenize one chat trajectory and build role-masked labels.

    Returns ``(input_ids, labels, meta_key)`` or ``None`` on empty input.
    """
    messages = json.loads(item["messages"])
    tools = json.loads(item["tools"]) or None

    if not messages:
        return None

    # Compute meta_key for grouping by tool set
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    mk = tool_meta_key(tools, system_msg)

    if pre_processor is not None:
        messages = pre_processor(messages)

    # Try native single-pass first, fall back to progressive
    result = _build_labels_native(tokenizer, messages, tools, template_kwargs, supervise_roles)
    if result is None:
        result = _build_labels_progressive(tokenizer, messages, tools, template_kwargs, supervise_roles)

    input_ids, labels = result
    input_ids = list(input_ids[: max_length])
    labels = list(labels[: max_length])
    return input_ids, labels, mk


def _tokenize_batch(examples, tokenizer, max_length, template_kwargs, pre_processor=None, supervise_roles=None):
    """Batched map function for Dataset.map(). Returns per-row lists.

    Also computes ``_valid`` (bool) and ``_n_supervised`` (int) per item
    so that downstream filtering and stats can avoid iterating over tokens.
    """
    all_input_ids = []
    all_labels = []
    all_meta_keys = []
    all_valid = []
    all_n_supervised = []
    for msg, tools in zip(examples["messages"], examples["tools"]):
        result = tokenize_item(
            tokenizer, {"messages": msg, "tools": tools}, max_length, template_kwargs,
            pre_processor=pre_processor, supervise_roles=supervise_roles,
        )
        if result is not None:
            ids, labels, mk = result
            valid = bool(ids) and any(l != -100 for l in labels)
            n_sup = sum(1 for l in labels if l != -100) if valid else 0
            all_input_ids.append(ids)
            all_labels.append(labels)
            all_meta_keys.append(mk)
            all_valid.append(valid)
            all_n_supervised.append(n_sup)
        else:
            all_input_ids.append([])
            all_labels.append([])
            all_meta_keys.append("")
            all_valid.append(False)
            all_n_supervised.append(0)
    return {
        "_input_ids": all_input_ids,
        "_labels": all_labels,
        "_meta_key": all_meta_keys,
        "_valid": all_valid,
        "_n_supervised": all_n_supervised,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PackedSFTDataset(Dataset):
    """Pre-tokenized SFT dataset with lazy tensor conversion.

    All tokenization happens once in ``__init__`` via ``Dataset.map()``.
    Valid samples are kept in Arrow format (``self._data``); tensors are
    created lazily in ``__getitem__``.

    Args:
        hf_dataset: HF dataset with ``messages`` and ``tools`` JSON columns.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        template_kwargs: Extra kwargs for ``apply_chat_template``
            (e.g. ``enable_thinking=False``).
        keep_raw: If ``True``, preserve raw ``messages``/``tools`` JSON
            strings in each sample (as ``_messages``/``_tools``).
        passthrough_columns: Column names to preserve unchanged through
            tokenization and include in each sample dict.  Useful for
            carrying per-sample metadata (e.g. ``trainable_tools``)
            into the training loop.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 4096,
        template_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        pre_processor=None,
        keep_raw: bool = False,
        passthrough_columns: Optional[List[str]] = None,
        supervise_roles: Optional[frozenset] = None,
    ):
        self.max_length = max_length
        self.keep_raw = keep_raw
        self.passthrough_columns = passthrough_columns or []

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        self.pad_token_id = pad_token_id

        tk = template_kwargs or {}

        # -- Pre-tokenize all samples (parallelized via Dataset.map) --
        n = len(hf_dataset)
        if num_proc is None:
            if n >= 500:
                num_proc = min(os.cpu_count() or 1, 16, max(2, n // 250))
            else:
                num_proc = 1

        # Preserve requested columns through .map()
        remove_cols = hf_dataset.column_names
        if keep_raw:
            remove_cols = [c for c in remove_cols if c not in ("messages", "tools")]
        if self.passthrough_columns:
            remove_cols = [c for c in remove_cols if c not in self.passthrough_columns]

        tokenized = hf_dataset.map(
            _tokenize_batch,
            batched=True,
            batch_size=256,
            num_proc=num_proc if num_proc > 1 else None,
            fn_kwargs=dict(
                tokenizer=tokenizer, max_length=max_length, template_kwargs=tk,
                pre_processor=pre_processor, supervise_roles=supervise_roles,
            ),
            remove_columns=remove_cols,
            # load_from_cache_file=False,
            desc="Tokenizing",
        )

        # -- Arrow-level filter (no Python iteration over tokens) --
        n_before = len(tokenized)
        filtered = tokenized.filter(lambda x: x["_valid"], desc="Filtering")
        skipped = n_before - len(filtered)

        # -- Stats from pre-computed columns (no token iteration) --
        n_supervised_col = filtered["_n_supervised"]
        supervised = sum(n_supervised_col)
        total_tokens = sum(len(ids) for ids in filtered["_input_ids"])
        print(
            f"Pre-tokenized {len(filtered)} samples ({skipped} skipped) | "
            f"{total_tokens:,} tokens, {supervised:,} supervised "
            f"({supervised / total_tokens * 100:.1f}%)"
        )

        self._data = filtered
        self.meta_keys: List[str] = filtered["_meta_key"]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        ids = row["_input_ids"]
        sample = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(row["_labels"], dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "meta_key": row["_meta_key"],
        }
        if self.keep_raw and "messages" in row:
            sample["_messages"] = row["messages"]
            sample["_tools"] = row["tools"]
        for col in self.passthrough_columns:
            if col in row:
                sample[col] = row[col]
        return sample


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------


def collate_padded(
    batch: List[Dict[str, Any]], pad_token_id: int = 0
) -> Dict[str, Any]:
    """Right-pad variable-length samples to the batch max.

    Padding values: ``input_ids`` → *pad_token_id*, ``labels`` → -100,
    everything else → 0.  Non-tensor fields (e.g. ``meta_key``) are
    collected as plain lists.
    """
    pad_values = {"input_ids": pad_token_id, "labels": -100}
    result = {}
    for k in batch[0]:
        values = [b[k] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[k] = pad_sequence(
                values, batch_first=True, padding_value=pad_values.get(k, 0)
            )
        else:
            result[k] = values
    return result
