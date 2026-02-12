"""Efficient SFT dataset with pre-tokenization and optional sequence packing.

Pre-tokenizes all samples once during construction (seconds for medium datasets).
When packing is enabled, multiple short sequences are packed into fixed-length bins
via Best-Fit Decreasing, eliminating padding waste.

Usage::

    from rosetta.optimize.dataset import PackedSFTDataset, collate_padded

    dataset = PackedSFTDataset(hf_dataset, tokenizer, max_length=4096, pack=True)
    loader = DataLoader(dataset, batch_size=2,
                        collate_fn=lambda b: collate_padded(b, tokenizer.pad_token_id))
"""

from __future__ import annotations

import json
import os

# Pre-tokenization happens before DataLoader forks; workers only return
# pre-computed tensors, so tokenizer parallelism is unnecessary.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Label building: find assistant turn boundaries
# ---------------------------------------------------------------------------


def _build_labels_native(tokenizer, messages, tools, template_kwargs):
    """Single-pass boundary detection via HF's return_assistant_tokens_mask.

    Requires the chat template to include {% generation %} tags.
    Returns None if the template doesn't support it (mask is all zeros).
    """
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
    return input_ids, labels


def _build_labels_progressive(tokenizer, messages, tools, template_kwargs):
    """Fallback: progressive tokenization to find per-turn boundaries.

    Calls apply_chat_template N+1 times (once full, once per prefix).
    """
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True,
        add_generation_prompt=False, **template_kwargs,
    )

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
        if msg["role"] == "assistant":
            start = boundaries[i - 1] if i > 0 else 0
            end = boundaries[i]
            labels[start:end] = full_ids[start:end]

    return full_ids, labels


def fill_reasoning(messages):
    """Add ``reasoning_content='\\n'`` to assistant messages missing it.

    Use as a ``pre_processor`` for :class:`PackedSFTDataset` so that
    Qwen3's chat template always renders ``<think>`` blocks, matching
    inference-time behaviour when thinking is disabled.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and "reasoning_content" not in m:
            m["reasoning_content"] = "\n"
        out.append(m)
    return out


def _tokenize_item(tokenizer, item, max_length, template_kwargs, pre_processor=None):
    """Tokenize one chat trajectory and build assistant-only labels."""
    messages = json.loads(item["messages"])
    tools = json.loads(item["tools"]) or None

    if not messages:
        return None

    if pre_processor is not None:
        messages = pre_processor(messages)

    # Try native single-pass first, fall back to progressive
    result = _build_labels_native(tokenizer, messages, tools, template_kwargs)
    if result is None:
        result = _build_labels_progressive(tokenizer, messages, tools, template_kwargs)

    input_ids, labels = result
    input_ids = list(input_ids[: max_length])
    labels = list(labels[: max_length])
    return input_ids, labels


def _tokenize_batch(examples, tokenizer, max_length, template_kwargs, pre_processor=None):
    """Batched map function for Dataset.map(). Returns per-row lists."""
    all_input_ids = []
    all_labels = []
    for msg, tools in zip(examples["messages"], examples["tools"]):
        result = _tokenize_item(
            tokenizer, {"messages": msg, "tools": tools}, max_length, template_kwargs,
            pre_processor=pre_processor,
        )
        if result is not None:
            all_input_ids.append(result[0])
            all_labels.append(result[1])
        else:
            all_input_ids.append([])
            all_labels.append([])
    return {"_input_ids": all_input_ids, "_labels": all_labels}


# ---------------------------------------------------------------------------
# BFD bin packing
# ---------------------------------------------------------------------------


def _pack_bfd(
    samples: List[Tuple[list, list]],
    max_length: int,
    pad_token_id: int,
) -> List[Dict[str, torch.Tensor]]:
    """Pack sequences into fixed-length bins using Best-Fit Decreasing.

    Each bin gets position_ids that reset at sub-sequence boundaries (for RoPE).
    """
    # Sort by length descending
    indexed = sorted(
        range(len(samples)), key=lambda i: len(samples[i][0]), reverse=True
    )

    # bins[i] = [list of sample indices, remaining capacity]
    bins: List[Tuple[List[int], int]] = []

    for idx in indexed:
        seq_len = len(samples[idx][0])
        if seq_len > max_length:
            continue

        # Find best-fit bin (least remaining capacity that still fits)
        best_bin = -1
        best_remaining = max_length + 1
        for b, (_, remaining) in enumerate(bins):
            if seq_len <= remaining < best_remaining:
                best_bin = b
                best_remaining = remaining

        if best_bin >= 0:
            bins[best_bin][0].append(idx)
            bins[best_bin] = (bins[best_bin][0], bins[best_bin][1] - seq_len)
        else:
            bins.append(([idx], max_length - seq_len))

    # Merge each bin into a single packed sequence (no padding — collator handles that)
    packed = []
    for sample_indices, _ in bins:
        all_ids: List[int] = []
        all_labels: List[int] = []
        all_position_ids: List[int] = []

        for sample_idx in sample_indices:
            input_ids, labels = samples[sample_idx]
            all_position_ids.extend(range(len(input_ids)))
            all_ids.extend(input_ids)
            all_labels.extend(labels)

        seq_len = len(all_ids)
        packed.append({
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "position_ids": torch.tensor(all_position_ids, dtype=torch.long),
        })

    return packed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PackedSFTDataset(Dataset):
    """Pre-tokenized SFT dataset with optional BFD sequence packing.

    All tokenization happens once in __init__; __getitem__ is O(1).

    Args:
        hf_dataset: HF dataset with ``messages`` and ``tools`` JSON columns.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length (and pack bin size).
        pack: If True, pack via BFD. If False, store individually.
        template_kwargs: Extra kwargs for ``apply_chat_template``
            (e.g. ``enable_thinking=False``).
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 4096,
        pack: bool = True,
        template_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        pre_processor=None,
    ):
        self.max_length = max_length
        self.pack = pack

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
        tokenized = hf_dataset.map(
            _tokenize_batch,
            batched=True,
            batch_size=256,
            num_proc=num_proc if num_proc > 1 else None,
            fn_kwargs=dict(
                tokenizer=tokenizer, max_length=max_length, template_kwargs=tk,
                pre_processor=pre_processor,
            ),
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing",
        )

        raw_samples: List[Tuple[list, list]] = []
        skipped = 0
        for ids, lab in zip(tokenized["_input_ids"], tokenized["_labels"]):
            if ids:
                raw_samples.append((ids, lab))
            else:
                skipped += 1

        total_tokens = sum(len(s[0]) for s in raw_samples)
        supervised = sum(sum(1 for l in s[1] if l != -100) for s in raw_samples)
        print(
            f"Pre-tokenized {len(raw_samples)} samples ({skipped} skipped) | "
            f"{total_tokens:,} tokens, {supervised:,} supervised "
            f"({supervised / total_tokens * 100:.1f}%)"
        )

        # -- Pack or store individually --
        if pack:
            self.samples = _pack_bfd(raw_samples, max_length, pad_token_id)
            useful = sum(s["input_ids"].shape[0] for s in self.samples)
            print(
                f"Packed {len(raw_samples)} samples into {len(self.samples)} bins "
                f"(max {max_length}) | {useful:,} tokens"
            )
        else:
            self.samples = []
            for input_ids, labels in raw_samples:
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------


def collate_padded(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Right-pad variable-length samples to the batch max.

    Works for both packed and unpacked samples.  Padding values:
    ``input_ids`` → *pad_token_id*, ``labels`` → -100, everything else → 0.
    """
    pad_values = {"input_ids": pad_token_id, "labels": -100}
    return {
        k: pad_sequence(
            [b[k] for b in batch],
            batch_first=True,
            padding_value=pad_values.get(k, 0),
        )
        for k in batch[0]
    }
