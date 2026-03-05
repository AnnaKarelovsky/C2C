"""Tests for PackedSFTDataset correctness and performance."""

from __future__ import annotations

import json
import time

import pytest
import torch
from datasets import Dataset

from rosetta.optimize.dataset import PackedSFTDataset, collate_padded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_dataset(items):
    """Build a HuggingFace Dataset from a list of (messages, tools) tuples."""
    return Dataset.from_dict({
        "messages": [json.dumps(msgs) for msgs, _ in items],
        "tools": [json.dumps(tools) for _, tools in items],
    })


def _simple_conversation(user_text="Hello", assistant_text="Hi there"):
    """Return a minimal multi-role conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestPackedSFTDataset:
    """Core correctness tests for PackedSFTDataset."""

    def test_basic_length(self, model_and_tokenizer):
        """Dataset length equals number of valid items."""
        _, tok = model_and_tokenizer
        items = [
            (_simple_conversation("Hi", "Hello!"), None),
            (_simple_conversation("Bye", "Goodbye!"), None),
        ]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        assert len(ds) == 2

    def test_getitem_keys_and_types(self, model_and_tokenizer):
        """__getitem__ returns dict with correct keys and tensor types."""
        _, tok = model_and_tokenizer
        items = [(_simple_conversation(), None)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        sample = ds[0]

        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample
        assert "meta_key" in sample

        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dtype == torch.long
        assert isinstance(sample["labels"], torch.Tensor)
        assert sample["labels"].dtype == torch.long
        assert isinstance(sample["attention_mask"], torch.Tensor)
        assert sample["attention_mask"].dtype == torch.long
        assert isinstance(sample["meta_key"], str)

    def test_labels_masking(self, model_and_tokenizer):
        """Labels have -100 for non-supervised tokens and real IDs for supervised."""
        _, tok = model_and_tokenizer
        items = [(_simple_conversation("What is 2+2?", "The answer is 4."), None)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        sample = ds[0]

        labels = sample["labels"]
        # Must have some -100 (system + user tokens are masked)
        assert (labels == -100).any(), "Expected some masked tokens"
        # Must have some supervised tokens (assistant response)
        assert (labels != -100).any(), "Expected some supervised tokens"

    def test_attention_mask_all_ones(self, model_and_tokenizer):
        """Attention mask is all 1s (no padding in individual samples)."""
        _, tok = model_and_tokenizer
        items = [(_simple_conversation(), None)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        sample = ds[0]
        assert (sample["attention_mask"] == 1).all()

    def test_input_ids_labels_same_length(self, model_and_tokenizer):
        """input_ids, labels, and attention_mask all have same length."""
        _, tok = model_and_tokenizer
        items = [(_simple_conversation(), None)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        sample = ds[0]
        n = sample["input_ids"].shape[0]
        assert sample["labels"].shape[0] == n
        assert sample["attention_mask"].shape[0] == n

    def test_meta_keys_populated(self, model_and_tokenizer):
        """meta_keys list is populated and matches dataset length."""
        _, tok = model_and_tokenizer
        items = [
            (_simple_conversation("a", "b"), SIMPLE_TOOLS),
            (_simple_conversation("c", "d"), SIMPLE_TOOLS),
        ]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        assert len(ds.meta_keys) == len(ds)
        # With identical tools, meta_keys should be the same
        assert ds.meta_keys[0] == ds.meta_keys[1]
        assert ds.meta_keys[0] != ""

    def test_empty_messages_filtered(self, model_and_tokenizer):
        """Items with empty messages are filtered out."""
        _, tok = model_and_tokenizer
        items = [
            ([], None),  # empty messages
            (_simple_conversation("Hi", "Hello"), None),  # valid
        ]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        assert len(ds) == 1

    def test_no_assistant_filtered(self, model_and_tokenizer):
        """Items with no assistant turn (no supervised tokens) are filtered."""
        _, tok = model_and_tokenizer
        no_assistant = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        items = [
            (no_assistant, None),  # no supervised tokens
            (_simple_conversation("Hi", "Hello"), None),  # valid
        ]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        assert len(ds) == 1

    def test_keep_raw(self, model_and_tokenizer):
        """keep_raw=True preserves _messages and _tools in samples."""
        _, tok = model_and_tokenizer
        msgs = _simple_conversation("Hi", "Hello")
        items = [(msgs, SIMPLE_TOOLS)]
        ds = PackedSFTDataset(
            _make_hf_dataset(items), tok, max_length=4096, keep_raw=True,
        )
        sample = ds[0]
        assert "_messages" in sample
        assert "_tools" in sample
        # Verify content round-trips
        assert json.loads(sample["_messages"]) == msgs
        assert json.loads(sample["_tools"]) == SIMPLE_TOOLS

    def test_keep_raw_false(self, model_and_tokenizer):
        """keep_raw=False (default) does NOT include _messages/_tools."""
        _, tok = model_and_tokenizer
        items = [(_simple_conversation(), SIMPLE_TOOLS)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        sample = ds[0]
        assert "_messages" not in sample
        assert "_tools" not in sample

    def test_max_length_truncation(self, model_and_tokenizer):
        """Sequences are truncated to max_length."""
        _, tok = model_and_tokenizer
        # Make a long conversation
        long_text = "word " * 500
        items = [(_simple_conversation("Q", long_text), None)]
        max_len = 64
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=max_len)
        if len(ds) > 0:
            sample = ds[0]
            assert sample["input_ids"].shape[0] <= max_len

    def test_collate_padded(self, model_and_tokenizer):
        """collate_padded works with samples from PackedSFTDataset."""
        _, tok = model_and_tokenizer
        items = [
            (_simple_conversation("Hi", "Hello"), None),
            (_simple_conversation("What is the meaning of life?",
                                  "42, according to Douglas Adams."), None),
        ]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        batch = collate_padded([ds[0], ds[1]], pad_token_id=tok.pad_token_id)
        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        assert isinstance(batch["meta_key"], list)
        assert len(batch["meta_key"]) == 2

    def test_with_tools(self, model_and_tokenizer):
        """Dataset handles conversations with tool schemas."""
        _, tok = model_and_tokenizer
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {"role": "assistant", "content": "Let me check the weather for you."},
        ]
        items = [(msgs, SIMPLE_TOOLS)]
        ds = PackedSFTDataset(_make_hf_dataset(items), tok, max_length=4096)
        assert len(ds) == 1
        sample = ds[0]
        assert (sample["labels"] != -100).any()


# ---------------------------------------------------------------------------
# Performance regression guard
# ---------------------------------------------------------------------------


class TestPerformance:
    """Guard against init performance regressions."""

    def test_init_speed(self, model_and_tokenizer):
        """Init with many items completes in reasonable time.

        Creates 200 items with ~300-token conversations. The init should
        complete well under 60 seconds (tokenization + Arrow filter).
        """
        _, tok = model_and_tokenizer
        long_assistant = "This is a moderately long response. " * 20
        items = [
            (_simple_conversation(f"Question {i}", long_assistant), SIMPLE_TOOLS)
            for i in range(200)
        ]
        hf_ds = _make_hf_dataset(items)

        t0 = time.perf_counter()
        ds = PackedSFTDataset(hf_ds, tok, max_length=4096)
        elapsed = time.perf_counter() - t0

        assert len(ds) == 200
        assert elapsed < 60, f"Init took {elapsed:.1f}s, expected < 60s"
