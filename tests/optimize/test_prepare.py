"""Tests for CacheOptimizeModel.prepare()."""

import pytest
import torch

from rosetta.optimize.wrapper import CacheOptimizeModel


@pytest.fixture(scope="module")
def registered_model(model_and_tokenizer):
    """Build a CacheOptimizeModel with one registered segment.

    Returns (opt_model, full_ids, seg_start, seg_end).
    """
    model, tokenizer = model_and_tokenizer
    opt_model = CacheOptimizeModel(model)

    prefix_text = "The capital of France is"
    segment_text = " Paris, a beautiful city"
    suffix_text = " known for the Eiffel Tower and fine cuisine."

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    segment_ids = tokenizer.encode(segment_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    seg_start = len(prefix_ids)
    seg_end = seg_start + len(segment_ids)

    full_ids = torch.tensor([prefix_ids + segment_ids + suffix_ids])

    seg_tensor = torch.tensor([segment_ids])
    opt_model.register(seg_tensor)

    return opt_model, full_ids, seg_start, seg_end


class TestRealisticToolPrepare:
    """Verify prepare()+forward() matches direct forward for tool segments."""

    SYSTEM_MSG = {
        "role": "system",
        "content": "You are a helpful research assistant. "
        "Use the provided tools to find information.",
    }
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search for information using a query string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a document by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "The document ID to retrieve",
                        }
                    },
                    "required": ["doc_id"],
                },
            },
        },
    ]

    @staticmethod
    def _split_template(tokenizer, system_msg, tools, user_msg=None):
        """Split chat template into prefix, segment, and suffix."""
        sys_tool_ids = tokenizer.apply_chat_template(
            [system_msg],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        sys_tool_str = tokenizer.decode(sys_tool_ids)

        first_desc = tools[0]["function"]["description"]
        desc_char_pos = sys_tool_str.find(first_desc)
        assert desc_char_pos > 0

        cumlen = 0
        split_idx = 0
        for i in range(len(sys_tool_ids)):
            cumlen += len(tokenizer.decode(sys_tool_ids[i : i + 1]))
            if cumlen > desc_char_pos:
                split_idx = i
                break

        prefix = torch.tensor([sys_tool_ids[:split_idx]])
        segment = torch.tensor([sys_tool_ids[split_idx:]])

        if user_msg is None:
            return prefix, segment

        full_ids = tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        suffix = torch.tensor([full_ids[len(sys_tool_ids):]])
        return prefix, segment, suffix

    USER_MSG = {
        "role": "user",
        "content": "What is machine learning?",
    }

    def test_prepare_matches_direct_forward(self, model_and_tokenizer):
        """Logits from prepare()+forward() match direct model forward."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        prefix, segment, suffix = self._split_template(
            tokenizer, self.SYSTEM_MSG, self.TOOLS, self.USER_MSG
        )
        prefix_len = prefix.shape[1]
        seg_len = segment.shape[1]

        opt_model.register(segment, prefix=prefix)

        full_ids = torch.cat([prefix, segment, suffix], dim=1)

        seg_start = prefix_len
        seg_end = prefix_len + seg_len

        # 1. Direct forward (reference)
        with torch.no_grad():
            direct_out = model(full_ids.to(model.device))
        direct_logits = direct_out.logits[:, seg_end:, :].float().cpu()

        # 2. Prepare + forward
        with torch.no_grad():
            result = opt_model.prepare(
                kv_cache_indices=[(seg_start, seg_end)],
                input_ids=full_ids,
            )
            prepared_out = opt_model.forward(**result)
        prepared_logits = prepared_out.logits.float().cpu()

        assert direct_logits.shape == prepared_logits.shape, (
            f"Shape mismatch: direct {direct_logits.shape} vs "
            f"prepared {prepared_logits.shape}"
        )

        direct_top1 = direct_logits.argmax(dim=-1)
        prepared_top1 = prepared_logits.argmax(dim=-1)
        assert torch.equal(direct_top1, prepared_top1), (
            f"Top-1 predictions differ:\n"
            f"  direct:   {direct_top1}\n"
            f"  prepared: {prepared_top1}"
        )

        max_diff = (direct_logits - prepared_logits).abs().max().item()
        assert max_diff < 8.0, (
            f"Logits differ too much: max_diff={max_diff:.4f}"
        )

    def test_prepare_matches_direct_generation(self, model_and_tokenizer):
        """Greedy generation from prepare+generate matches direct generate."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        prefix, segment, suffix = self._split_template(
            tokenizer, self.SYSTEM_MSG, self.TOOLS, self.USER_MSG
        )
        prefix_len = prefix.shape[1]
        seg_len = segment.shape[1]

        opt_model.register(segment, prefix=prefix)

        full_ids = torch.cat([prefix, segment, suffix], dim=1)
        full_len = full_ids.shape[1]

        seg_start = prefix_len
        seg_end = prefix_len + seg_len
        max_new = 20

        # 1. Direct generation (reference)
        with torch.no_grad():
            direct_gen = model.generate(
                full_ids.to(model.device),
                max_new_tokens=max_new,
                do_sample=False,
            )
        direct_tokens = direct_gen[0, full_len:]

        # 2. Prepare + generate
        # generate() internally trims input_ids to only process tokens
        # beyond the cache via prepare_inputs_for_generation.
        with torch.no_grad():
            result = opt_model.prepare(
                kv_cache_indices=[(seg_start, seg_end)],
                input_ids=full_ids,
            )
            cache = result["past_key_values"]

            prepared_gen = model.generate(
                full_ids.to(model.device),
                past_key_values=cache,
                max_new_tokens=max_new,
                do_sample=False,
            )
        prepared_tokens = prepared_gen[0, full_len:]

        assert torch.equal(direct_tokens, prepared_tokens), (
            f"Generated tokens differ:\n"
            f"  direct:   {tokenizer.decode(direct_tokens)!r}\n"
            f"  prepared: {tokenizer.decode(prepared_tokens)!r}"
        )

        # 3. Zero out KV params and generate again -- output should differ
        with torch.no_grad():
            for p in opt_model.parameters():
                if p.requires_grad:
                    p.zero_()

            zeroed_result = opt_model.prepare(
                kv_cache_indices=[(seg_start, seg_end)],
                input_ids=full_ids,
            )
            zeroed_cache = zeroed_result["past_key_values"]

            zeroed_gen = model.generate(
                full_ids.to(model.device),
                past_key_values=zeroed_cache,
                max_new_tokens=max_new,
                do_sample=False,
            )
        zeroed_tokens = zeroed_gen[0, full_len:]

        assert not torch.equal(direct_tokens, zeroed_tokens), (
            "Zeroed KV params produced identical output -- "
            "registered KV cache is not being used"
        )


class TestUnregisteredRangeRaises:
    def test_unregistered_range_raises(self, model_and_tokenizer):
        """prepare() with unregistered range raises ValueError."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        random_ids = torch.randint(0, 1000, (1, 10))
        with pytest.raises(ValueError, match="not registered"):
            opt_model.prepare(
                kv_cache_indices=[(0, 5)],
                input_ids=random_ids,
            )


class TestBatchMismatchRaises:
    def test_batch_mismatch_raises(self, model_and_tokenizer):
        """Different tokens at registered range across batch raises ValueError."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        segment_ids = tokenizer.encode("hello world", add_special_tokens=False)
        seg_tensor = torch.tensor([segment_ids])
        opt_model.register(seg_tensor)

        seg_len = len(segment_ids)
        row0 = segment_ids + [1, 2, 3]
        row1 = [999] * seg_len + [1, 2, 3]
        batch_ids = torch.tensor([row0, row1])

        with pytest.raises(ValueError, match="differs from element 0"):
            opt_model.prepare(
                kv_cache_indices=[(0, seg_len)],
                input_ids=batch_ids,
            )


class TestOutputInputIdsShape:
    def test_output_input_ids_shape(self, registered_model):
        """Output input_ids is (B, full_len - prefill_end)."""
        opt_model, full_ids, seg_start, seg_end = registered_model
        B, full_len = full_ids.shape

        result = opt_model.prepare(
            kv_cache_indices=[(seg_start, seg_end)],
            input_ids=full_ids,
        )

        expected_remaining = full_len - seg_end
        assert result["input_ids"].shape == (B, expected_remaining), (
            f"Expected ({B}, {expected_remaining}), "
            f"got {result['input_ids'].shape}"
        )


class TestCacheSeqLength:
    def test_cache_seq_length(self, registered_model):
        """Full-attention layer cache length equals prefill_end."""
        opt_model, full_ids, seg_start, seg_end = registered_model

        result = opt_model.prepare(
            kv_cache_indices=[(seg_start, seg_end)],
            input_ids=full_ids,
        )

        cache = result["past_key_values"]
        fa_idx = opt_model.full_attention_layers[0]
        cache_len = cache.get_seq_length(fa_idx)
        assert cache_len == seg_end, (
            f"Cache seq_length {cache_len} != prefill_end {seg_end}"
        )


class TestLabelsLength:
    def test_labels_length(self, registered_model):
        """Output labels length matches output input_ids length."""
        opt_model, full_ids, seg_start, seg_end = registered_model
        B, full_len = full_ids.shape

        labels = torch.randint(0, 1000, (B, full_len))

        result = opt_model.prepare(
            kv_cache_indices=[(seg_start, seg_end)],
            input_ids=full_ids,
            labels=labels,
        )

        assert result["labels"].shape == result["input_ids"].shape, (
            f"Labels shape {result['labels'].shape} != "
            f"input_ids shape {result['input_ids'].shape}"
        )


class TestGradientFlow:
    def test_gradient_flow(self, model_and_tokenizer):
        """forward(**prepare(...)).loss.backward() gives gradients on params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        segment_text = " is a large language model"
        segment_ids = tokenizer.encode(segment_text, add_special_tokens=False)
        seg_tensor = torch.tensor([segment_ids])
        opt_model.register(seg_tensor)

        seg_len = len(segment_ids)

        suffix_text = " that can generate text and answer questions accurately"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        full_token_ids = segment_ids + suffix_ids
        full_ids = torch.tensor([full_token_ids])

        labels = full_ids.clone()
        labels[0, :seg_len] = -100

        result = opt_model.prepare(
            kv_cache_indices=[(0, seg_len)],
            input_ids=full_ids,
            labels=labels,
        )
        output = opt_model.forward(**result)
        loss = output.loss
        assert loss is not None, "Loss is None -- labels may not be set correctly"

        loss.backward()

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")
        assert key_param.grad is not None, "Key param has no gradient"
        assert val_param.grad is not None, "Value param has no gradient"
        assert torch.any(key_param.grad != 0), "Key gradient is all zeros"
        assert torch.any(val_param.grad != 0), "Value gradient is all zeros"


class TestFrozenModel:
    def test_frozen_model(self, model_and_tokenizer):
        """After backward, all model.parameters() still have grad=None."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        segment_text = " computes attention scores"
        segment_ids = tokenizer.encode(segment_text, add_special_tokens=False)
        seg_tensor = torch.tensor([segment_ids])
        opt_model.register(seg_tensor)

        seg_len = len(segment_ids)

        suffix_text = " using key and value projections from each layer"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        full_token_ids = segment_ids + suffix_ids
        full_ids = torch.tensor([full_token_ids])

        labels = full_ids.clone()
        labels[0, :seg_len] = -100

        result = opt_model.prepare(
            kv_cache_indices=[(0, seg_len)],
            input_ids=full_ids,
            labels=labels,
        )
        output = opt_model.forward(**result)
        output.loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is None, (
                f"Model param {name} has gradient (should be frozen)"
            )
