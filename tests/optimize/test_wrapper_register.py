"""Tests for CacheOptimizeModel.__init__ and register()."""

import pytest
import torch

from rosetta.optimize.wrapper import CacheOptimizeModel


class TestFreeze:
    def test_freeze(self, model_and_tokenizer):
        """After wrapping, ALL model params have requires_grad=False."""
        model, _ = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        for name, param in model.named_parameters():
            assert not param.requires_grad, f"Model param {name} not frozen"

        trainable = [p for p in opt_model.parameters() if p.requires_grad]
        assert len(trainable) == 0, (
            f"Expected 0 trainable params before register, got {len(trainable)}"
        )


class TestHashDeterminism:
    def test_hash_determinism(self, model_and_tokenizer):
        """Same input_ids -> same hash; different -> different; 1D == 2D."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        ids_a = tokenizer.encode("Hello world", return_tensors="pt").squeeze(0)
        ids_b = tokenizer.encode("Goodbye world", return_tensors="pt").squeeze(0)

        h1 = opt_model._hash_input_ids(ids_a)
        h2 = opt_model._hash_input_ids(ids_a)
        assert h1 == h2

        h3 = opt_model._hash_input_ids(ids_b)
        assert h1 != h3

        h_1d = opt_model._hash_input_ids(ids_a)
        h_2d = opt_model._hash_input_ids(ids_a.unsqueeze(0))
        assert h_1d == h_2d


class TestRegisterParamsDiscovered:
    def test_register_params_discovered(self, model_and_tokenizer):
        """After register(), optimizer discovers exactly 2 new trainable params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        input_ids = tokenizer.encode(
            "The quick brown fox", return_tensors="pt"
        )
        opt_model.register(input_ids)

        trainable = [p for p in opt_model.parameters() if p.requires_grad]
        assert len(trainable) == 2, (
            f"Expected 2 trainable params after register, got {len(trainable)}"
        )
        for p in trainable:
            assert p.requires_grad


class TestFullAttentionLayers:
    def test_full_attention_layer_detection(self, model_and_tokenizer, model_info):
        """Full-attention layers match expected from config."""
        model, _ = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        assert opt_model.full_attention_layers == model_info["full_attention_layers"]
        assert len(opt_model.full_attention_layers) == model_info["num_fa_layers"]


class TestRegisterShape:
    def test_register_shape(self, model_and_tokenizer, model_info):
        """Key/value shapes: (num_fa_layers, B=1, H_kv, N=tokens, D)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        text = "Cache optimization test"
        input_ids = tokenizer.encode(text, return_tensors="pt")
        seq_len = input_ids.shape[1]

        opt_model.register(input_ids)

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")

        L = model_info["num_fa_layers"]
        H = model_info["num_kv_heads"]
        D = model_info["head_dim"]
        expected_shape = (L, 1, H, seq_len, D)
        assert key_param.shape == expected_shape, (
            f"Key shape {key_param.shape} != expected {expected_shape}"
        )
        assert val_param.shape == expected_shape, (
            f"Val shape {val_param.shape} != expected {expected_shape}"
        )


class TestRegisterNonzeroInit:
    def test_register_nonzero_init(self, model_and_tokenizer):
        """Registered params are warm-started (not all zeros)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        input_ids = tokenizer.encode("Warm start check", return_tensors="pt")
        opt_model.register(input_ids)

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")

        assert not torch.all(key_param == 0), "Key param is all zeros"
        assert not torch.all(val_param == 0), "Value param is all zeros"


class TestRegisterIdempotent:
    def test_register_idempotent(self, model_and_tokenizer):
        """Registering same input_ids twice returns same hash, no dup params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        input_ids = tokenizer.encode("Idempotent test", return_tensors="pt")

        h1 = opt_model.register(input_ids)
        h2 = opt_model.register(input_ids)
        assert h1 == h2

        trainable = [p for p in opt_model.parameters() if p.requires_grad]
        assert len(trainable) == 2, (
            f"Expected 2 trainable params (no duplicates), got {len(trainable)}"
        )


class TestPrefixHashIgnored:
    def test_prefix_does_not_affect_hash(self, model_and_tokenizer):
        """Hash is computed from input_ids only; prefix is excluded."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        segment = tokenizer.encode("target segment", return_tensors="pt")
        prefix = tokenizer.encode("some context prefix", return_tensors="pt")

        h_no_prefix = opt_model._hash_input_ids(segment)
        h_with_prefix = opt_model.register(segment, prefix=prefix)
        assert h_no_prefix == h_with_prefix


class TestPrefixShape:
    def test_prefix_shape_is_segment_only(self, model_and_tokenizer, model_info):
        """Registered param shape reflects segment length, not prefix+segment."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        prefix = tokenizer.encode("This is context.", return_tensors="pt")
        segment = tokenizer.encode("actual segment", return_tensors="pt")
        seg_len = segment.shape[1]

        opt_model.register(segment, prefix=prefix)

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")

        L = model_info["num_fa_layers"]
        H = model_info["num_kv_heads"]
        D = model_info["head_dim"]
        expected_shape = (L, 1, H, seg_len, D)
        assert key_param.shape == expected_shape, (
            f"Key shape {key_param.shape} != expected {expected_shape}"
        )
        assert val_param.shape == expected_shape, (
            f"Val shape {val_param.shape} != expected {expected_shape}"
        )


class TestPrefixAffectsValues:
    def test_prefix_changes_cached_values(self, model_and_tokenizer):
        """Registering with vs without prefix produces different param values."""
        model, tokenizer = model_and_tokenizer

        segment = tokenizer.encode("the target text", return_tensors="pt")
        prefix = tokenizer.encode(
            "You are a helpful assistant.", return_tensors="pt"
        )

        opt_no_prefix = CacheOptimizeModel(model)
        opt_no_prefix.register(segment)
        key_no = getattr(opt_no_prefix, "kv_key_0").detach().clone()
        val_no = getattr(opt_no_prefix, "kv_val_0").detach().clone()

        opt_with_prefix = CacheOptimizeModel(model)
        opt_with_prefix.register(segment, prefix=prefix)
        key_with = getattr(opt_with_prefix, "kv_key_0").detach().clone()
        val_with = getattr(opt_with_prefix, "kv_val_0").detach().clone()

        assert not torch.allclose(val_no, val_with, atol=1e-3), (
            "Values should differ when prefix provides different context"
        )
        assert not torch.allclose(key_no, key_with, atol=1e-3), (
            "Position-free keys should differ when prefix provides context"
        )


class TestRealisticToolRegistration:
    """Real scenario: register tool descriptions with system message as prefix."""

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
    def _split_template(tokenizer, system_msg, tools):
        """Split chat template at the first tool description string."""
        full_ids = tokenizer.apply_chat_template(
            [system_msg],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_str = tokenizer.decode(full_ids)

        first_desc = tools[0]["function"]["description"]
        desc_char_pos = full_str.find(first_desc)
        assert desc_char_pos > 0, (
            f"Could not find {first_desc!r} in rendered template"
        )

        cumlen = 0
        split_idx = 0
        for i in range(len(full_ids)):
            cumlen += len(tokenizer.decode(full_ids[i : i + 1]))
            if cumlen > desc_char_pos:
                split_idx = i
                break

        return (
            torch.tensor([full_ids[:split_idx]]),
            torch.tensor([full_ids[split_idx:]]),
        )

    def test_tool_register_shape(self, model_and_tokenizer, model_info):
        """Registered tool description shape is segment-only (no prefix)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        prefix, segment = self._split_template(
            tokenizer, self.SYSTEM_MSG, self.TOOLS
        )
        seg_len = segment.shape[1]

        opt_model.register(segment, prefix=prefix)

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")

        L = model_info["num_fa_layers"]
        H = model_info["num_kv_heads"]
        D = model_info["head_dim"]
        expected_shape = (L, 1, H, seg_len, D)
        assert key_param.shape == expected_shape, (
            f"Key shape {key_param.shape} != expected {expected_shape}"
        )
        assert val_param.shape == expected_shape, (
            f"Val shape {val_param.shape} != expected {expected_shape}"
        )

    def test_tool_register_nonzero(self, model_and_tokenizer):
        """Tool description KV params are warm-started (not zeros)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        prefix, segment = self._split_template(
            tokenizer, self.SYSTEM_MSG, self.TOOLS
        )
        opt_model.register(segment, prefix=prefix)

        key_param = getattr(opt_model, "kv_key_0")
        val_param = getattr(opt_model, "kv_val_0")
        assert not torch.all(key_param == 0), "Key param is all zeros"
        assert not torch.all(val_param == 0), "Value param is all zeros"

    def test_system_prefix_changes_tool_cache(self, model_and_tokenizer):
        """System message prefix alters tool description KV values."""
        model, tokenizer = model_and_tokenizer

        prefix, segment = self._split_template(
            tokenizer, self.SYSTEM_MSG, self.TOOLS
        )

        opt_with = CacheOptimizeModel(model)
        opt_with.register(segment, prefix=prefix)
        val_with = getattr(opt_with, "kv_val_0").detach().clone()

        opt_without = CacheOptimizeModel(model)
        opt_without.register(segment)
        val_without = getattr(opt_without, "kv_val_0").detach().clone()

        assert not torch.allclose(val_with, val_without, atol=1e-3), (
            "System prefix should change tool description cache values"
        )
