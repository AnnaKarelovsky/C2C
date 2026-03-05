"""Tests for CacheOptimizeModel.save_pretrained / load_pretrained."""

import json

import pytest
import torch

from rosetta.optimize.wrapper import CacheOptimizeModel

SYSTEM_MSG = {
    "role": "system",
    "content": "You are a helpful research assistant.",
}

TOOLS_A = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information using a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    }
]

TOOLS_B = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


class TestSaveLoadRoundtrip:
    def test_single_tool(self, model_and_tokenizer, tmp_path):
        """Metadata and params survive a save/load cycle."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        per_tool = original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        # Metadata
        assert loaded._param_counter == original._param_counter
        assert loaded.registered_tools == original.registered_tools

        # Parameter values (resolve via json_hash)
        for entry in per_tool:
            json_hash = entry["hash"]
            oe = original.get_registry_entry(json_hash)
            le = loaded.get_registry_entry(json_hash)
            assert torch.equal(
                getattr(original, oe["key_param"]),
                getattr(loaded, le["key_param"]),
            )
            assert torch.equal(
                getattr(original, oe["val_param"]),
                getattr(loaded, le["val_param"]),
            )

    def test_multiple_tools(self, model_and_tokenizer, tmp_path):
        """Two tool sets both survive a save/load cycle."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        per_tool_a = original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        per_tool_b = original.register_tools(tokenizer, TOOLS_B, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        assert loaded.registered_tools == original.registered_tools

        trainable_orig = [p for p in original.parameters() if p.requires_grad]
        trainable_load = [p for p in loaded.parameters() if p.requires_grad]
        assert len(trainable_load) == len(trainable_orig) == 4

        # Parameter values (resolve via json_hash)
        for entry in per_tool_a + per_tool_b:
            json_hash = entry["hash"]
            oe = original.get_registry_entry(json_hash)
            le = loaded.get_registry_entry(json_hash)
            assert torch.equal(
                getattr(original, oe["key_param"]),
                getattr(loaded, le["key_param"]),
            )
            assert torch.equal(
                getattr(original, oe["val_param"]),
                getattr(loaded, le["val_param"]),
            )


class TestModifiedParams:
    def test_modified_params_preserved(self, model_and_tokenizer, tmp_path):
        """Simulated training: mutated params survive save/load."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        per_tool = original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)

        # Mutate params (simulate training)
        with torch.no_grad():
            for p in original.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.1)

        json_hash = per_tool[0]["hash"]
        entry = original.get_registry_entry(json_hash)
        modified_key = getattr(original, entry["key_param"]).detach().clone()
        modified_val = getattr(original, entry["val_param"]).detach().clone()

        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        le = loaded.get_registry_entry(json_hash)
        assert torch.equal(getattr(loaded, le["key_param"]), modified_key)
        assert torch.equal(getattr(loaded, le["val_param"]), modified_val)

    def test_unmodified_params_match(self, model_and_tokenizer, tmp_path):
        """Without modification, save/load produces identical params."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        per_tool = original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)

        json_hash = per_tool[0]["hash"]
        entry = original.get_registry_entry(json_hash)
        orig_key = getattr(original, entry["key_param"]).detach().clone()

        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        le = loaded.get_registry_entry(json_hash)
        assert torch.equal(getattr(loaded, le["key_param"]), orig_key)


class TestLoadedModelWorks:
    def test_prepare_chat_after_load(self, model_and_tokenizer, tmp_path):
        """Loaded model runs prepare_chat() without errors."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        messages = [
            SYSTEM_MSG,
            {"role": "user", "content": "What is the capital of France?"},
        ]

        result = loaded.prepare_chat(tokenizer, messages, TOOLS_A)
        assert "input_ids" in result
        assert "past_key_values" in result
        assert result["input_ids"].shape[0] == 1

    def test_prepare_chat_multiple_tools_after_load(
        self, model_and_tokenizer, tmp_path
    ):
        """Both tool sets work after loading."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        original.register_tools(tokenizer, TOOLS_B, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        messages = [
            SYSTEM_MSG,
            {"role": "user", "content": "Hello"},
        ]

        result_a = loaded.prepare_chat(tokenizer, messages, TOOLS_A)
        result_b = loaded.prepare_chat(tokenizer, messages, TOOLS_B)
        assert "past_key_values" in result_a
        assert "past_key_values" in result_b

    def test_output_matches_original(self, model_and_tokenizer, tmp_path):
        """Loaded model produces same cache as original."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)

        messages = [
            SYSTEM_MSG,
            {"role": "user", "content": "What is 2+2?"},
        ]

        orig_result = original.prepare_chat(tokenizer, messages, TOOLS_A)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        load_result = loaded.prepare_chat(tokenizer, messages, TOOLS_A)

        assert torch.equal(orig_result["input_ids"], load_result["input_ids"])

        orig_cache = orig_result["past_key_values"]
        load_cache = load_result["past_key_values"]
        for i in range(len(orig_cache.layers)):
            assert torch.allclose(
                orig_cache.layers[i].keys,
                load_cache.layers[i].keys,
                atol=1e-5,
            ), f"Layer {i} keys differ"
            assert torch.allclose(
                orig_cache.layers[i].values,
                load_cache.layers[i].values,
                atol=1e-5,
            ), f"Layer {i} values differ"


class TestKvParameters:
    def test_empty_before_register(self, model_and_tokenizer):
        """No KV params before any registration."""
        model, _ = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        assert list(opt_model.kv_parameters()) == []

    def test_yields_registered_params(self, model_and_tokenizer):
        """kv_parameters() yields exactly the registered key/val pairs."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)

        kv = list(opt_model.kv_parameters())
        assert len(kv) == 2
        assert all(p.requires_grad for p in kv)

    def test_multiple_registrations(self, model_and_tokenizer):
        """Two tool sets → 4 KV params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        opt_model.register_tools(tokenizer, TOOLS_B, SYSTEM_MSG)

        kv = list(opt_model.kv_parameters())
        assert len(kv) == 4

    def test_after_load(self, model_and_tokenizer, tmp_path):
        """kv_parameters() works after load_pretrained."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        original.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        assert len(list(loaded.kv_parameters())) == 2


class TestSavedFiles:
    def test_files_created(self, model_and_tokenizer, tmp_path):
        """save_pretrained creates kv_params.pt and kv_config.json."""
        model, tokenizer = model_and_tokenizer

        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        opt_model.save_pretrained(str(tmp_path))

        assert (tmp_path / "kv_params.pt").exists()
        assert (tmp_path / "kv_config.json").exists()

    def test_config_is_valid_json(self, model_and_tokenizer, tmp_path):
        """kv_config.json is valid and contains expected keys."""
        model, tokenizer = model_and_tokenizer

        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS_A, SYSTEM_MSG)
        opt_model.save_pretrained(str(tmp_path))

        with open(tmp_path / "kv_config.json") as f:
            config = json.load(f)

        assert "registry" in config
        assert "param_counter" in config
        assert config["param_counter"] == 1
        assert len(config["registry"]) == 1
