"""Tests for per-tool KV registration.

Tests register_tools() which registers each tool as an independent
learnable KV cache segment, enabling composable tool combinations at runtime.
"""

import json

import pytest
import torch

from rosetta.optimize.wrapper import CacheOptimizeModel

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
            "description": "Search for information using a query string.",
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

# Tool with a description that mentions another tool's name
TOOLS_WITH_COLLISION = [
    {
        "type": "function",
        "function": {
            "name": "search_advanced",
            "description": "Advanced search — use with search to find information.",
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
            "name": "search",
            "description": "Search for information using a query string.",
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
]

USER_MSG = {
    "role": "user",
    "content": "What is machine learning?",
}


# ----------------------------------------------------------------
# Boundary detection
# ----------------------------------------------------------------


class TestCharToTokenBoundaries:
    def test_covers_substring(self, model_and_tokenizer):
        """Decoded token slice contains the target substring."""
        _, tokenizer = model_and_tokenizer
        text = "Hello, world! This is a test sentence."
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Target: "world"
        char_start = text.find("world")
        char_end = char_start + len("world")

        tok_start, tok_end = CacheOptimizeModel._char_to_token_boundaries(
            tokenizer, token_ids, char_start, char_end
        )

        decoded_slice = tokenizer.decode(token_ids[tok_start:tok_end])
        assert "world" in decoded_slice, (
            f"Decoded slice {decoded_slice!r} does not contain 'world'"
        )
        assert tok_start < tok_end


class TestFindToolCharBoundaries:
    def test_two_tool_boundaries(self, model_and_tokenizer):
        """Both tools found, non-overlapping, in input order."""
        _, tokenizer = model_and_tokenizer
        msgs = [SYSTEM_MSG]
        ids = tokenizer.apply_chat_template(
            msgs, tools=TOOLS, tokenize=True, add_generation_prompt=False
        )
        text = tokenizer.decode(ids)

        bounds = CacheOptimizeModel._find_tool_char_boundaries(text, TOOLS)
        assert len(bounds) == 2

        (s0, e0), (s1, e1) = bounds
        assert s0 < e0
        assert s1 < e1
        # Non-overlapping
        assert e0 <= s1, (
            f"Tool boundaries overlap: first ends at {e0}, second starts at {s1}"
        )

    def test_boundaries_contain_tool_name(self, model_and_tokenizer):
        """Each boundary's decoded text contains its tool's name."""
        _, tokenizer = model_and_tokenizer
        msgs = [SYSTEM_MSG]
        ids = tokenizer.apply_chat_template(
            msgs, tools=TOOLS, tokenize=True, add_generation_prompt=False
        )
        text = tokenizer.decode(ids)

        bounds = CacheOptimizeModel._find_tool_char_boundaries(text, TOOLS)
        for i, (cs, ce) in enumerate(bounds):
            tool_name = TOOLS[i]["function"]["name"]
            snippet = text[cs:ce]
            assert tool_name in snippet, (
                f"Tool '{tool_name}' not found in boundary text: {snippet!r}"
            )

    def test_tool_name_in_description_no_collision(self, model_and_tokenizer):
        """Tool A describes 'use with search', tool B is 'search' — boundaries correct."""
        _, tokenizer = model_and_tokenizer
        msgs = [SYSTEM_MSG]
        ids = tokenizer.apply_chat_template(
            msgs,
            tools=TOOLS_WITH_COLLISION,
            tokenize=True,
            add_generation_prompt=False,
        )
        text = tokenizer.decode(ids)

        bounds = CacheOptimizeModel._find_tool_char_boundaries(
            text, TOOLS_WITH_COLLISION
        )
        assert len(bounds) == 2

        (s0, e0), (s1, e1) = bounds
        # Non-overlapping
        assert e0 <= s1

        # First boundary contains "search_advanced", second contains "search"
        first_text = text[s0:e0]
        second_text = text[s1:e1]
        assert "search_advanced" in first_text
        assert "search" in second_text


# ----------------------------------------------------------------
# Registration
# ----------------------------------------------------------------


class TestRegistrationStructure:
    def test_two_tools_four_params(self, model_and_tokenizer):
        """2 tools → 4 trainable params (requires_grad=True)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        trainable = [p for p in opt_model.parameters() if p.requires_grad]
        assert len(trainable) == 4, (
            f"Expected 4 trainable params (2 tools × key+val), got {len(trainable)}"
        )

    def test_meta_structure(self, model_and_tokenizer):
        """register_tools() returns per-tool list with required keys."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        assert len(per_tool) == 2
        assert per_tool[0]["tool_name"] == "search"
        assert per_tool[1]["tool_name"] == "get_document"

        required_keys = {"tool_name", "token_start", "token_end", "hash"}
        for entry in per_tool:
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys: {required_keys - entry.keys()}"
            )

    def test_param_shapes(self, model_and_tokenizer, model_info):
        """Each tool's KV shape is (L_fa, 1, H, seg_len_i, D)."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        L = model_info["num_fa_layers"]
        H = model_info["num_kv_heads"]
        D = model_info["head_dim"]

        for entry in per_tool:
            reg = opt_model.get_registry_entry(entry["hash"])
            key_param = getattr(opt_model, reg["key_param"])
            val_param = getattr(opt_model, reg["val_param"])

            seg_len = entry["token_end"] - entry["token_start"]
            expected = (L, 1, H, seg_len, D)
            assert key_param.shape == expected, (
                f"Key param shape {key_param.shape} != expected {expected} "
                f"for tool '{entry['tool_name']}'"
            )
            assert val_param.shape == expected, (
                f"Val param shape {val_param.shape} != expected {expected} "
                f"for tool '{entry['tool_name']}'"
            )

    def test_idempotent(self, model_and_tokenizer):
        """Calling twice → same results, no new params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        per_tool_1 = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)
        n_params_1 = len([p for p in opt_model.parameters() if p.requires_grad])

        per_tool_2 = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)
        n_params_2 = len([p for p in opt_model.parameters() if p.requires_grad])

        # Same hashes returned
        hashes_1 = [e["hash"] for e in per_tool_1]
        hashes_2 = [e["hash"] for e in per_tool_2]
        assert hashes_1 == hashes_2
        assert n_params_1 == n_params_2

    def test_segments_non_overlapping(self, model_and_tokenizer):
        """Token ranges in the all-tools template don't overlap."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        entries = sorted(per_tool, key=lambda e: e["token_start"])
        for i in range(len(entries) - 1):
            assert entries[i]["token_end"] <= entries[i + 1]["token_start"], (
                f"Segments overlap: {entries[i]['tool_name']} ends at "
                f"{entries[i]['token_end']}, {entries[i+1]['tool_name']} starts at "
                f"{entries[i+1]['token_start']}"
            )


# ----------------------------------------------------------------
# Forward + generation correctness
# ----------------------------------------------------------------


class TestPerToolForward:
    def test_logits_match_direct(self, model_and_tokenizer):
        """Per-tool prepare_chat + forward → top-1 match + max_diff < 8.0 vs direct."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        # prefill_end = max token_end across all per-tool entries
        prefill_end = max(e["token_end"] for e in per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids_t = torch.tensor([full_ids])

        # 1. Direct forward (reference)
        with torch.no_grad():
            direct_out = model(full_ids_t.to(model.device))
        direct_logits = direct_out.logits[:, prefill_end:, :].float().cpu()

        # 2. Per-tool prepare_chat + forward
        with torch.no_grad():
            result = opt_model.prepare_chat(
                tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS
            )
            prepared_out = opt_model.forward(**result)
        prepared_logits = prepared_out.logits.float().cpu()

        assert direct_logits.shape == prepared_logits.shape, (
            f"Shape mismatch: direct {direct_logits.shape} vs "
            f"prepared {prepared_logits.shape}"
        )

        direct_top1 = direct_logits.argmax(dim=-1)
        prepared_top1 = prepared_logits.argmax(dim=-1)
        match_rate = (direct_top1 == prepared_top1).float().mean().item()
        assert match_rate >= 0.9, (
            f"Top-1 match rate too low: {match_rate:.2%}\n"
            f"  direct:   {direct_top1}\n"
            f"  prepared: {prepared_top1}"
        )

        max_diff = (direct_logits - prepared_logits).abs().max().item()
        assert max_diff < 8.0, (
            f"Logits differ too much: max_diff={max_diff:.4f}"
        )

    def test_generation_matches_direct(self, model_and_tokenizer):
        """Per-tool prepare_chat + generate 20 tokens → identical to direct generate."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids_t = torch.tensor([full_ids])
        full_len = len(full_ids)
        max_new = 20

        # 1. Direct generation (reference)
        with torch.no_grad():
            direct_gen = model.generate(
                full_ids_t.to(model.device),
                max_new_tokens=max_new,
                do_sample=False,
            )
        direct_tokens = direct_gen[0, full_len:]

        # 2. Per-tool prepare_chat + generate
        with torch.no_grad():
            result = opt_model.prepare_chat(
                tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS
            )
            cache = result["past_key_values"]

            prepared_gen = model.generate(
                full_ids_t.to(model.device),
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


# ----------------------------------------------------------------
# Gradient flow + training
# ----------------------------------------------------------------


class TestPerToolGradients:
    def test_all_tool_params_get_gradients(self, model_and_tokenizer):
        """After backward, all per-tool key/val params have non-None, non-zero grad."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        prefill_end = max(e["token_end"] for e in per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = torch.tensor([full_ids])
        labels[0, :prefill_end] = -100

        result = opt_model.prepare_chat(
            tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS, labels=labels
        )
        output = opt_model.forward(**result)
        loss = output.loss
        assert loss is not None, "Loss is None -- labels may not be set correctly"
        loss.backward()

        for entry in per_tool:
            reg = opt_model.get_registry_entry(entry["hash"])
            key_param = getattr(opt_model, reg["key_param"])
            val_param = getattr(opt_model, reg["val_param"])

            assert key_param.grad is not None, (
                f"Key param for '{entry['tool_name']}' has no gradient"
            )
            assert val_param.grad is not None, (
                f"Val param for '{entry['tool_name']}' has no gradient"
            )
            assert torch.any(key_param.grad != 0), (
                f"Key gradient for '{entry['tool_name']}' is all zeros"
            )
            assert torch.any(val_param.grad != 0), (
                f"Val gradient for '{entry['tool_name']}' is all zeros"
            )

    def test_optimizer_step_updates_params(self, model_and_tokenizer):
        """After step, all per-tool params have changed."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        prefill_end = max(e["token_end"] for e in per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = torch.tensor([full_ids])
        labels[0, :prefill_end] = -100

        # Store original param values
        originals = {}
        for entry in per_tool:
            reg = opt_model.get_registry_entry(entry["hash"])
            originals[entry["tool_name"]] = {
                "key": getattr(opt_model, reg["key_param"]).detach().clone(),
                "val": getattr(opt_model, reg["val_param"]).detach().clone(),
            }

        optimizer = torch.optim.Adam(
            [p for p in opt_model.parameters() if p.requires_grad], lr=1e-3
        )

        result = opt_model.prepare_chat(
            tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS, labels=labels
        )
        output = opt_model.forward(**result)
        output.loss.backward()
        optimizer.step()

        for entry in per_tool:
            reg = opt_model.get_registry_entry(entry["hash"])
            key_param = getattr(opt_model, reg["key_param"])
            val_param = getattr(opt_model, reg["val_param"])
            name = entry["tool_name"]

            assert not torch.equal(key_param.data, originals[name]["key"]), (
                f"Key param for '{name}' did not change after optimizer step"
            )
            assert not torch.equal(val_param.data, originals[name]["val"]), (
                f"Val param for '{name}' did not change after optimizer step"
            )

    def test_loss_changes_over_steps(self, model_and_tokenizer):
        """3 training steps: all finite, loss changes."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        prefill_end = max(e["token_end"] for e in per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = torch.tensor([full_ids])
        labels[0, :prefill_end] = -100

        optimizer = torch.optim.Adam(
            [p for p in opt_model.parameters() if p.requires_grad], lr=1e-3
        )

        losses = []
        for step in range(3):
            optimizer.zero_grad()
            result = opt_model.prepare_chat(
                tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS, labels=labels
            )
            output = opt_model.forward(**result)
            loss = output.loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        for i, l in enumerate(losses):
            assert torch.isfinite(torch.tensor(l)), (
                f"Step {i} loss is not finite: {l}"
            )

        assert losses[0] != losses[1] or losses[1] != losses[2], (
            f"Loss did not change across 3 steps: {losses}"
        )


# ----------------------------------------------------------------
# Save / Load
# ----------------------------------------------------------------


class TestPerToolSaveLoad:
    def test_save_load_roundtrip(self, model_and_tokenizer, tmp_path):
        """Per-tool registry and params survive save/load."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        per_tool = original.register_tools(tokenizer, TOOLS, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        # Tool metadata round-trips
        assert loaded.registered_tools == original.registered_tools
        assert loaded._param_counter == original._param_counter

        # Per-tool structure preserved
        assert len(loaded.registered_tools) == 2

        # Parameter values match (resolve via json_hash from per_tool info)
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

    def test_prepare_chat_after_load(self, model_and_tokenizer, tmp_path):
        """prepare_chat() works after loading per-tool checkpoint."""
        model, tokenizer = model_and_tokenizer

        original = CacheOptimizeModel(model)
        original.register_tools(tokenizer, TOOLS, SYSTEM_MSG)
        original.save_pretrained(str(tmp_path))

        loaded = CacheOptimizeModel(model)
        loaded.load_pretrained(str(tmp_path))

        messages = [SYSTEM_MSG, USER_MSG]
        result = loaded.prepare_chat(tokenizer, messages, TOOLS)

        assert "input_ids" in result
        assert "past_key_values" in result
        assert result["input_ids"].shape[0] == 1
