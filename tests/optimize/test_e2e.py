"""End-to-end tests for CacheOptimizeModel high-level tool API.

Tests the user-facing workflow: register_tools -> prepare_chat -> forward.
Low-level register/prepare tests are in test_prepare.py.
"""

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

USER_MSG = {
    "role": "user",
    "content": "What is machine learning?",
}


def _prefill_end(per_tool):
    """Compute prefill_end from per-tool metadata."""
    return max(e["token_end"] for e in per_tool)


class TestForwardDelegation:
    def test_forward_matches_model(self, model_and_tokenizer):
        """opt_model.forward(**kwargs) produces identical logits to model()."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        input_ids = tokenizer.encode(
            "The quick brown fox", return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            direct = model(input_ids=input_ids)
            wrapped = opt_model.forward(input_ids=input_ids)

        assert torch.equal(direct.logits, wrapped.logits), (
            "Wrapped forward should produce identical logits to direct model"
        )


class TestRegisterTools:
    def test_shape_and_meta(self, model_and_tokenizer, model_info):
        """register_tools() creates per-tool params with correct shape."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)

        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        assert len(per_tool) == 2
        assert per_tool[0]["tool_name"] == "search"
        assert per_tool[1]["tool_name"] == "get_document"

        L = model_info["num_fa_layers"]
        H = model_info["num_kv_heads"]
        D = model_info["head_dim"]

        for entry in per_tool:
            reg = opt_model.get_registry_entry(entry["hash"])
            key_param = getattr(opt_model, reg["key_param"])
            val_param = getattr(opt_model, reg["val_param"])
            seg_len = entry["token_end"] - entry["token_start"]
            expected_shape = (L, 1, H, seg_len, D)
            assert key_param.shape == expected_shape, (
                f"Key param shape {key_param.shape} != expected {expected_shape}"
            )
            assert val_param.shape == expected_shape, (
                f"Val param shape {val_param.shape} != expected {expected_shape}"
            )
            assert key_param.requires_grad
            assert val_param.requires_grad


class TestPrepareChatStructure:
    def test_output_shapes(self, model_and_tokenizer):
        """prepare_chat() output has correct input_ids shape and cache length."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        result = opt_model.prepare_chat(
            tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS
        )

        seg_end = _prefill_end(per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_len = len(full_ids)

        expected_remaining = full_len - seg_end
        assert result["input_ids"].shape == (1, expected_remaining), (
            f"Expected (1, {expected_remaining}), got {result['input_ids'].shape}"
        )

        cache = result["past_key_values"]
        fa_idx = opt_model.full_attention_layers[0]
        cache_len = cache.get_seq_length(fa_idx)
        assert cache_len == seg_end, (
            f"Cache seq_length {cache_len} != seg_end {seg_end}"
        )


class TestPrepareChatForward:
    def test_logits_match_direct(self, model_and_tokenizer):
        """Before optimization, register_tools->prepare_chat->forward logits match direct."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        seg_end = _prefill_end(per_tool)

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
        direct_logits = direct_out.logits[:, seg_end:, :].float().cpu()

        # 2. prepare_chat + forward
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


class TestPrepareChatGenerate:
    def test_generation_matches_direct(self, model_and_tokenizer):
        """Greedy-generate 20 tokens via prepare_chat matches direct generate."""
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

        # 2. prepare_chat + generate
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


class TestPrepareVsPrepareChatEquivalence:
    """Verify that prepare() with register_tools() indices matches prepare_chat().

    This is the pattern used in cache_optimize_training.py: call register_tools()
    once, store the indices, then use prepare() directly in the training loop
    instead of re-computing boundaries via prepare_chat() every step.
    """

    def test_logits_match(self, model_and_tokenizer):
        """prepare() with manual indices produces identical logits to prepare_chat()."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids_t = torch.tensor([full_ids])

        kv_cache_indices = [(e["token_start"], e["token_end"]) for e in per_tool]

        with torch.no_grad():
            result_chat = opt_model.prepare_chat(
                tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS
            )
            result_manual = opt_model.prepare(
                kv_cache_indices=kv_cache_indices,
                input_ids=full_ids_t,
            )

        # Same input_ids
        assert torch.equal(result_chat["input_ids"], result_manual["input_ids"]), (
            f"input_ids differ: chat {result_chat['input_ids'].shape} vs "
            f"manual {result_manual['input_ids'].shape}"
        )

        # Same cache content
        chat_cache = result_chat["past_key_values"]
        manual_cache = result_manual["past_key_values"]
        for i in range(len(chat_cache.layers)):
            assert torch.equal(
                chat_cache.layers[i].keys, manual_cache.layers[i].keys
            ), f"Layer {i} keys differ"
            assert torch.equal(
                chat_cache.layers[i].values, manual_cache.layers[i].values
            ), f"Layer {i} values differ"

    def test_logits_match_with_labels(self, model_and_tokenizer):
        """prepare() with labels produces identical loss to prepare_chat()."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        seg_end = _prefill_end(per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids_t = torch.tensor([full_ids])
        labels = torch.tensor([full_ids])
        labels[0, :seg_end] = -100

        kv_cache_indices = [(e["token_start"], e["token_end"]) for e in per_tool]

        with torch.no_grad():
            result_chat = opt_model.prepare_chat(
                tokenizer, [SYSTEM_MSG, USER_MSG], TOOLS, labels=labels.clone()
            )
            out_chat = opt_model.forward(**result_chat)

            result_manual = opt_model.prepare(
                kv_cache_indices=kv_cache_indices,
                input_ids=full_ids_t,
                labels=labels.clone(),
            )
            out_manual = opt_model.forward(**result_manual)

        assert torch.equal(result_chat["labels"], result_manual["labels"]), (
            "Labels differ between prepare_chat and prepare"
        )
        assert out_chat.loss.item() == pytest.approx(out_manual.loss.item()), (
            f"Loss differs: chat={out_chat.loss.item():.6f} vs "
            f"manual={out_manual.loss.item():.6f}"
        )


class TestToolPipeline:
    def test_backward_step_updates_params(self, model_and_tokenizer):
        """Full pipeline: register_tools->prepare_chat->forward->backward->step updates KV params."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        seg_end = _prefill_end(per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = torch.tensor([full_ids])
        labels[0, :seg_end] = -100

        # Snapshot all per-tool params before step
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
        loss = output.loss

        assert loss is not None, "Loss is None -- labels may not be set correctly"

        loss.backward()
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


class TestToolTrainingLoop:
    def test_loss_changes_over_steps(self, model_and_tokenizer):
        """3 training steps with zero_grad: all losses finite, loss changes."""
        model, tokenizer = model_and_tokenizer
        opt_model = CacheOptimizeModel(model)
        per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)

        seg_end = _prefill_end(per_tool)

        full_ids = tokenizer.apply_chat_template(
            [SYSTEM_MSG, USER_MSG],
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = torch.tensor([full_ids])
        labels[0, :seg_end] = -100

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
