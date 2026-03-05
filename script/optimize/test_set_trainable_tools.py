"""Test set_trainable_tools: verify freeze/unfreeze, gradient flow, and optimizer updates.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n c2c --no-capture-output python script/optimize/test_set_trainable_tools.py
"""

from __future__ import annotations

import json
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.wrapper import CacheOptimizeModel

MODEL = "Qwen/Qwen3-1.7B"

# Two dummy tools for testing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight between two airports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure airport"},
                    "destination": {"type": "string", "description": "Arrival airport"},
                    "date": {"type": "string", "description": "Flight date"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]

SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


def _get_tool_params(opt_model, tool_name):
    """Return (key_param, val_param) tensors for a tool by name."""
    for entry in opt_model._registry.values():
        if entry.get("tool_name") == tool_name:
            k = getattr(opt_model, entry["key_param"])
            v = getattr(opt_model, entry["val_param"])
            return k, v
    raise KeyError(f"Tool '{tool_name}' not found in registry")


def _snapshot(opt_model):
    """Return a dict of {tool_name: (key_data_clone, val_data_clone)}."""
    snap = {}
    for entry in opt_model._registry.values():
        name = entry.get("tool_name")
        if name:
            k = getattr(opt_model, entry["key_param"]).data.clone()
            v = getattr(opt_model, entry["val_param"]).data.clone()
            snap[name] = (k, v)
    return snap


def _params_changed(snap_before, snap_after, tool_name):
    """Check if a tool's params changed between snapshots."""
    k_before, v_before = snap_before[tool_name]
    k_after, v_after = snap_after[tool_name]
    k_diff = (k_before - k_after).abs().max().item()
    v_diff = (v_before - v_after).abs().max().item()
    return k_diff > 0 or v_diff > 0, k_diff, v_diff


def test_requires_grad_toggle(opt_model):
    """Test 1: set_trainable_tools correctly toggles requires_grad."""
    print("\n=== Test 1: requires_grad toggle ===")

    # Freeze all except get_weather
    opt_model.set_trainable_tools(["get_weather"])
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    assert k_w.requires_grad and v_w.requires_grad, "get_weather should be trainable"
    assert not k_f.requires_grad and not v_f.requires_grad, "book_flight should be frozen"
    print("  [PASS] get_weather=trainable, book_flight=frozen")

    # Freeze all except book_flight
    opt_model.set_trainable_tools(["book_flight"])
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    assert not k_w.requires_grad and not v_w.requires_grad, "get_weather should be frozen"
    assert k_f.requires_grad and v_f.requires_grad, "book_flight should be trainable"
    print("  [PASS] get_weather=frozen, book_flight=trainable")

    # Unfreeze all (None)
    opt_model.set_trainable_tools(None)
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    assert k_w.requires_grad and v_w.requires_grad, "get_weather should be trainable"
    assert k_f.requires_grad and v_f.requires_grad, "book_flight should be trainable"
    print("  [PASS] None -> all trainable")

    # Empty list -> all frozen
    opt_model.set_trainable_tools([])
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    assert not k_w.requires_grad and not v_w.requires_grad, "get_weather should be frozen"
    assert not k_f.requires_grad and not v_f.requires_grad, "book_flight should be frozen"
    print("  [PASS] [] -> all frozen")

    # Reset
    opt_model.set_trainable_tools(None)


def test_gradient_flow(opt_model, tokenizer):
    """Test 2: gradients only flow to trainable tools."""
    print("\n=== Test 2: gradient flow ===")

    messages = [
        SYSTEM_MSG,
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "Let me check the weather for you."},
    ]
    tmpl_kwargs = {"enable_thinking": False}

    # --- Only get_weather trainable ---
    opt_model.set_trainable_tools(["get_weather"])

    # Zero all grads
    for p in opt_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    prepared = opt_model.prepare_chat(
        tokenizer, messages, TOOLS, for_generate=False,
        template_kwargs=tmpl_kwargs,
    )
    # Build labels: supervise last 10 tokens
    input_ids = prepared["input_ids"]
    labels = torch.full_like(input_ids, -100)
    labels[:, -10:] = input_ids[:, -10:]
    prepared["labels"] = labels

    output = opt_model.forward(**prepared)
    n_tokens = (labels != -100).sum()
    loss = output.loss
    loss.backward()

    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")

    assert k_w.grad is not None and k_w.grad.abs().max() > 0, "get_weather key should have grad"
    assert v_w.grad is not None and v_w.grad.abs().max() > 0, "get_weather val should have grad"
    assert k_f.grad is None or k_f.grad.abs().max() == 0, "book_flight key should have no grad"
    assert v_f.grad is None or v_f.grad.abs().max() == 0, "book_flight val should have no grad"
    print(f"  [PASS] get_weather grad: key={k_w.grad.abs().max():.6f}, val={v_w.grad.abs().max():.6f}")
    print(f"  [PASS] book_flight grad: key={k_f.grad is None or k_f.grad.abs().max() == 0}, val={v_f.grad is None or v_f.grad.abs().max() == 0}")

    # Reset
    opt_model.set_trainable_tools(None)
    for p in opt_model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def test_optimizer_update(opt_model, tokenizer):
    """Test 3: optimizer.step() only updates trainable tools."""
    print("\n=== Test 3: optimizer update ===")

    all_kv_params = list(opt_model.kv_parameters())
    optimizer = torch.optim.AdamW(all_kv_params, lr=1e-2)
    tmpl_kwargs = {"enable_thinking": False}

    messages = [
        SYSTEM_MSG,
        {"role": "user", "content": "Book a flight from NYC to LAX on Jan 1."},
        {"role": "assistant", "content": "I'll book that flight for you."},
    ]

    # Freeze get_weather, train only book_flight
    opt_model.set_trainable_tools(["book_flight"])
    optimizer.zero_grad()

    snap_before = _snapshot(opt_model)

    prepared = opt_model.prepare_chat(
        tokenizer, messages, TOOLS, for_generate=False,
        template_kwargs=tmpl_kwargs,
    )
    input_ids = prepared["input_ids"]
    labels = torch.full_like(input_ids, -100)
    labels[:, -10:] = input_ids[:, -10:]
    prepared["labels"] = labels

    output = opt_model.forward(**prepared)
    output.loss.backward()
    optimizer.step()

    snap_after = _snapshot(opt_model)

    weather_changed, w_k_diff, w_v_diff = _params_changed(snap_before, snap_after, "get_weather")
    flight_changed, f_k_diff, f_v_diff = _params_changed(snap_before, snap_after, "book_flight")

    assert not weather_changed, f"get_weather should NOT change (k_diff={w_k_diff}, v_diff={w_v_diff})"
    assert flight_changed, f"book_flight SHOULD change (k_diff={f_k_diff}, v_diff={f_v_diff})"
    print(f"  [PASS] get_weather unchanged (k_diff={w_k_diff:.8f}, v_diff={w_v_diff:.8f})")
    print(f"  [PASS] book_flight updated  (k_diff={f_k_diff:.8f}, v_diff={f_v_diff:.8f})")

    # Reset
    opt_model.set_trainable_tools(None)


def test_grad_accumulation(opt_model, tokenizer):
    """Test 4: gradient accumulation across batches with different active tools."""
    print("\n=== Test 4: gradient accumulation ===")

    all_kv_params = list(opt_model.kv_parameters())
    optimizer = torch.optim.AdamW(all_kv_params, lr=1e-2)
    optimizer.zero_grad()
    tmpl_kwargs = {"enable_thinking": False}

    snap_before = _snapshot(opt_model)

    # Micro-batch 1: only get_weather trainable
    opt_model.set_trainable_tools(["get_weather"])
    messages1 = [
        SYSTEM_MSG,
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "Checking weather now."},
    ]
    prepared1 = opt_model.prepare_chat(
        tokenizer, messages1, TOOLS, for_generate=False,
        template_kwargs=tmpl_kwargs,
    )
    ids1 = prepared1["input_ids"]
    labels1 = torch.full_like(ids1, -100)
    labels1[:, -5:] = ids1[:, -5:]
    prepared1["labels"] = labels1
    out1 = opt_model.forward(**prepared1)
    out1.loss.backward()

    # Check: get_weather has grad, book_flight does not
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    weather_grad_after_mb1 = k_w.grad.clone()
    assert k_w.grad is not None and k_w.grad.abs().max() > 0
    assert k_f.grad is None or k_f.grad.abs().max() == 0
    print("  [PASS] After micro-batch 1: get_weather has grad, book_flight has none")

    # Micro-batch 2: only book_flight trainable
    opt_model.set_trainable_tools(["book_flight"])
    messages2 = [
        SYSTEM_MSG,
        {"role": "user", "content": "Book me a flight."},
        {"role": "assistant", "content": "Booking your flight."},
    ]
    prepared2 = opt_model.prepare_chat(
        tokenizer, messages2, TOOLS, for_generate=False,
        template_kwargs=tmpl_kwargs,
    )
    ids2 = prepared2["input_ids"]
    labels2 = torch.full_like(ids2, -100)
    labels2[:, -5:] = ids2[:, -5:]
    prepared2["labels"] = labels2
    out2 = opt_model.forward(**prepared2)
    out2.loss.backward()

    # Check: get_weather's grad from mb1 is preserved, book_flight now has grad too
    k_w, v_w = _get_tool_params(opt_model, "get_weather")
    k_f, v_f = _get_tool_params(opt_model, "book_flight")
    assert torch.equal(k_w.grad, weather_grad_after_mb1), \
        "get_weather grad should be preserved from micro-batch 1"
    assert k_f.grad is not None and k_f.grad.abs().max() > 0, \
        "book_flight should now have grad from micro-batch 2"
    print("  [PASS] After micro-batch 2: get_weather grad preserved, book_flight has new grad")

    # Optimizer step: both tools should update
    optimizer.step()
    snap_after = _snapshot(opt_model)

    weather_changed, w_k_diff, w_v_diff = _params_changed(snap_before, snap_after, "get_weather")
    flight_changed, f_k_diff, f_v_diff = _params_changed(snap_before, snap_after, "book_flight")

    assert weather_changed, "get_weather should update (grad from mb1)"
    assert flight_changed, "book_flight should update (grad from mb2)"
    print(f"  [PASS] Both updated: get_weather (k={w_k_diff:.8f}), book_flight (k={f_k_diff:.8f})")

    # Reset
    opt_model.set_trainable_tools(None)


def test_no_trainable_skipped(opt_model, tokenizer):
    """Test 5: with all tools frozen, no grad flows and params don't change."""
    print("\n=== Test 5: all frozen -> no update ===")

    all_kv_params = list(opt_model.kv_parameters())
    optimizer = torch.optim.AdamW(all_kv_params, lr=1e-2)
    optimizer.zero_grad()
    tmpl_kwargs = {"enable_thinking": False}

    snap_before = _snapshot(opt_model)

    opt_model.set_trainable_tools([])
    messages = [
        SYSTEM_MSG,
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    prepared = opt_model.prepare_chat(
        tokenizer, messages, TOOLS, for_generate=False,
        template_kwargs=tmpl_kwargs,
    )
    ids = prepared["input_ids"]
    labels = torch.full_like(ids, -100)
    labels[:, -3:] = ids[:, -3:]
    prepared["labels"] = labels
    out = opt_model.forward(**prepared)

    # When ALL tools are frozen, no tensors in the graph require grad,
    # so backward() raises RuntimeError. This is correct behavior —
    # verify it raises and that no params changed.
    try:
        out.loss.backward()
        assert False, "backward() should raise when no tensors require grad"
    except RuntimeError:
        pass  # expected
    print("  [PASS] backward() correctly raises when all tools frozen")

    # Verify no grads accumulated and optimizer step is a no-op
    for name in ["get_weather", "book_flight"]:
        k, v = _get_tool_params(opt_model, name)
        assert k.grad is None or k.grad.abs().max() == 0, f"{name} key should have no grad"
        assert v.grad is None or v.grad.abs().max() == 0, f"{name} val should have no grad"

    optimizer.step()
    snap_after = _snapshot(opt_model)
    for name in ["get_weather", "book_flight"]:
        changed, k_d, v_d = _params_changed(snap_before, snap_after, name)
        assert not changed, f"{name} should not change when frozen (k={k_d}, v={v_d})"
    print("  [PASS] All tools frozen -> no params changed")

    opt_model.set_trainable_tools(None)


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    opt_model = CacheOptimizeModel(model)

    # Register tools
    tmpl_kwargs = {"enable_thinking": False}
    per_tool = opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG, **tmpl_kwargs)
    print(f"Registered {len(per_tool)} tools: {[t['tool_name'] for t in per_tool]}")

    # Verify registry
    for entry in opt_model._registry.values():
        name = entry.get("tool_name", "?")
        k = getattr(opt_model, entry["key_param"])
        v = getattr(opt_model, entry["val_param"])
        print(f"  {name}: key={k.shape}, val={v.shape}, requires_grad={k.requires_grad}")

    test_requires_grad_toggle(opt_model)
    test_gradient_flow(opt_model, tokenizer)
    test_optimizer_update(opt_model, tokenizer)
    test_grad_accumulation(opt_model, tokenizer)
    test_no_trainable_skipped(opt_model, tokenizer)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
