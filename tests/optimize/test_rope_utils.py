"""Tests for rosetta.optimize.rope_utils."""

import pytest
import torch

from rosetta.optimize.rope_utils import (
    RoPEConfig,
    RoPEStyle,
    apply_rope,
    extract_rope_config,
    inverse_rope,
)
from rosetta.optimize.rope_utils import build_inv_freq, compute_rope_cos_sin

# Static configs for pure-math tests (no model loading needed)
GPT_OSS_CONFIG = RoPEConfig(
    rope_theta=150000.0,
    head_dim=64,
    num_kv_heads=8,
    num_layers=24,
    rotary_dim=64,
    style=RoPEStyle.GPT,
)

QWEN3_CONFIG = RoPEConfig(
    rope_theta=1000000.0,
    head_dim=128,
    num_kv_heads=8,
    num_layers=28,
    rotary_dim=128,
    style=RoPEStyle.QWEN,
)


class TestRoundTrip:
    """apply_rope -> inverse_rope should recover the original keys."""

    @pytest.mark.parametrize("cfg,B,H,D", [
        (GPT_OSS_CONFIG, 2, 8, 64),
        (QWEN3_CONFIG, 2, 8, 128),
    ], ids=["gpt_oss", "qwen3"])
    def test_round_trip(self, cfg, B, H, D):
        torch.manual_seed(42)
        N = 16
        keys = torch.randn(B, H, N, D)
        positions = torch.arange(N)

        rotated = apply_rope(keys, positions, cfg)
        recovered = inverse_rope(rotated, positions, cfg)

        torch.testing.assert_close(recovered, keys, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("cfg,B,H,D", [
        (GPT_OSS_CONFIG, 1, 8, 64),
        (QWEN3_CONFIG, 1, 8, 128),
    ], ids=["gpt_oss", "qwen3"])
    def test_round_trip_nonzero_start(self, cfg, B, H, D):
        """Round trip with positions not starting at 0."""
        torch.manual_seed(123)
        N = 8
        keys = torch.randn(B, H, N, D)
        positions = torch.arange(100, 100 + N)

        rotated = apply_rope(keys, positions, cfg)
        recovered = inverse_rope(rotated, positions, cfg)

        torch.testing.assert_close(recovered, keys, atol=1e-5, rtol=1e-5)


class TestHFConsistency:
    """Our apply_rope must match the model's HF implementation exactly."""

    def test_matches_hf_gpt_oss(self):
        pytest.importorskip(
            "transformers.models.gpt_oss.modeling_gpt_oss",
            reason="gpt_oss model not available",
        )
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            _apply_rotary_emb,
        )

        torch.manual_seed(42)
        B, H, N, D = 2, 8, 16, 64
        keys = torch.randn(B, H, N, D)
        positions = torch.arange(N)

        ours = apply_rope(keys, positions, GPT_OSS_CONFIG)

        inv_freq = build_inv_freq(GPT_OSS_CONFIG, device=keys.device)
        cos, sin = compute_rope_cos_sin(positions, inv_freq, RoPEStyle.GPT)
        cos_hf = cos.unsqueeze(0).unsqueeze(1)
        sin_hf = sin.unsqueeze(0).unsqueeze(1)
        hf_result = _apply_rotary_emb(keys, cos_hf, sin_hf)

        torch.testing.assert_close(ours, hf_result, atol=1e-6, rtol=1e-6)

    def test_matches_hf_qwen3(self):
        pytest.importorskip(
            "transformers.models.qwen3.modeling_qwen3",
            reason="qwen3 model not available",
        )
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
        )

        torch.manual_seed(42)
        B, H, N, D = 2, 8, 16, 128
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        positions = torch.arange(N)

        ours = apply_rope(k, positions, QWEN3_CONFIG)

        inv_freq = build_inv_freq(QWEN3_CONFIG, device=k.device)
        cos, sin = compute_rope_cos_sin(positions, inv_freq, RoPEStyle.QWEN)
        # Qwen3's apply_rotary_pos_emb expects (B, N, D) and does
        # unsqueeze(1) internally for the head dimension.
        cos_hf = cos.unsqueeze(0)  # (1, N, D)
        sin_hf = sin.unsqueeze(0)
        _, hf_k = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)

        torch.testing.assert_close(ours, hf_k, atol=1e-6, rtol=1e-6)


class TestConfigExtraction:
    """extract_rope_config should correctly read model config attributes."""

    def test_config_extraction(self, model_and_tokenizer, model_info):
        model, _ = model_and_tokenizer
        rope_cfg = extract_rope_config(model)

        assert rope_cfg.rope_theta == model_info["rope_theta"]
        assert rope_cfg.head_dim == model_info["head_dim"]
        assert rope_cfg.num_kv_heads == model_info["num_kv_heads"]
        assert rope_cfg.num_layers == model_info["num_layers"]
        assert rope_cfg.rotary_dim == model_info["rotary_dim"]
        assert rope_cfg.style == model_info["rope_style"]


class TestPositionShift:
    """Verify position re-assignment: remove at pos A, re-apply at pos B."""

    @pytest.mark.parametrize("cfg,H,D", [
        (GPT_OSS_CONFIG, 8, 64),
        (QWEN3_CONFIG, 8, 128),
    ], ids=["gpt_oss", "qwen3"])
    def test_position_shift(self, cfg, H, D):
        torch.manual_seed(42)
        B, N = 1, 4
        keys = torch.randn(B, H, N, D)

        pos_5 = torch.arange(5, 5 + N)
        pos_10 = torch.arange(10, 10 + N)

        rotated_5 = apply_rope(keys, pos_5, cfg)
        stripped = inverse_rope(rotated_5, pos_5, cfg)
        result = apply_rope(stripped, pos_10, cfg)

        expected = apply_rope(keys, pos_10, cfg)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestPartialRotaryDim:
    """When rotary_dim < head_dim, only the first rotary_dim dims are rotated."""

    def test_partial_rotary_gpt_oss_style(self):
        torch.manual_seed(42)
        cfg = RoPEConfig(
            rope_theta=150000.0,
            head_dim=64,
            num_kv_heads=8,
            num_layers=24,
            rotary_dim=32,
            style=RoPEStyle.GPT,
        )
        B, H, N, D = 2, 8, 8, 64
        keys = torch.randn(B, H, N, D)
        positions = torch.arange(N)

        rotated = apply_rope(keys, positions, cfg)

        torch.testing.assert_close(
            rotated[..., 32:], keys[..., 32:], atol=1e-7, rtol=0.0
        )
        assert not torch.allclose(rotated[..., :32], keys[..., :32], atol=1e-3)

    def test_partial_rotary_standard_style(self):
        """Also verify partial rotary with standard (Qwen2/Qwen3) convention."""
        torch.manual_seed(42)
        cfg = RoPEConfig(
            rope_theta=10000.0,
            head_dim=128,
            num_kv_heads=8,
            num_layers=28,
            rotary_dim=64,
            style=RoPEStyle.QWEN,
        )
        B, H, N, D = 2, 8, 8, 128
        keys = torch.randn(B, H, N, D)
        positions = torch.arange(N)

        rotated = apply_rope(keys, positions, cfg)

        torch.testing.assert_close(
            rotated[..., 64:], keys[..., 64:], atol=1e-7, rtol=0.0
        )
        assert not torch.allclose(rotated[..., :64], keys[..., :64], atol=1e-3)

        recovered = inverse_rope(apply_rope(keys, positions, cfg), positions, cfg)
        torch.testing.assert_close(recovered, keys, atol=1e-5, rtol=1e-5)


class TestPrefillRoundTrip:
    """Load model, prefill to get real KV cache, then inverse+apply."""

    @pytest.mark.slow
    def test_prefill_inverse_apply(self, model_and_tokenizer, model_info):
        model, tokenizer = model_and_tokenizer
        model.eval()

        rope_cfg = extract_rope_config(model)

        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        positions = torch.arange(seq_len, device=model.device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        cache = outputs.past_key_values

        for layer_idx in range(rope_cfg.num_layers):
            cached_keys = cache.layers[layer_idx].keys
            stripped = inverse_rope(cached_keys.float(), positions, rope_cfg)
            reapplied = apply_rope(stripped, positions, rope_cfg)

            torch.testing.assert_close(
                reapplied.to(cached_keys.dtype),
                cached_keys,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Layer {layer_idx}: inverse+apply failed to recover cached keys",
            )
