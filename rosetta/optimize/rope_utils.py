"""RoPE (Rotary Position Embedding) utilities for position-free KV cache storage.

Provides pure-math primitives to apply and invert RoPE on cached keys,
supporting both the Qwen (Qwen2/Llama) and GPT (GPT-OSS) rotation conventions.
"""

__all__ = ["RoPEConfig", "RoPEStyle", "extract_rope_config", "apply_rope", "inverse_rope"]

from dataclasses import dataclass
from enum import Enum, auto

import torch
from transformers import PreTrainedModel


class RoPEStyle(Enum):
    """Which rotation convention the model uses.

    QWEN (Qwen2/Llama):
        cos/sin have full head_dim via cat(freqs, freqs).
        rotate_half: cat(-x2, x1)
        apply: x * cos + rotate_half(x) * sin

    GPT (GPT-OSS):
        cos/sin have half head_dim (no doubling).
        apply: cat(x1*cos - x2*sin, x2*cos + x1*sin)
    """

    QWEN = auto()
    GPT = auto()


@dataclass
class RoPEConfig:
    rope_theta: float  # Base frequency (default 10000.0)
    head_dim: int  # Dimension per head
    num_kv_heads: int  # Number of KV heads (for GQA)
    num_layers: int  # Number of transformer layers
    rotary_dim: int  # Dims that get rotated (usually == head_dim)
    style: RoPEStyle = RoPEStyle.QWEN  # Rotation convention


def extract_rope_config(model: PreTrainedModel) -> RoPEConfig:
    """Auto-extract RoPE configuration from a HuggingFace model."""
    config = model.config
    rope_theta = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))

    # Detect style from model type
    model_type = getattr(config, "model_type", "")
    if model_type == "gpt_oss":
        style = RoPEStyle.GPT
    else:
        style = RoPEStyle.QWEN
    return RoPEConfig(
        rope_theta=rope_theta,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        rotary_dim=rotary_dim,
        style=style,
    )


def build_inv_freq(
    config: RoPEConfig, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Compute inverse frequencies for RoPE.

    Returns shape (rotary_dim // 2,), matching HF's _compute_default_rope_parameters.
    """
    dim = config.rotary_dim
    inv_freq = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=dtype)
            / dim
        )
    )
    return inv_freq


def compute_rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, style: RoPEStyle
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos and sin embeddings from positions and inverse frequencies.

    Args:
        positions: (N,) — 1D position indices
        inv_freq: (D//2,) — inverse frequencies
        style: which convention to use

    Returns:
        cos, sin — both shape (N, D) for STANDARD, (N, D//2) for GPT_OSS
    """
    # freqs: (N, D//2)
    freqs = torch.outer(positions.float(), inv_freq.float())
    if style == RoPEStyle.QWEN:
        emb = torch.cat((freqs, freqs), dim=-1)  # (N, D)
    else:
        emb = freqs  # (N, D//2) for GPT_OSS
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (Qwen2/Llama convention).

    Matches HF exactly: cat(-x2, x1).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_standard(
    keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, negate_sin: bool
) -> torch.Tensor:
    """Apply RoPE using the standard (Qwen2/Llama) convention.

    Formula: keys * cos + rotate_half(keys) * sin
    Inverse: keys * cos + rotate_half(keys) * (-sin)
    """
    s = sin if not negate_sin else -sin
    return (keys * cos) + (_rotate_half(keys) * s)


def _apply_rope_gpt_oss(
    keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, negate_sin: bool
) -> torch.Tensor:
    """Apply RoPE using the GPT-OSS convention.

    Forward:  cat(x1*cos - x2*sin, x2*cos + x1*sin)
    Inverse:  cat(x1*cos + x2*sin, x2*cos - x1*sin)  (negate sin)
    """
    first_half, second_half = torch.chunk(keys, 2, dim=-1)
    s = sin if not negate_sin else -sin
    first_ = first_half * cos - second_half * s
    second_ = second_half * cos + first_half * s
    return torch.cat((first_, second_), dim=-1)


def _apply_rope_impl(
    keys: torch.Tensor,
    positions: torch.Tensor,
    config: RoPEConfig,
    negate_sin: bool,
) -> torch.Tensor:
    """Shared implementation for apply_rope and inverse_rope.

    Args:
        keys: (..., N, D) — supports any leading dims
        positions: (N,) — 1D position indices
        config: RoPE configuration
        negate_sin: if True, negate sin for inverse rotation
    """
    input_dtype = keys.dtype
    keys_f32 = keys.float()

    inv_freq = build_inv_freq(config, device=keys.device, dtype=torch.float32)
    cos, sin = compute_rope_cos_sin(positions.to(keys.device), inv_freq, config.style)

    # Reshape cos/sin to broadcast: (1,...,1, N, cos_dim)
    # keys shape: (..., N, D), we need cos/sin to have same number of dims
    n_leading = keys.ndim - 2  # number of leading dimensions before N
    for _ in range(n_leading):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if config.rotary_dim < config.head_dim:
        # Partial rotary: only rotate first rotary_dim dims
        keys_rot = keys_f32[..., : config.rotary_dim]
        keys_pass = keys_f32[..., config.rotary_dim :]

        if config.style == RoPEStyle.QWEN:
            keys_rot = _apply_rope_standard(keys_rot, cos, sin, negate_sin)
        else:
            keys_rot = _apply_rope_gpt_oss(keys_rot, cos, sin, negate_sin)

        result = torch.cat((keys_rot, keys_pass), dim=-1)
    else:
        if config.style == RoPEStyle.QWEN:
            result = _apply_rope_standard(keys_f32, cos, sin, negate_sin)
        else:
            result = _apply_rope_gpt_oss(keys_f32, cos, sin, negate_sin)

    return result.to(input_dtype)


def apply_rope(
    keys: torch.Tensor, positions: torch.Tensor, config: RoPEConfig
) -> torch.Tensor:
    """Apply RoPE positional encoding to keys.

    Args:
        keys: (..., N, D) — supports any leading dims (L,B,H,N,D) or (B,H,N,D)
        positions: (N,) — 1D position indices
        config: RoPE configuration

    Returns:
        Tensor of same shape as keys with RoPE applied.
    """
    return _apply_rope_impl(keys, positions, config, negate_sin=False)


def inverse_rope(
    keys: torch.Tensor, positions: torch.Tensor, config: RoPEConfig
) -> torch.Tensor:
    """Remove RoPE positional encoding from keys (inverse rotation).

    Since RoPE is an orthogonal transformation, the inverse is simply
    the rotation with negated sin.

    Args:
        keys: (..., N, D) — supports any leading dims
        positions: (N,) — 1D position indices
        config: RoPE configuration

    Returns:
        Tensor of same shape as keys with RoPE removed.
    """
    return _apply_rope_impl(keys, positions, config, negate_sin=True)
