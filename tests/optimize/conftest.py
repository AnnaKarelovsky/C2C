"""Shared fixtures for optimize tests.

Usage:
    pytest test/optimize/                          # default: openai/gpt-oss-20b
    pytest test/optimize/ --model Qwen/Qwen3-1.7B  # use Qwen3
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.rope_utils import extract_rope_config
from rosetta.optimize.wrapper import _get_full_attention_layers


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default="openai/gpt-oss-20b",
        help="HuggingFace model name for optimize tests",
    )


@pytest.fixture(scope="session")
def model_name(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def model_and_tokenizer(model_name):
    """Load model + tokenizer once per test session."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


@pytest.fixture(scope="session")
def model_info(model_and_tokenizer):
    """Derived model info for assertions (computed, not hardcoded)."""
    model, _ = model_and_tokenizer
    config = model.config
    rope_cfg = extract_rope_config(model)
    fa_layers = _get_full_attention_layers(model)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return {
        "model_type": getattr(config, "model_type", ""),
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "rope_theta": rope_cfg.rope_theta,
        "rotary_dim": rope_cfg.rotary_dim,
        "rope_style": rope_cfg.style,
        "full_attention_layers": fa_layers,
        "num_fa_layers": len(fa_layers),
    }
