"""Example: generate with CacheOptimizeModel using the high-level tool API.

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,4 python script/opitmize/examples/generation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.wrapper import CacheOptimizeModel

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "openai/gpt-oss-20b"

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

# ── Load model ──────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

# ── Wrap & register tools ──────────────────────────────────────────────
opt_model = CacheOptimizeModel(model)
opt_model.register_tools(tokenizer, TOOLS, SYSTEM_MSG)
print("Tools registered as learnable KV segment.")

# ── Generate ────────────────────────────────────────────────────────────
user_msg = {"role": "user", "content": "What is machine learning?"}
messages = [SYSTEM_MSG, user_msg]

# Build the full token sequence (needed by model.generate)
full_ids = tokenizer.apply_chat_template(
    messages, tools=TOOLS, tokenize=True, add_generation_prompt=True,
)
full_ids_t = torch.tensor([full_ids])

# prepare_chat builds a KV cache with the tool segment replaced by
# learnable parameters (initialized from the frozen prefill).
with torch.no_grad():
    result = opt_model.prepare_chat(tokenizer, messages, TOOLS)
    output = model.generate(
        full_ids_t.to(model.device),
        past_key_values=result["past_key_values"],
        max_new_tokens=64,
        do_sample=False,
    )

generated = tokenizer.decode(output[0, len(full_ids):], skip_special_tokens=True)
print(f"\nQ: {user_msg['content']}")
print(f"A: {generated}")
