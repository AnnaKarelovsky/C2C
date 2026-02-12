"""Cache-optimize training: learn KV cache parameters for tool descriptions.

Freezes model weights and trains only the KV cache segments registered via
CacheOptimizeModel.register_tools(). Only assistant turns are supervised.

Usage:
    python script/optimize/cache_optimize_training.py train
    python script/optimize/cache_optimize_training.py generate
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.dataset import fill_reasoning
from rosetta.optimize.train_utils import create_dataloader, seed_everything, train_loop
from rosetta.optimize.wrapper import CacheOptimizeModel

QUESTION = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"


def _register_tools(opt_model, tokenizer, hf_dataset):
    """Scan dataset for unique tool sets and register them. Returns seg_info."""
    seg_info = None
    registered = set()
    for i in range(len(hf_dataset)):
        item = hf_dataset[i]
        messages = json.loads(item["messages"])
        tools = json.loads(item["tools"]) or None
        if not messages or not tools:
            continue
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        meta_key = CacheOptimizeModel._tool_meta_key(tools, system_msg)
        if meta_key not in registered:
            opt_model.register_tools(tokenizer, tools, system_msg)
            registered.add(meta_key)
            meta = opt_model._tool_metas[meta_key]
            seg_info = (meta["prefix_len"], meta["prefix_len"] + meta["segment_len"])
            print(f"Registered tools (meta_key={meta_key})")
    return seg_info


def train(args):
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    opt_model = CacheOptimizeModel(model)
    seg_start, seg_end = _register_tools(opt_model, tokenizer, hf_dataset)

    dataloader = create_dataloader(
        hf_dataset, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        pack=False, seed=args.seed,
        template_kwargs={"enable_thinking": False} if args.no_thinking else None,
        pre_processor=fill_reasoning if args.no_thinking else None,
    )
    device = next(model.parameters()).device

    def forward_fn(batch):
        prepared = opt_model.prepare(
            kv_cache_indices=[(seg_start, seg_end)], **batch,
        )
        output = opt_model.forward(**prepared)
        n_tokens = (prepared["labels"] != -100).sum()
        return output, n_tokens

    trainable_params = [p for p in opt_model.parameters() if p.requires_grad]
    wandb_run = _init_wandb(args) if not args.no_wandb else None
    train_loop(
        dataloader, trainable_params, forward_fn, opt_model.save_pretrained,
        args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, wandb_run=wandb_run,
    )


def generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract tools and system message from dataset
    hf_dataset = load_from_disk(args.dataset)
    item = hf_dataset[0]
    messages = json.loads(item["messages"])
    tools = json.loads(item["tools"]) or None
    system_msg = next((m for m in messages if m["role"] == "system"), None)

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    opt_model = CacheOptimizeModel(model)
    opt_model.load_pretrained(args.output_dir)
    print(f"Loaded KV params from {args.output_dir}")

    chat = []
    if system_msg:
        chat.append(system_msg)
    chat.append({"role": "user", "content": QUESTION})

    result = opt_model.prepare_chat(tokenizer, chat, tools, for_generate=True)
    prompt_len = result["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids=result["input_ids"].to(model.device),
            attention_mask=result["attention_mask"].to(model.device),
            past_key_values=result["past_key_values"],
            max_new_tokens=1024,
            do_sample=False,
        )

    generated = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    print(f"\nQ: {QUESTION}")
    print(f"A: {generated}")


def _init_wandb(args):
    import wandb
    return wandb.init(
        project=args.wandb_project, name=args.wandb_name, config=vars(args),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "generate", "both"], nargs="?", default="both")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="local/trajectory/qwen/qwen3_235b_thinking/hotpotqa/100_300/hotpotqa_dataset")
    parser.add_argument("--output-dir", default="local/checkpoints/optCache_example")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.6e-2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (useful for debugging)")
    parser.add_argument("--wandb-project", default="c2c-optimize")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "both":
        train(args)
        generate(args)
