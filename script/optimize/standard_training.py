"""Standard SFT training on singletool trajectory data.

Usage:
    python script/optimize/standard_training.py train
    python script/optimize/standard_training.py generate
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.train import create_dataloader, seed_everything, train_loop

QUESTION = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"


def train(args):
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)
    dataloader = create_dataloader(
        hf_dataset, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        pack=False, seed=args.seed,
        template_kwargs={"enable_thinking": False} if args.no_thinking else None,
    )

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.train()
    device = next(model.parameters()).device

    def forward_fn(batch):
        output = model(**batch)
        n_tokens = (batch["labels"] != -100).sum()
        return output, n_tokens

    def save_fn(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    wandb_run = _init_wandb(args) if args.wandb else None
    train_loop(
        dataloader, list(model.parameters()), forward_fn, save_fn, args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, wandb_run=wandb_run,
    )


def generate(args):
    from camel.toolkits import FunctionTool
    from rosetta.workflow.retriever import search_engine

    tools = [FunctionTool(search_engine).get_openai_tool_schema()]

    print(f"Loading {args.output_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Use the provided tools to find information."},
        {"role": "user", "content": QUESTION},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True,
    )
    full_ids_t = torch.tensor([full_ids])

    with torch.no_grad():
        output = model.generate(
            full_ids_t.to(model.device), max_new_tokens=1024, do_sample=False,
        )

    generated = tokenizer.decode(output[0, len(full_ids):], skip_special_tokens=True)
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
    parser.add_argument("--output-dir", default="local/checkpoints/sft_example")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.6e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
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
