"""On-Policy Distillation (OPD) training.

Distills a teacher model (e.g. Qwen3-4B) into a student model
(e.g. Qwen3-1.7B) using REINFORCE-style per-token KL loss.

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/

Usage:
    python script/optimize/opd_training.py train \
        --student Qwen/Qwen3-1.7B \
        --teacher Qwen/Qwen3-4B \
        --dataset local/datasets/full/apigen_test
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.dataset import fill_reasoning
from rosetta.optimize.train_utils import (
    create_dataloader,
    on_policy_generate,
    opd_forward_step,
    seed_everything,
    train_loop,
)


def train(args):
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)

    split = hf_dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    dataloader = create_dataloader(
        train_dataset, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        seed=args.seed,
        template_kwargs=tmpl_kwargs or None,
        pre_processor=fill_reasoning if args.no_thinking else None,
    )

    print(f"Loading student: {args.student} ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="auto",
    )
    student_model.train()
    device = next(student_model.parameters()).device

    print(f"Loading teacher: {args.teacher} ...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="auto",
    )
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    def student_forward(batch):
        out = student_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        return out.logits, batch["labels"].to(device)

    def teacher_forward(batch):
        out = teacher_model(
            input_ids=batch["input_ids"].to(teacher_model.device),
            attention_mask=batch["attention_mask"].to(teacher_model.device),
        )
        return out.logits.to(device)

    def generate_fn(batch):
        return on_policy_generate(
            student_model, batch, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    def forward_fn(batch):
        return opd_forward_step(
            batch, student_forward, teacher_forward,
            generate_fn=generate_fn, lmbda=args.lmbda,
        )

    def save_fn(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        student_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args)) if not args.no_wandb else None

    eval_fn = None
    if args.eval_step > 0 and len(eval_dataset) > 0:
        eval_samples = [eval_dataset[i] for i in range(min(args.eval_samples, len(eval_dataset)))]

        def eval_fn(global_step):
            student_model.eval()
            correct = 0
            for item in eval_samples:
                messages = json.loads(item["messages"])
                prompt = [m for m in messages if m["role"] != "assistant"]
                expected = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), None,
                )
                if expected is None:
                    continue
                input_ids = tokenizer.apply_chat_template(
                    prompt, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", **tmpl_kwargs,
                ).to(device)
                with torch.no_grad():
                    output = student_model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    )
                generated = tokenizer.decode(
                    output[0, input_ids.shape[1]:], skip_special_tokens=True,
                )
                if expected.strip() in generated:
                    correct += 1
            student_model.train()
            acc = correct / len(eval_samples)
            print(f"  [Eval] accuracy: {acc:.2%} ({correct}/{len(eval_samples)})")
            if wandb_run is not None:
                wandb_run.log({"eval/accuracy": acc}, step=global_step)

    train_loop(
        dataloader, trainable_params, forward_fn, save_fn, args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, max_grad_norm=args.grad_norm,
        wandb_run=wandb_run, save_step=args.save_step,
        eval_fn=eval_fn, eval_step=args.eval_step,
        training_args=vars(args),
    )


def generate(args):
    print(f"Loading model from {args.output_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir, dtype=torch.bfloat16, device_map="auto",
    )

    question = "What is the capital of France and what is it known for?"
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=256, do_sample=False)
    generated = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {question}")
    print(f"A: {generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-policy distillation training")
    parser.add_argument("command", choices=["train", "generate", "both"], nargs="?", default="train")
    # Models
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    # Data
    parser.add_argument("--dataset", default="local/datasets/full/apigen")
    parser.add_argument("--output-dir", default="local/checkpoints/opd_example")
    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    # OPD hyperparameters
    parser.add_argument("--lmbda", type=float, default=1.0,
                        help="On-policy fraction: 0=supervised, 1=full on-policy")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for on-policy generation")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens per on-policy generation")
    # Logging
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="c2c-optimize")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template")
    parser.add_argument("--save-step", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--eval-step", type=int, default=0,
                        help="Run eval every N steps (0 = disabled)")
    parser.add_argument("--eval-samples", type=int, default=10,
                        help="Number of eval samples for accuracy check")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "both":
        train(args)
        generate(args)
