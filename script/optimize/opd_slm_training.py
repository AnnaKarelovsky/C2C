"""OPD SLM training: distill a teacher into the full student model.

On-policy distillation (REINFORCE-style per-token KL loss) where the
student is a full SLM and weight sync with the SGLang rollout engine
uses ``update_weights_from_disk``.

Usage:
    python script/optimize/opd_slm_training.py train
    python script/optimize/opd_slm_training.py generate
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.dataset import _tokenize_item, collate_padded, fill_reasoning
from rosetta.optimize.train_utils import (
    RolloutEngine,
    create_dataloader,
    opd_forward_step,
    seed_everything,
    train_loop,
)
from rosetta.optimize.interface.aime import AimeInterface
from rosetta.optimize.interface.countdown import CountdownInterface
from rosetta.optimize.interface.tau import TauInterface

INTERFACES = {
    "tau": TauInterface,
    "aime": AimeInterface,
    "countdown": CountdownInterface,
}

QUESTION = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"


def _extract_prompt(messages):
    """Return the prompt portion (everything up to the last assistant turn)."""
    last_user = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user = i
            break
    if last_user == -1:
        return messages
    return messages[: last_user + 1]


def train(args):
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)
    n_before = len(hf_dataset)
    hf_dataset = hf_dataset.filter(lambda x: bool(json.loads(x["tools"])))
    if len(hf_dataset) < n_before:
        print(f"Filtered {n_before - len(hf_dataset)} samples without tools")

    split = hf_dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")

    print(f"Loading student: {args.student} ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )

    print(f"Loading teacher: {args.teacher} ...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    engine = RolloutEngine(args.rollout_url, args.student)
    print(f"Using rollout engine at {args.rollout_url}")

    # Task-specific interface (eval + reward)
    eval_prompt, eval_tools = None, None
    if args.eval_step > 0 and len(eval_dataset) > 0:
        eval_item = eval_dataset[0]
        eval_msgs = json.loads(eval_item["messages"])
        eval_tools = json.loads(eval_item["tools"]) or None
        eval_prompt = _extract_prompt(eval_msgs)

    dataloader = create_dataloader(
        train_dataset, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        seed=args.seed,
        template_kwargs=tmpl_kwargs or None,
        pre_processor=fill_reasoning if args.no_thinking else None,
        keep_raw=True,
    )
    device = next(student_model.parameters()).device

    def forward_fn(batch):
        # Pop raw fields before they reach model forward
        raw_messages = batch.pop("_messages", None)
        raw_tools = batch.pop("_tools", None)
        batch.pop("meta_key", None)

        def student_forward(b):
            b_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            b_dev.pop("meta_key", None)
            out = student_model(
                input_ids=b_dev["input_ids"],
                attention_mask=b_dev.get("attention_mask",
                    torch.ones_like(b_dev["input_ids"])),
            )
            labels = b_dev.get("labels", b["labels"].to(device))
            return out.logits, labels

        def teacher_forward(b):
            out = teacher_model(
                input_ids=b["input_ids"].to(teacher_model.device),
                attention_mask=b.get("attention_mask",
                    torch.ones_like(b["input_ids"])).to(teacher_model.device),
            )
            return out.logits.to(device)

        def generate_fn(b):
            if not raw_messages:
                return None
            prompts, tools_list, tool_jsons = [], [], []
            for msg_json, tool_json in zip(raw_messages, raw_tools):
                msgs = json.loads(msg_json)
                tls = json.loads(tool_json) or None
                prompts.append(_extract_prompt(msgs))
                tools_list.append(tls)
                tool_jsons.append(tool_json)

            extra = {"chat_template_kwargs": tmpl_kwargs} if tmpl_kwargs else {}
            completions = engine.generate(
                prompts, max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                tools_list=tools_list,
                **extra,
            )

            # Compute reward against ground truth
            rewards = task_interface.reward(completions, raw_messages)

            new_items = []
            for prompt, completion, tool_json in zip(
                prompts, completions, tool_jsons,
            ):
                full = prompt + [completion]
                result = _tokenize_item(
                    tokenizer,
                    {"messages": json.dumps(full), "tools": tool_json},
                    args.max_length, tmpl_kwargs,
                    pre_processor=fill_reasoning if args.no_thinking else None,
                )
                if result is None:
                    return None
                ids, labels, _ = result
                new_items.append({
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(ids), dtype=torch.long),
                })
            return collate_padded(new_items, pad_token_id=tokenizer.pad_token_id), rewards

        return opd_forward_step(
            batch, student_forward, teacher_forward,
            generate_fn=generate_fn, lmbda=args.lmbda,
        )

    trainable_params = list(student_model.parameters())
    wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args)) if not args.no_wandb else None

    InterfaceCls = INTERFACES[args.interface]
    task_interface = InterfaceCls(
        engine=engine, eval_prompt=eval_prompt, eval_tools=eval_tools,
        tmpl_kwargs=tmpl_kwargs, wandb_run=wandb_run,
    )
    eval_fn = task_interface.eval_fn if eval_prompt is not None else None

    tmp_dir = tempfile.mkdtemp(prefix="opd_slm_sync_")

    def post_step_fn(step):
        # Clean previous files before saving to avoid stale shards
        shutil.rmtree(tmp_dir, ignore_errors=True)
        student_model.save_pretrained(tmp_dir)
        engine.update_weights_from_disk(tmp_dir)

    def save_fn(output_dir):
        student_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    train_loop(
        dataloader, trainable_params, forward_fn, save_fn,
        args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, max_grad_norm=args.grad_norm,
        warmup_ratio=args.warmup_ratio, wandb_run=wandb_run,
        save_step=args.save_step,
        eval_fn=eval_fn, eval_step=args.eval_step,
        post_step_fn=post_step_fn,
        training_args=vars(args),
    )


def generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)
    item = hf_dataset[0]
    messages = json.loads(item["messages"])
    tools = json.loads(item["tools"]) or None
    system_msg = next((m for m in messages if m["role"] == "system"), None)

    print(f"Loading {args.student} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )

    chat = []
    if system_msg:
        chat.append(system_msg)
    chat.append({"role": "user", "content": QUESTION})

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    inputs = tokenizer.apply_chat_template(
        chat, tools=tools, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
        **tmpl_kwargs,
    )
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **{k: v.to(model.device) if isinstance(v, torch.Tensor) else v
               for k, v in inputs.items()},
            max_new_tokens=1024, do_sample=False,
        )

    generated = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    print(f"\nQ: {QUESTION}")
    print(f"A: {generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPD SLM training")
    parser.add_argument("command", choices=["train", "generate", "both"], nargs="?", default="train")
    # Models
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        help="Attention implementation for HF models (e.g. flash_attention_2, sdpa, eager)")
    # Data
    parser.add_argument("--dataset", default="local/datasets/full/apigen")
    parser.add_argument("--output-dir", default="local/checkpoints/opd_slm_example")
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Fraction of total steps for linear warmup")
    parser.add_argument("--seed", type=int, default=42)
    # OPD hyperparameters
    parser.add_argument("--lmbda", type=float, default=1.0,
                        help="On-policy fraction: 0=supervised, 1=full on-policy")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for on-policy generation")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling for on-policy generation")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling for on-policy generation (0=disabled)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens per on-policy generation")
    # Logging
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="c2c-optimize")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template")
    parser.add_argument("--save-step", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--eval-step", type=int, default=100,
                        help="Generate from a fixed sample every N steps (0 = disabled)")
    # Rollout engine (SGLang server for on-policy generation)
    parser.add_argument("--rollout-url", default="http://localhost:30000",
                        help="Base URL of an SGLang server")
    parser.add_argument("--interface", default="tau", choices=list(INTERFACES),
                        help="Task interface for reward/eval (tau or aime)")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "both":
        train(args)
        generate(args)
