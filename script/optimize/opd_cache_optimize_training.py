"""OPD + CacheOptimize training: distill a teacher into learnable KV cache params.

Combines on-policy distillation (REINFORCE-style per-token KL loss) with
CacheOptimizeModel. The student uses learned KV cache segments for tool
descriptions; the teacher runs standard forward passes. Only KV cache
parameters receive gradients.

On-policy generation is handled by a minisglang server (--rollout-url).

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/

Usage:
    python script/optimize/opd_cache_optimize_training.py train
    python script/optimize/opd_cache_optimize_training.py generate
"""

from __future__ import annotations

import argparse
import json

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.dataset import tokenize_item, collate_padded, fill_reasoning, parse_supervise_roles
from rosetta.optimize.train_utils import (
    RolloutEngine,
    create_dataloader,
    opd_forward_step,
    register_tools,
    seed_everything,
    supervised_logits_to_keep,
    train_loop,
)
from rosetta.optimize.wrapper import CacheOptimizeModel
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
    opt_model = CacheOptimizeModel(student_model)

    print(f"Loading teacher: {args.teacher} ...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    supervise_roles = parse_supervise_roles(args.supervise)
    indices_map = register_tools(opt_model, tokenizer, train_dataset, **tmpl_kwargs)
    register_tools(opt_model, tokenizer, eval_dataset, **tmpl_kwargs)

    engine = RolloutEngine(args.rollout_url, args.student)
    print(f"Using rollout engine at {args.rollout_url}")
    engine.update_opt_kv(opt_model.get_opt_kv())
    print("Pushed initial KV params to rollout server")

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
        group_by_meta_key=True,
        keep_raw=True,
        supervise_roles=supervise_roles,
    )
    device = next(student_model.parameters()).device

    def forward_fn(batch):
        meta_key = batch.pop("meta_key")[0]
        kv_indices = indices_map[meta_key]
        prefill_end = max(e for _, e in kv_indices)

        # Pop raw fields before they reach model forward
        raw_messages = batch.pop("_messages", None)
        raw_tools = batch.pop("_tools", None)

        def student_forward(b):
            b_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            b_dev.pop("meta_key", None)
            n_keep = supervised_logits_to_keep(b["labels"].to(device))
            prepared = opt_model.prepare(kv_cache_indices=kv_indices, **b_dev)
            labels = prepared.pop("labels", b_dev.get("labels"))
            n_keep = min(n_keep, prepared["input_ids"].shape[1])
            out = opt_model.forward(**prepared, logits_to_keep=n_keep)
            if labels is not None:
                labels = labels[:, -n_keep:]
            return out.logits, labels

        def teacher_forward(b):
            n_keep = supervised_logits_to_keep(b["labels"].to(device))
            out = teacher_model(
                input_ids=b["input_ids"].to(teacher_model.device),
                attention_mask=b.get("attention_mask",
                    torch.ones_like(b["input_ids"])).to(teacher_model.device),
                logits_to_keep=n_keep,
            )
            return out.logits.to(device)

        def generate_fn(b):
            if not raw_messages:
                return None
            prompts, tools_list, tool_jsons, msgs_for_reward = [], [], [], []
            for msg_json, tool_json in zip(raw_messages, raw_tools):
                msgs = json.loads(msg_json)
                prompt = _extract_prompt(msgs)
                tls = json.loads(tool_json) or None
                for _ in range(args.n_rollouts):
                    prompts.append(prompt)
                    tools_list.append(tls)
                    tool_jsons.append(tool_json)
                    msgs_for_reward.append(msg_json)

            extra = {"chat_template_kwargs": tmpl_kwargs} if tmpl_kwargs else {}
            # Tell minisglang to use optimized KV cache for these tools
            opt_tool_names = [
                t["function"]["name"] for t in (tools_list[0] or [])
            ]
            if opt_tool_names:
                extra["opt_tools"] = opt_tool_names
            completions = engine.generate(
                prompts, max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                tools_list=tools_list,
                **extra,
            )

            # Compute reward against ground truth
            rewards = task_interface.reward(completions, msgs_for_reward)

            new_items = []
            for prompt, completion, tool_json in zip(
                prompts, completions, tool_jsons,
            ):
                full = prompt + [completion]
                result = tokenize_item(
                    tokenizer,
                    {"messages": json.dumps(full), "tools": tool_json},
                    args.max_length, tmpl_kwargs,
                    pre_processor=fill_reasoning if args.no_thinking else None,
                    supervise_roles=supervise_roles,
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
            top_k=args.kl_top_k,
            use_outcome_reward=args.outcome_reward,
        )

    trainable_params = [p for p in opt_model.parameters() if p.requires_grad]
    wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args)) if not args.no_wandb else None

    InterfaceCls = INTERFACES[args.interface]
    interface_kwargs = dict(
        engine=engine, eval_prompt=eval_prompt, eval_tools=eval_tools,
        tmpl_kwargs=tmpl_kwargs, wandb_run=wandb_run,
    )
    if args.interface == "countdown":
        interface_kwargs["tokenizer"] = tokenizer
    task_interface = InterfaceCls(**interface_kwargs)
    eval_fn = task_interface.eval_fn if eval_prompt is not None else None

    def post_step_fn(step):
        engine.update_opt_kv(opt_model.get_opt_kv())

    train_loop(
        dataloader, trainable_params, forward_fn, opt_model.save_pretrained,
        args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, max_grad_norm=args.grad_norm,
        weight_decay=args.weight_decay,
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
        args.student, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl,
    )
    opt_model = CacheOptimizeModel(model)
    opt_model.load_pretrained(args.output_dir)
    print(f"Loaded KV params from {args.output_dir}")

    chat = []
    if system_msg:
        chat.append(system_msg)
    chat.append({"role": "user", "content": QUESTION})

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    result = opt_model.prepare_chat(
        tokenizer, chat, tools, for_generate=True,
        template_kwargs=tmpl_kwargs or None,
    )
    prompt_len = result["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **{k: v.to(model.device) if isinstance(v, torch.Tensor) else v
               for k, v in result.items()},
            max_new_tokens=1024, do_sample=False,
        )

    generated = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    print(f"\nQ: {QUESTION}")
    print(f"A: {generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPD + CacheOptimize training")
    parser.add_argument("command", choices=["train", "generate", "both"], nargs="?", default="train")
    # Models
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        help="Attention implementation for HF models (e.g. flash_attention_2, sdpa, eager)")
    # Data
    parser.add_argument("--dataset", default="local/datasets/full/apigen")
    parser.add_argument("--output-dir", default="local/checkpoints/opd_cacheopt_example")
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
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
    parser.add_argument("--kl-top-k", type=int, default=256,
                        help="Top-k for sparse KL loss (0=REINFORCE-style single-token)")
    parser.add_argument("--n-rollouts", type=int, default=4,
                        help="Number of completions per prompt (more data per sample)")
    parser.add_argument("--outcome-reward", action="store_true",
                        help="Weight KL loss per sample by its reward (outcome-weighted distillation)")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="AdamW weight decay")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Fraction of total steps for linear warmup")
    # Rollout engine (minisglang server for on-policy generation)
    parser.add_argument("--rollout-url", default="http://localhost:30000",
                        help="Base URL of a minisglang server")
    parser.add_argument("--interface", default="tau", choices=list(INTERFACES),
                        help="Task interface for reward/eval (tau or aime)")
    parser.add_argument("--supervise", default="assistant",
                        help="Comma-separated roles to supervise: assistant, tool, tool_call")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "both":
        train(args)
        generate(args)
