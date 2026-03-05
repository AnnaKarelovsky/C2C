"""Multi-round OPD training on tau-bench episodes.

Round-by-round on-policy distillation: each training step generates one
completion from the student (via RolloutEngine), trains on it (KL loss
vs teacher), then advances the episode (tool execution or user sim).
Unfinished episodes recycle into the next batch.

Usage:
    # Step 1: Launch minisgl server
    CUDA_VISIBLE_DEVICES=0 python -m minisgl --model Qwen/Qwen3-1.7B --port 30000 --tool-call-parser qwen --force-opt --opt-cache

    # Step 2: Run training
    CUDA_VISIBLE_DEVICES=1 python script/optimize/opd_multi_round.py \
        --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-1.7B \
        --domain airline --rollout-url http://localhost:30000 \
        --user-sim-model accounts/fireworks/models/kimi-k2p5 \
        --total-steps 100 --batch-size 4 --no-thinking
"""

from __future__ import annotations

import argparse
import random

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.benchmark.tau.evaluate import UserSimulator
from rosetta.benchmark.tau.interface import (
    get_system_prompt,
    get_tools_info,
    load_tau_tasks,
)
from rosetta.optimize.dataset import fill_reasoning
from rosetta.optimize.episode import Episode, EpisodeSource
from rosetta.optimize.interface.tau import TauInterface
from rosetta.optimize.train_utils import (
    RolloutEngine,
    opd_forward_step,
    seed_everything,
    supervised_logits_to_keep,
    train_loop,
)
from rosetta.optimize.wrapper import CacheOptimizeModel
from rosetta.workflow.basic_utils import msg_system, msg_user
from rosetta.workflow.camel_utils import create_model
from dotenv import find_dotenv, load_dotenv

def train(args):
    seed_everything(args.seed)
    load_dotenv(find_dotenv())

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Models ---
    student_device_map = {"": f"cuda:{args.student_gpu}"} if args.student_gpu is not None else "auto"
    teacher_device_map = {"": f"cuda:{args.teacher_gpu}"} if args.teacher_gpu is not None else "auto"

    print(f"Loading student: {args.student} (device_map={student_device_map}) ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map=student_device_map,
        attn_implementation=args.attn_impl,
    )
    opt_model = CacheOptimizeModel(student_model)

    print(f"Loading teacher: {args.teacher} (device_map={teacher_device_map}) ...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map=teacher_device_map,
        attn_implementation=args.attn_impl,
    )
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    # --- Register tools (single domain = one tool set) ---
    tools_info = get_tools_info(args.domain)
    wiki = get_system_prompt(args.domain)
    system_msg = msg_system(wiki)
    per_tool = opt_model.register_tools(tokenizer, tools_info, system_msg, **tmpl_kwargs)
    kv_indices = [(e["token_start"], e["token_end"]) for e in per_tool]
    tool_names = [e["tool_name"] for e in per_tool]
    print(f"Registered {len(tool_names)} tools: {tool_names}")

    # --- Rollout engine ---
    engine = RolloutEngine(args.rollout_url, args.student)
    print(f"Using rollout engine at {args.rollout_url}")
    engine.update_opt_kv(opt_model.get_opt_kv())
    print("Pushed initial KV params to rollout server")

    # --- User sim model ---
    user_sim_model = create_model(
        "fireworks", args.user_sim_model,
        temperature=0.0,
        max_tokens=2048,
        stream=True,
    )

    # --- Tasks ---
    tasks = load_tau_tasks(args.domain, split="test")
    print(f"Loaded {len(tasks)} tasks for domain={args.domain}")

    # --- Episode factory ---
    def new_episode():
        task = random.choice(tasks)
        user_sim = UserSimulator(user_sim_model)
        env = TauInterface.make_env(task, args.domain, user_sim=user_sim)
        env.reset()
        obs = user_sim.reset(task.instruction)
        return Episode(task=task, env=env, messages=[msg_system(wiki), msg_user(obs)])

    episode_source = EpisodeSource(
        new_episode_fn=new_episode,
        engine=engine,
        tokenizer=tokenizer, pool_size=args.batch_size,
        total_steps=args.total_steps,
        max_length=args.max_length,
        template_kwargs=tmpl_kwargs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pre_processor=fill_reasoning if args.no_thinking else None,
    )

    device = next(student_model.parameters()).device

    # --- forward_fn ---
    def forward_fn(batch):
        # Shared n_keep: computed once, used by both student and teacher.
        # Must account for prepare() stripping the prefix from student input.
        n_keep_box = [None]  # mutable box for closure sharing

        def student_forward(b):
            b_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            n_keep = supervised_logits_to_keep(b["labels"].to(device))
            prepared = opt_model.prepare(kv_cache_indices=kv_indices, **b_dev)
            labels = prepared.pop("labels", b_dev.get("labels"))
            n_keep = min(n_keep, prepared["input_ids"].shape[1])
            n_keep_box[0] = n_keep
            out = opt_model.forward(**prepared, logits_to_keep=n_keep)
            if labels is not None:
                labels = labels[:, -n_keep:]
            return out.logits, labels

        def teacher_forward(b):
            n_keep = n_keep_box[0]
            out = teacher_model(
                input_ids=b["input_ids"].to(teacher_model.device),
                attention_mask=b.get(
                    "attention_mask", torch.ones_like(b["input_ids"])
                ).to(teacher_model.device),
                logits_to_keep=n_keep,
            )
            return out.logits.to(device)

        return opd_forward_step(batch, student_forward, teacher_forward, top_k=args.kl_top_k)

    # --- Training ---
    trainable_params = [p for p in opt_model.parameters() if p.requires_grad]
    wandb_run = (
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        if not args.no_wandb else None
    )

    def post_step_fn(_step):
        engine.update_opt_kv(opt_model.get_opt_kv())

    train_loop(
        episode_source, trainable_params, forward_fn, opt_model.save_pretrained,
        args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, max_grad_norm=args.grad_norm,
        warmup_ratio=args.warmup_ratio, wandb_run=wandb_run,
        save_step=args.save_step,
        post_step_fn=post_step_fn,
        training_args=vars(args),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-round OPD training on tau-bench episodes")
    # Models
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-32B")
    parser.add_argument("--student-gpu", type=int, default=None, help="GPU id for student/SLM (default: auto)")
    parser.add_argument("--teacher-gpu", type=int, default=None, help="GPU id for teacher/LLM (default: auto)")
    parser.add_argument("--attn-impl", default="flash_attention_2")
    # Domain
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"])
    parser.add_argument("--user-sim-model", default="accounts/fireworks/models/kimi-k2p5")
    # Output
    parser.add_argument("--output-dir", default="local/checkpoints/test/opd_multi_round")
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200, help="Total training batches")
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Loss
    parser.add_argument("--kl-top-k", type=int, default=256, help="Top-k for sparse KL loss (0=REINFORCE-style single-token)")
    # Logging
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="c2c-optimize")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--save-step", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    # Rollout engine
    parser.add_argument("--rollout-url", default="http://localhost:30002")
    args = parser.parse_args()

    train(args)
