"""
Tau-bench multi-turn tool-calling evaluation.

Evaluates agents on airline (50 tasks) and retail (115 tasks) domains
using LLM-simulated users and stateful tool execution.

Usage:
    # Single task dry run
    python script/workflow/eval_tau.py --domain airline --limit 1 --num-workers 1

    # 5 airline tasks
    python script/workflow/eval_tau.py --domain airline --limit 5 --num-workers 1

    # Full airline (50 tasks)
    python script/workflow/eval_tau.py --domain airline --num-workers 4

    # Full retail (115 tasks)
    python script/workflow/eval_tau.py --domain retail --num-workers 4

    # Resume interrupted run
    python script/workflow/eval_tau.py --domain airline --resume

    # Multiple trials per task
    python script/workflow/eval_tau.py --domain airline --num-trials 3
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rosetta.workflow.camel_utils import create_model, read_jsonl, setup_env, write_jsonl


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(
    worker_id: int,
    task_indices: list[int],
    args: argparse.Namespace,
    process_dir: Path,
) -> None:
    """Worker process: create models, loop over tasks, write results."""
    setup_env()

    # Lazy imports inside worker to avoid tau-bench shim issues in main process
    from rosetta.benchmark.tau.evaluate import UserSimulator, solve_task
    from rosetta.benchmark.tau.interface import (
        get_data_load_func,
        get_system_prompt,
        get_tools_info,
        get_tools_map,
        load_tau_tasks,
        reset_data,
    )

    # --- Create agent model ---
    if args.model_provider == "hf":
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpu_ids = visible.split(",")
            assigned = gpu_ids[worker_id % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = assigned

        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_type, dtype=torch.bfloat16, device_map="auto",
        )

        if args.lora_checkpoint:
            from peft import PeftModel
            hf_model = PeftModel.from_pretrained(hf_model, args.lora_checkpoint)
            hf_model.eval()

        opt_model = None
        if args.opt_checkpoint or args.use_cache_opt:
            from rosetta.optimize.wrapper import CacheOptimizeModel
            opt_model = CacheOptimizeModel(hf_model)
            if args.opt_checkpoint:
                opt_model.load_pretrained(args.opt_checkpoint)

        agent_model = create_model(
            provider="hf",
            model=hf_model,
            tokenizer=hf_tokenizer,
            opt_model=opt_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=not args.no_thinking,
        )
    else:
        opt_model = None
        hf_tokenizer = None
        agent_model = create_model(
            provider=args.model_provider,
            model_type=args.model_type,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # --- Create user simulator model ---
    user_model = create_model(
        provider=args.user_model_provider,
        model_type=args.user_model_type,
        temperature=0.0,
        max_tokens=2048,
    )
    user_sim = UserSimulator(user_model)

    # --- Load domain data ---
    tasks = load_tau_tasks(args.domain, split=args.task_split)
    tools_map = get_tools_map(args.domain)
    tools_info = get_tools_info(args.domain)
    if args.no_thinking_tool:
        tools_info = [t for t in tools_info if t["function"]["name"] != "think"]
        tools_map = {k: v for k, v in tools_map.items() if k != "think"}
    wiki = get_system_prompt(args.domain)
    data_load_func = get_data_load_func(args.domain)

    # Register full domain tools on CacheOptimizeModel.
    # Even with a checkpoint, we register the full tool set so that
    # prepare_chat() can find them.  Individual tool KV params are
    # matched by content hash, so trained params are reused.
    if opt_model is not None:
        tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
        system_msg = {"role": "system", "content": wiki}
        opt_model.register_tools(hf_tokenizer, tools_info, system_msg, **tmpl_kwargs)

    out_file = process_dir / f"worker_{worker_id}.jsonl"
    traj_file = process_dir / f"worker_{worker_id}_traj.jsonl"

    with out_file.open("a", encoding="utf-8") as fout, \
         traj_file.open("a", encoding="utf-8") as ftraj:
        for task_idx in task_indices:
            for trial in range(args.num_trials):
                t0 = time.time()
                err = None
                result = None

                try:
                    data = data_load_func()
                    result = solve_task(
                        model=agent_model,
                        user_sim=user_sim,
                        task=tasks[task_idx],
                        tools_info=tools_info,
                        tools_map=tools_map,
                        wiki=wiki,
                        data=data,
                        data_load_func=data_load_func,
                        max_steps=args.max_steps,
                    )
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"

                seconds = time.time() - t0

                # Lightweight record (answers)
                record = {
                    "task_id": task_idx,
                    "trial": trial,
                    "reward": result["reward"] if result else 0.0,
                    "info": result.get("info") if result else {},
                    "seconds": round(seconds, 2),
                    "error": err,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                # Heavy trajectory (messages + actions)
                # Strip internal _-prefixed keys from messages before saving
                saved_msgs = [
                    {k: v for k, v in m.items() if not k.startswith("_")}
                    for m in (result["messages"] if result else [])
                ]
                trajectory = {
                    "task_id": task_idx,
                    "trial": trial,
                    "messages": saved_msgs,
                    "tools": tools_info,
                    "actions": result["actions"] if result else [],
                }
                ftraj.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                ftraj.flush()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pass_at_k(records: list[dict], num_trials: int) -> dict:
    """Compute pass rate and pass^k metrics.

    pass^k = fraction of tasks passing in at least 1 of k trials.
    """
    # Group by task_id
    by_task: dict[int, list[float]] = {}
    for rec in records:
        tid = rec["task_id"]
        by_task.setdefault(tid, []).append(rec.get("reward", 0.0))

    num_tasks = len(by_task)
    if num_tasks == 0:
        return {"num_tasks": 0, "pass_rate": 0.0}

    # pass^1 = average reward across tasks (best single trial)
    pass_1 = sum(max(rewards) for rewards in by_task.values()) / num_tasks

    metrics = {"num_tasks": num_tasks, "pass_rate": pass_1}

    # pass^k for k = 1..num_trials
    if num_trials > 1:
        for k in range(1, num_trials + 1):
            # pass^k: task passes if at least 1 of k trials succeeds
            # For each task with n trials, P(fail all k) = C(n-s, k) / C(n, k)
            # where s = number of successes, n = total trials
            pass_k_count = 0
            for rewards in by_task.values():
                n = len(rewards)
                s = sum(1 for r in rewards if r >= 1.0)
                if s >= 1 and k <= n:
                    # P(at least 1 success in k) = 1 - C(n-s, k) / C(n, k)
                    if n - s < k:
                        prob = 1.0
                    else:
                        prob = 1.0 - math.comb(n - s, k) / math.comb(n, k)
                    pass_k_count += prob
            metrics[f"pass^{k}"] = pass_k_count / num_tasks

    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    output_path: Path,
    records: list[dict],
    metrics: dict,
    args: argparse.Namespace,
) -> Path:
    """Write summary.txt next to the output file. Returns summary path."""
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    avg_seconds = sum(r.get("seconds", 0) for r in records) / total if total else 0

    # Per-task breakdown
    by_task: dict[int, list[float]] = {}
    for rec in records:
        by_task.setdefault(rec["task_id"], []).append(rec.get("reward", 0.0))

    lines = [
        "=" * 80,
        f"TAU-BENCH EVALUATION SUMMARY — {args.domain.upper()}",
        "=" * 80,
        f"Domain: {args.domain} ({args.task_split} split)",
        f"Tasks: {metrics['num_tasks']}, Trials per task: {args.num_trials}",
        f"Total records: {total}, Errors: {errors}",
        f"Average time per record: {avg_seconds:.1f}s",
        "",
        "--- Metrics ---",
        f"  Pass rate: {metrics['pass_rate']:.4f}",
    ]
    if args.num_trials > 1:
        for k in range(1, args.num_trials + 1):
            key = f"pass^{k}"
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]:.4f}")

    lines += [
        "",
        "--- Arguments ---",
    ]
    for k, v in vars(args).items():
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        lines.append(f"  --{k.replace('_', '-')} {v}")

    lines += [
        "",
        "--- Per-task Breakdown ---",
        f"{'Task ID':<10} {'Best Reward':<15} {'Trials':}",
        "-" * 50,
    ]
    for tid in sorted(by_task.keys()):
        rewards = by_task[tid]
        best = max(rewards)
        detail = ", ".join(f"{r:.0f}" for r in rewards)
        lines.append(f"{tid:<10} {best:<15.0f} {detail}")

    lines += ["", f"Output: {output_path}", "=" * 80]

    summary_path = output_path.parent / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tau-bench evaluation")

    # Domain
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"])
    parser.add_argument("--task-split", default="test", choices=["test", "train", "dev"])

    # Agent model
    parser.add_argument("--model-provider", default="fireworks")
    parser.add_argument(
        "--model-type",
        default="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=30)

    # User simulator model
    parser.add_argument("--user-model-provider", default="fireworks")
    parser.add_argument(
        "--user-model-type",
        default="accounts/fireworks/models/kimi-k2p5",
    )

    # HF-specific
    parser.add_argument("--use-cache-opt", action="store_true",
                        help="Use CacheOptimizeModel for learnable KV cache optimization (hf provider only)")
    parser.add_argument("--opt-checkpoint", default=None,
                        help="Path to pretrained CacheOptimizeModel checkpoint (implies --use-cache-opt)")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="Path to PEFT LoRA adapter directory (hf provider only)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to HF chat template")
    parser.add_argument("--no-thinking-tool", action="store_true",
                        help="Remove the 'think' tool from the tool set")

    # Data / output
    parser.add_argument("--limit", type=int, nargs="+", default=[0])
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-trials", type=int, default=1)

    # Workers
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    # Output path — follows subfolder structure:
    #   local/trajectory/tau/{experiment_name}/{domain}.jsonl
    if args.output is None:
        # Derive experiment name from model/checkpoint
        if args.opt_checkpoint:
            exp = Path(args.opt_checkpoint).name
        elif args.lora_checkpoint:
            exp = Path(args.lora_checkpoint).name
        else:
            exp = args.model_type.split("/")[-1]
        args.output = f"local/trajectory/tau/{exp}/{args.domain}.jsonl"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load tasks to get count ---
    setup_env()
    from rosetta.benchmark.tau.interface import load_tau_tasks

    tasks = load_tau_tasks(args.domain, split=args.task_split)
    num_tasks = len(tasks)

    # --- Determine task indices ---
    limit = args.limit
    if len(limit) == 2:
        start, end = limit
    elif limit[0] > 0:
        start, end = 0, limit[0]
    else:
        start, end = 0, num_tasks

    task_indices = list(range(start, min(end, num_tasks)))

    # --- Resume ---
    process_dir = output_path.parent / "process"
    process_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done: set[tuple[int, int]] = set()
        for wf in process_dir.glob("worker_*.jsonl"):
            for rec in read_jsonl(wf):
                done.add((rec["task_id"], rec.get("trial", 0)))
        before = len(task_indices)
        # Only keep tasks that have incomplete trials
        remaining = []
        for idx in task_indices:
            trials_done = sum(1 for t in range(args.num_trials) if (idx, t) in done)
            if trials_done < args.num_trials:
                remaining.append(idx)
        task_indices = remaining
        print(f"Resume: {before - len(task_indices)} tasks fully done, "
              f"{len(task_indices)} remaining")

    if not task_indices:
        print("No tasks to evaluate.")
        return

    console = Console()
    console.print(
        f"[bold]Tau-bench {args.domain}[/bold]: "
        f"{len(task_indices)} tasks x {args.num_trials} trials, "
        f"{args.num_workers} workers"
    )

    # --- Spawn workers ---
    num_workers = min(args.num_workers, len(task_indices))
    chunk_size = (len(task_indices) + num_workers - 1) // num_workers
    chunks = [
        task_indices[i : i + chunk_size]
        for i in range(0, len(task_indices), chunk_size)
    ]

    processes = []
    for wid, chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(wid, chunk, args, process_dir))
        p.start()
        processes.append(p)

    # --- Progress bar ---
    worker_files = [process_dir / f"worker_{i}.jsonl" for i in range(len(chunks))]

    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r") as f:
            return sum(1 for line in f if line.strip())

    def _refresh(progress, worker_tasks, overall):
        total_done = 0
        for wid, wf in enumerate(worker_files):
            count = _count_lines(wf)
            progress.update(worker_tasks[wid], completed=count)
            total_done += count
        progress.update(overall, completed=total_done)

    total_items = len(task_indices) * args.num_trials

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        worker_tasks = []
        for wid in range(len(chunks)):
            worker_tasks.append(
                progress.add_task(
                    f"Worker {wid}",
                    total=len(chunks[wid]) * args.num_trials,
                )
            )
        overall = progress.add_task("[bold]Overall", total=total_items)

        while any(p.is_alive() for p in processes):
            _refresh(progress, worker_tasks, overall)
            time.sleep(0.5)
        _refresh(progress, worker_tasks, overall)

    for p in processes:
        p.join()

    # --- Aggregate ---
    all_records: list[dict] = []
    all_trajs: list[dict] = []
    for wid in range(len(chunks)):
        rec_file = process_dir / f"worker_{wid}.jsonl"
        traj_file = process_dir / f"worker_{wid}_traj.jsonl"
        if rec_file.exists():
            all_records.extend(read_jsonl(rec_file))
        if traj_file.exists():
            all_trajs.extend(read_jsonl(traj_file))

    write_jsonl(output_path, all_records)
    traj_output = output_path.parent / (output_path.stem + "_trajectories.jsonl")
    write_jsonl(traj_output, all_trajs)

    # --- Write first trajectory as pretty-printed example.json ---
    if all_trajs:
        example_path = output_path.parent / "example.json"
        example_path.write_text(
            json.dumps(all_trajs[0], indent=4, ensure_ascii=False), encoding="utf-8",
        )

    # --- Metrics ---
    metrics = compute_pass_at_k(all_records, args.num_trials)
    errors = sum(1 for r in all_records if r.get("error"))

    # --- Write summary.txt ---
    summary_path = write_summary(output_path, all_records, metrics, args)

    # --- Console output ---
    console.print(f"\n[bold]Results ({args.domain}):[/bold]")
    console.print(f"  Tasks: {metrics['num_tasks']}")
    console.print(f"  Pass rate: {metrics['pass_rate']:.3f}")
    if args.num_trials > 1:
        for k in range(1, args.num_trials + 1):
            key = f"pass^{k}"
            if key in metrics:
                console.print(f"  {key}: {metrics[key]:.3f}")
    if errors:
        console.print(f"  Errors: {errors}")

    console.print(f"\n  Records: {output_path}")
    console.print(f"  Trajectories: {traj_output}")
    if all_trajs:
        console.print(f"  Example: {output_path.parent / 'example.json'}")
    console.print(f"  Summary: {summary_path}")

    # Show per-task breakdown
    by_task: dict[int, list[float]] = {}
    for rec in all_records:
        by_task.setdefault(rec["task_id"], []).append(rec.get("reward", 0.0))

    table = Table(title="Per-task Results", show_header=True, header_style="bold cyan")
    table.add_column("Task ID", justify="right")
    table.add_column("Reward", justify="right")
    if args.num_trials > 1:
        table.add_column("Trials", justify="right")

    for tid in sorted(by_task.keys()):
        rewards = by_task[tid]
        best = max(rewards)
        style = "green" if best >= 1.0 else "red"
        if args.num_trials > 1:
            detail = ", ".join(f"{r:.0f}" for r in rewards)
            table.add_row(str(tid), f"{best:.0f}", detail, style=style)
        else:
            table.add_row(str(tid), f"{best:.0f}", style=style)

    console.print(table)


if __name__ == "__main__":
    main()
