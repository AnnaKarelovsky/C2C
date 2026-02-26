"""Cache-optimize training: learn KV cache parameters for tool descriptions.

Freezes model weights and trains only the KV cache segments registered via
CacheOptimizeModel.register_tools(). Only assistant turns are supervised.

Usage:
    python script/optimize/cache_optimize_training.py train
    python script/optimize/cache_optimize_training.py generate
    python script/optimize/cache_optimize_training.py infer --no-thinking --port 1919
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys

import openai
import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.optimize.dataset import fill_reasoning
from rosetta.optimize.train_utils import create_dataloader, register_tools, seed_everything, train_loop, wait_for_server
from rosetta.optimize.wrapper import CacheOptimizeModel

QUESTION = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"


def _round_tool_names(messages_json):
    """Return tool names called after the last user message in this round."""
    messages = json.loads(messages_json)
    lu = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    return [
        tc["function"]["name"]
        for m in messages[lu + 1:]
        if m["role"] == "assistant"
        for tc in (m.get("tool_calls") or [])
    ]


def _prepare_full_tools(hf_dataset, full_tools_domain, round_level=False, filter_empty=False):
    """Replace dataset tools with full domain set, stash originals as trainable_tools.

    When ``round_level=True``, trainable_tools are the tools called in this
    round's supervised portion only.  Empty means "all trainable" at forward time.
    """
    from rosetta.optimize.interface.tau import TauInterface

    # Build per-domain lookup: {domain: (tools_json, valid_names)}
    domains = ["airline", "retail"] if full_tools_domain == "all" else [full_tools_domain]
    if full_tools_domain == "all" and "domain" not in hf_dataset.column_names:
        raise ValueError("Dataset must have a 'domain' column for --full-tools all")

    domain_lookup = {}
    for d in domains:
        tools = TauInterface.full_tools(domain=d)
        domain_lookup[d] = (json.dumps(tools), {t["function"]["name"] for t in tools})
        print(f"Full tool set ({d}): {len(tools)} tools")

    def _add_trainable(example):
        d = example.get("domain", domains[0])
        if d not in domain_lookup:
            return {"trainable_tools": json.dumps([]), "tools": example["tools"]}
        tools_json, valid_names = domain_lookup[d]
        if round_level:
            called = _round_tool_names(example["messages"])
        else:
            called = [t["function"]["name"] for t in json.loads(example["tools"])]
        called = [n for n in called if n in valid_names]
        return {"trainable_tools": json.dumps(called), "tools": tools_json}

    hf_dataset = hf_dataset.map(_add_trainable, desc="Preparing full-tools")

    if not round_level or filter_empty:
        n_before = len(hf_dataset)
        hf_dataset = hf_dataset.filter(
            lambda x: bool(json.loads(x["trainable_tools"])),
            desc="Filtering samples with no trainable tools",
        )
        n_dropped = n_before - len(hf_dataset)
        if n_dropped:
            print(f"Dropped {n_dropped}/{n_before} samples with no trainable tools")
    return hf_dataset


def train(args):
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = load_from_disk(args.dataset)
    n_before = len(hf_dataset)
    hf_dataset = hf_dataset.filter(lambda x: bool(json.loads(x["tools"])))
    if len(hf_dataset) < n_before:
        print(f"Filtered {n_before - len(hf_dataset)} samples without tools")

    # Full-tool mode: replace per-trajectory tools with full domain set
    use_full_tools = args.full_tools is not None
    passthrough = []
    if use_full_tools:
        hf_dataset = _prepare_full_tools(hf_dataset, args.full_tools, round_level=args.round_level_trainable, filter_empty=args.remove_no_tool_rounds)
        passthrough = ["trainable_tools"]

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    opt_model = CacheOptimizeModel(model)
    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    indices_map = register_tools(opt_model, tokenizer, hf_dataset, **tmpl_kwargs)

    dataloader = create_dataloader(
        hf_dataset, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        seed=args.seed,
        template_kwargs=tmpl_kwargs or None,
        pre_processor=fill_reasoning if args.no_thinking else None,
        group_by_meta_key=True,
        passthrough_columns=passthrough or None,
    )
    device = next(model.parameters()).device

    def forward_fn(batch):
        meta_key = batch.pop("meta_key")[0]  # all same in grouped batch
        # Freeze non-active tools when using full-tool mode
        trainable_tools_json = batch.pop("trainable_tools", None)
        if trainable_tools_json is not None:
            trainable_names = json.loads(trainable_tools_json[0])
            # Empty list = all tools trainable (round with no tool calls)
            opt_model.set_trainable_tools(trainable_names or None)
        prepared = opt_model.prepare(
            kv_cache_indices=indices_map[meta_key], **batch,
        )
        output = opt_model.forward(**prepared)
        n_tokens = (prepared["labels"] != -100).sum()
        return output, n_tokens

    trainable_params = [p for p in opt_model.parameters() if p.requires_grad]
    wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args)) if not args.no_wandb else None
    train_loop(
        dataloader, trainable_params, forward_fn, opt_model.save_pretrained,
        args.output_dir,
        device=device, lr=args.lr, grad_accum=args.grad_accum,
        max_length=args.max_length, max_grad_norm=args.grad_norm,
        wandb_run=wandb_run, save_step=args.save_step,
        training_args=vars(args),
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


def infer(args):
    base_url = f"http://localhost:{args.port}"

    # Read opt_tools from checkpoint config
    config_path = os.path.join(args.output_dir, "kv_config.json")
    with open(config_path) as f:
        kv_config = json.load(f)
    opt_tools = kv_config.get("opt_tools", [])
    print(f"Optimized tools from checkpoint: {opt_tools}")

    # Extract tools and system message from dataset
    hf_dataset = load_from_disk(args.dataset)
    item = hf_dataset[0]
    messages = json.loads(item["messages"])
    tools = json.loads(item["tools"]) or None
    system_msg = next((m for m in messages if m["role"] == "system"), None)

    # Launch mini-sglang server
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", args.model,
        "--opt-cache", args.output_dir,
        "--port", str(args.port),
        "--tool-call-parser", "qwen",
        "--tp", str(args.tp),
    ]
    print(f"Launching: {' '.join(cmd)}")
    server_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        wait_for_server(base_url)
        print("Server ready.")

        client = openai.OpenAI(base_url=f"{base_url}/v1", api_key="unused")

        chat = []
        if system_msg:
            chat.append(system_msg)
        chat.append({"role": "user", "content": QUESTION})

        extra_body = {"opt_tools": opt_tools}
        if args.no_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        response = client.chat.completions.create(
            model=args.model,
            messages=chat,
            tools=tools,
            extra_body=extra_body,
        )

        msg = response.choices[0].message
        print(f"\nQ: {QUESTION}")
        if msg.content:
            print(f"A: {msg.content}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"Tool call: {tc.function.name}({tc.function.arguments})")
    finally:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        server_proc.wait()
        print("Server terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "generate", "infer", "both"], nargs="?", default="both")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="local/datasets/apigen_200")
    parser.add_argument("--output-dir", default="local/checkpoints/optCache_example")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=32e-3)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (useful for debugging)")
    parser.add_argument("--wandb-project", default="c2c-optimize")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template")
    parser.add_argument("--full-tools", default=None, metavar="DOMAIN",
                        help="Register full domain tool set (e.g. 'airline', 'retail') "
                             "and freeze non-active tools per batch")
    parser.add_argument("--round-level-trainable", action="store_true",
                        help="Only unfreeze tools called in this round's supervised portion "
                             "(requires --full-tools). Falls back to all-trainable if no "
                             "tools are called in the round.")
    parser.add_argument("--remove-no-tool-rounds", action="store_true",
                        help="Drop rounds where no tools are called "
                             "(requires --round-level-trainable).")
    parser.add_argument("--save-step", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--port", type=int, default=1919,
                        help="Port for mini-sglang server (infer command)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree (infer command)")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "both":
        train(args)
        generate(args)
