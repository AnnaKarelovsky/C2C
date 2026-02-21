"""Unified training utilities for standard SFT, cache-optimized, and GKD training."""

from __future__ import annotations

import io
import json
import math
import os
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from openai import OpenAI
from torch.utils.data import DataLoader, Sampler
from transformers import get_cosine_schedule_with_warmup

from rosetta.optimize.dataset import PackedSFTDataset, collate_padded
from rosetta.optimize.utils import tool_meta_key


# ---------------------------------------------------------------------------
# On-policy distillation (OPD)
# ---------------------------------------------------------------------------


@dataclass
class OPDOutput:
    """Minimal output container with a ``.loss`` attribute."""

    loss: torch.Tensor
    metrics: dict = None


def opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Importance-sampling on-policy distillation loss (memory-efficient).

    Implements the OPD algorithm::

        advantage_t = log p_teacher(x_t) - log p_student(x_t)   [detached]
        ratio_t     = p_current(x_t) / p_old(x_t)               [differentiable]
        loss        = -(ratio * advantage).mean()

    In our single-process setup ``current = old`` (same model, no policy
    drift), so ``ratio = 1`` exactly and the gradient reduces to the
    REINFORCE policy gradient ``-∇log π_θ · A``.

    Uses ``F.cross_entropy`` (fused log-softmax + gather) so we never
    materialise a full ``(B, L, V)`` log-prob tensor.

    Reference: https://thinkingmachines.ai/blog/on-policy-distillation/

    Args:
        student_logits: ``(B, L, V)`` raw logits from the student.
        teacher_logits: ``(B, L, V)`` raw logits from the teacher.
        labels: ``(B, L)`` token ids with ``-100`` for positions to ignore.
    """
    V = student_logits.size(-1)
    flat_labels = labels.reshape(-1)

    # current log p_student (with gradient, via fused cross-entropy)
    student_nll = F.cross_entropy(
        student_logits.reshape(-1, V), flat_labels,
        ignore_index=-100, reduction="none",
    )
    current_logprobs = -student_nll

    # sampled log p_student (detached = behavior policy snapshot)
    sampled_logprobs = current_logprobs.detach()

    # teacher log p_teacher (no gradient)
    with torch.no_grad():
        teacher_nll = F.cross_entropy(
            teacher_logits.reshape(-1, V), flat_labels,
            ignore_index=-100, reduction="none",
        )
        teacher_logprobs = -teacher_nll

    mask = flat_labels != -100
    n_tokens = mask.sum().clamp(min=1)

    # advantage = log p_teacher - log p_student (detached)
    advantage = teacher_logprobs - sampled_logprobs

    # importance sampling ratio (gradient flows through current_logprobs)
    ratio = torch.exp(current_logprobs - sampled_logprobs)
    # IS loss: -(ratio * advantage)
    loss = -(ratio * advantage)[mask].sum() / n_tokens

    # Metrics (detached)
    with torch.no_grad():
        kl = -advantage[mask].sum() / n_tokens  # KL(θ||teacher) per token
    metrics = {"kl": kl}

    return loss, metrics


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GroupedBatchSampler(Sampler):
    """Yields batches where all samples share the same group key.

    Shuffles within each group, then interleaves batches across groups
    (round-robin) so all groups are trained evenly.

    Args:
        group_keys: Per-sample group key (e.g. ``dataset.meta_keys``).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle within groups and across batches.
        generator: Optional ``torch.Generator`` for reproducibility.
    """

    def __init__(self, group_keys, batch_size, shuffle=True, generator=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

        # Group sample indices by key
        self.groups = defaultdict(list)
        for idx, key in enumerate(group_keys):
            self.groups[key].append(idx)

    def __iter__(self):
        # Shuffle within each group and chunk into batches
        all_batches = []
        for _, indices in self.groups.items():
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator)
                indices = [indices[i] for i in perm]
            all_batches.extend(
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            )

        # Shuffle batches so groups appear proportional to their size
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=self.generator)
            all_batches = [all_batches[i] for i in perm]

        yield from all_batches

    def __len__(self):
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups.values()
        )


def create_dataloader(
    hf_dataset,
    tokenizer,
    *,
    batch_size: int = 2,
    max_length: int = 4096,
    seed: int = 42,
    template_kwargs: Optional[dict] = None,
    pre_processor=None,
    group_by_meta_key: bool = False,
    keep_raw: bool = False,
    passthrough_columns: Optional[List[str]] = None,
) -> DataLoader:
    """Create a DataLoader with role-masked labels.

    Args:
        hf_dataset: HuggingFace dataset with ``messages`` and ``tools`` columns.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        seed: RNG seed for shuffling.
        template_kwargs: Extra kwargs for ``apply_chat_template``
            (e.g. ``enable_thinking=False``).
        pre_processor: Optional callable ``(messages) -> messages`` applied
            before tokenization (e.g. :func:`~rosetta.optimize.dataset.fill_reasoning`).
        group_by_meta_key: If ``True``, use :class:`GroupedBatchSampler` so
            each batch contains only samples with the same tool set
            (``meta_key``).
        keep_raw: If ``True``, preserve raw ``messages``/``tools`` JSON
            strings in each sample (as ``_messages``/``_tools``).
        passthrough_columns: Column names to preserve in each batch.
            See :class:`~rosetta.optimize.dataset.PackedSFTDataset`.
    """
    dataset = PackedSFTDataset(
        hf_dataset, tokenizer, max_length=max_length,
        template_kwargs=template_kwargs, pre_processor=pre_processor,
        keep_raw=keep_raw, passthrough_columns=passthrough_columns,
    )
    g = torch.Generator()
    g.manual_seed(seed)

    collate_fn = lambda b: collate_padded(b, pad_token_id=tokenizer.pad_token_id)

    if group_by_meta_key and hasattr(dataset, "meta_keys"):
        sampler = GroupedBatchSampler(
            dataset.meta_keys, batch_size, shuffle=True, generator=g,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=g,
    )


def on_policy_generate(
    model,
    batch: dict,
    tokenizer,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
) -> Optional[dict]:
    """Generate on-policy outputs from the student model's prompts.

    Finds the prompt boundary (first supervised token in ``labels``)
    for each sample, left-pads prompts, and generates completions
    in a single batched call.

    Args:
        model: Student model (or callable supporting ``.generate()``).
        batch: Dict with ``input_ids`` ``(B, L)``, ``labels`` ``(B, L)``,
            and optionally ``attention_mask``.
        tokenizer: Tokenizer (used for ``pad_token_id``).
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        New batch dict, or ``None`` if generation failed / produced nothing.
    """
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    device = input_ids.device
    B = input_ids.shape[0]
    pad_id = tokenizer.pad_token_id

    # Find prompt boundary for each sample
    prompt_lens = []
    for i in range(B):
        supervised = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(supervised) == 0:
            return None
        prompt_end = supervised[0].item()
        if prompt_end == 0:
            return None
        prompt_lens.append(prompt_end)

    max_prompt_len = max(prompt_lens)

    # Left-pad prompts for batched generation
    padded_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
    for i in range(B):
        pl = prompt_lens[i]
        padded_ids[i, max_prompt_len - pl:] = input_ids[i, :pl]
        attn_mask[i, max_prompt_len - pl:] = 1

    with torch.no_grad():
        generated = model.generate(
            padded_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            pad_token_id=pad_id,
        )

    if generated.shape[1] <= max_prompt_len:
        return None

    new_attention_mask = (generated != pad_id).long()
    new_labels = generated.clone()
    new_labels[:, :max_prompt_len] = -100
    new_labels[generated == pad_id] = -100

    return {
        "input_ids": generated,
        "attention_mask": new_attention_mask,
        "labels": new_labels,
    }


def opd_forward_step(
    batch: dict,
    student_forward: Callable,
    teacher_forward: Callable,
    *,
    generate_fn: Optional[Callable] = None,
    lmbda: float = 1.0,
):
    """Single on-policy distillation step: generate + REINFORCE KL loss.

    Compatible with :func:`train_loop`'s ``forward_fn`` interface —
    returns ``(OPDOutput, n_tokens)``.

    Callbacks must return **aligned, un-shifted** logits:

    - ``student_forward(batch) -> (student_logits, labels)``
    - ``teacher_forward(batch) -> teacher_logits``

    This function handles the causal-LM shift and loss computation.

    Args:
        batch: Input batch dict.
        student_forward: ``(batch) -> (logits, labels)``.
        teacher_forward: ``(batch) -> logits``.
        generate_fn: ``(batch) -> (new_batch, rewards) | None`` —
            on-policy generator.  Mean of ``rewards`` is logged as
            ``"reward"`` in metrics.
        lmbda: On-policy probability (0=supervised only, 1=full on-policy).
    """
    # 1. On-policy generation
    rewards = None
    if generate_fn is not None and lmbda > 0 and random.random() <= lmbda:
        result = generate_fn(batch)
        if result is not None:
            batch, rewards = result

    # 2. Student forward (with grad)
    student_logits, labels = student_forward(batch)

    # 3. Teacher forward (no grad)
    with torch.no_grad():
        teacher_logits = teacher_forward(batch)

    # 4. Causal-LM shift: logit at position i predicts token at i+1
    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 5. OPD loss
    loss, metrics = opd_loss(shift_student, shift_teacher, shift_labels)
    if rewards:
        metrics["reward"] = torch.tensor(sum(rewards) / len(rewards))
    n_tokens = (shift_labels != -100).sum().clamp(min=1)
    return OPDOutput(loss=loss, metrics=metrics), n_tokens


def save_training_args(output_dir: str, training_args: dict):
    """Save training arguments as JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, "training_args.json")
    with open(args_path, "w") as f:
        json.dump(training_args, f, indent=2, default=str)


def train_loop(
    dataloader: DataLoader,
    trainable_params: List[torch.nn.Parameter],
    forward_fn: Callable,
    save_fn: Callable,
    output_dir: str,
    *,
    device: torch.device,
    lr: float = 5e-5,
    grad_accum: int = 4,
    max_length: int = 4096,
    warmup_ratio: float = 0.05,
    wandb_run=None,
    save_step: int = 0,
    eval_fn: Optional[Callable[[int], None]] = None,
    eval_step: int = 0,
    post_step_fn: Optional[Callable[[int], None]] = None,
    training_args: Optional[dict] = None,
):
    """Token-weighted training loop with gradient accumulation.

    Args:
        dataloader: Training data loader.
        trainable_params: Parameters to optimize.
        forward_fn: ``(batch) -> (output, n_supervised_tokens)`` where
            ``output.loss`` is the mean per-token loss.
        save_fn: ``(output_dir) -> None`` called after training.
        output_dir: Where to save the trained model/params.
        device: Device to move batches to.
        lr: Learning rate.
        grad_accum: Gradient accumulation steps.
        max_length: Max sequence length (used for gradient normalization).
        warmup_ratio: Fraction of total steps for linear warmup.
        wandb_run: Optional ``wandb.Run`` for logging.
        save_step: Save checkpoint every N steps (0 = only at end).
        eval_fn: Optional ``(global_step) -> None`` called every
            ``eval_step`` optimizer steps.  Handles its own printing
            and wandb logging.
        eval_step: Run ``eval_fn`` every N optimizer steps (0 = disabled).
        post_step_fn: Optional ``(global_step) -> None`` called after
            each optimizer step (e.g. to sync params to an inference
            server).
        training_args: Optional dict of training arguments (e.g.
            ``vars(args)``) to save as ``training_args.json`` alongside
            each checkpoint and the final model.
    """
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(dataloader) / grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Dataset: {len(dataloader.dataset)} samples | Steps: {total_steps}")
    print("=" * 60)

    NORM = max_length * grad_accum
    global_step = 0
    accum_loss = 0.0
    accum_tokens = 0
    accum_metrics = defaultdict(float)
    optimizer.zero_grad()

    if eval_fn is not None and eval_step > 0:
        eval_fn(0)

    for step, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        output, n_tokens = forward_fn(batch)

        loss = output.loss * (n_tokens / NORM)
        if loss.requires_grad:
            loss.backward()
        else:
            # All trainable params frozen for this batch — skip backward.
            pass
        accum_loss += output.loss.item() * n_tokens.item()
        accum_tokens += n_tokens.item()
        if getattr(output, "metrics", None):
            nt = n_tokens.item()
            for k, v in output.metrics.items():
                accum_metrics[k] += v.item() * nt

        if (step + 1) % grad_accum == 0:
            if accum_tokens == 0:
                optimizer.zero_grad()
                global_step += 1
                accum_loss = 0.0
                continue
            scale = NORM / accum_tokens
            for p in trainable_params:
                if p.grad is not None:
                    p.grad *= scale
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float("inf"))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            avg_loss = accum_loss / accum_tokens
            avg_metrics = {k: v / accum_tokens for k, v in accum_metrics.items()}
            cur_lr = scheduler.get_last_lr()[0]
            extra = "".join(f" | {k}: {v:.4f}" for k, v in avg_metrics.items())
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.4f}{extra} | "
                f"GradNorm: {grad_norm:.4f} | LR: {cur_lr:.2e}"
            )
            if wandb_run is not None:
                log_dict = {"train/loss": avg_loss, "train/lr": cur_lr,
                            "train/grad_norm": grad_norm.item()}
                for k, v in avg_metrics.items():
                    log_dict[f"train/{k}"] = v
                wandb_run.log(log_dict, step=global_step)
            if save_step > 0 and global_step % save_step == 0:
                ckpt_dir = f"{output_dir}/step_{global_step}"
                save_fn(ckpt_dir)
                if training_args is not None:
                    save_training_args(ckpt_dir, training_args)
                print(f"  Checkpoint saved to {ckpt_dir}")
            if post_step_fn is not None:
                post_step_fn(global_step)
            if eval_step > 0 and global_step % eval_step == 0 and eval_fn is not None:
                eval_fn(global_step)
            accum_loss = 0.0
            accum_tokens = 0
            accum_metrics = defaultdict(float)

    if (step + 1) % grad_accum != 0 and accum_tokens > 0:
        scale = NORM / accum_tokens
        for p in trainable_params:
            if p.grad is not None:
                p.grad *= scale
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float("inf"))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        avg_loss = accum_loss / accum_tokens
        avg_metrics = {k: v / accum_tokens for k, v in accum_metrics.items()}
        extra = "".join(f" | {k}: {v:.4f}" for k, v in avg_metrics.items())
        print(
            f"Step {global_step}/{total_steps} (partial) | "
            f"Loss: {avg_loss:.4f}{extra} | GradNorm: {grad_norm:.4f}"
        )
        if wandb_run is not None:
            log_dict = {"train/loss": avg_loss, "train/grad_norm": grad_norm.item()}
            for k, v in avg_metrics.items():
                log_dict[f"train/{k}"] = v
            wandb_run.log(log_dict, step=global_step)
        if post_step_fn is not None:
            post_step_fn(global_step)

    save_fn(output_dir)
    if training_args is not None:
        save_training_args(output_dir, training_args)
    print(f"Saved to {output_dir}")


def register_tools(opt_model, tokenizer, hf_dataset, **template_kwargs):
    """Scan dataset for unique tool sets and register them.

    Returns a ``meta_key -> kv_cache_indices`` mapping so that ``forward_fn``
    can look up the correct indices for each batch (grouped by tool set).
    """
    indices_map = {}
    for i in range(len(hf_dataset)):
        item = hf_dataset[i]
        messages = json.loads(item["messages"])
        tools = json.loads(item["tools"]) or None
        if not messages or not tools:
            continue
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        meta_key = tool_meta_key(tools, system_msg)
        if meta_key not in indices_map:
            per_tool = opt_model.register_tools(tokenizer, tools, system_msg, **template_kwargs)
            indices_map[meta_key] = [
                (entry["token_start"], entry["token_end"])
                for entry in per_tool
            ]
            tool_names = [e["tool_name"] for e in per_tool]
            print(f"Registered {len(tool_names)} tools: {tool_names} (meta_key={meta_key})")
    print(f"Total unique tool sets registered: {len(indices_map)}")
    return indices_map


class RolloutEngine:
    """Minimal wrapper around minisglang's OpenAI-compatible API + KV sync.

    Args:
        base_url: Server URL (e.g. ``http://localhost:1919``).
        model: Model name for the completions API.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1", api_key="none",
            timeout=600.0,
        )
        self.model = model

    def generate(
        self,
        messages_list: List[list],
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        tools_list: Optional[List] = None,
        **extra_body,
    ) -> List[dict]:
        """Generate completions for a batch of conversations concurrently.

        Fires all requests in parallel via threads so minisglang can
        batch them with continuous batching.  Uses streaming and
        collects both content and tool_calls per request.

        Requires the server to be launched with ``--tool-call-parser qwen``
        so that tool calls are returned as structured ``tool_calls``
        rather than raw ``<tool_call>`` tags in content.

        Args:
            messages_list: List of message lists, one per sample.
            max_tokens: Max tokens per generation.
            temperature: Sampling temperature.
            tools_list: Per-sample tool schemas (or ``None`` for all).
            **extra_body: Additional fields passed in ``extra_body``.

        Returns:
            List of assistant message dicts, each with ``role``,
            ``content``, and optionally ``tool_calls``.
        """
        n = len(messages_list)
        if tools_list is None:
            tools_list = [None] * n

        def _call(i):
            extra = dict(extra_body)
            if top_k > 0:
                extra["top_k"] = top_k
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages_list[i],
                tools=tools_list[i],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body=extra or None,
            )
            content_parts = []
            tool_calls_by_index = {}
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name or "",
                                    "arguments": tc.function.arguments or "",
                                },
                            }
                        else:
                            entry = tool_calls_by_index[idx]
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function.name:
                                entry["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments
            msg = {"role": "assistant", "content": "".join(content_parts)}
            tool_calls = [tool_calls_by_index[k]
                          for k in sorted(tool_calls_by_index)]
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return msg

        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(_call, range(n)))
        return results

    def update_weights_from_disk(self, model_path: str):
        """Tell the SGLang server to reload weights from disk."""
        requests.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": model_path, "flush_cache": True},
            timeout=120,
        ).raise_for_status()

    def update_opt_kv(self, kv_dict: dict):
        """POST ``{hash: (K, V)}`` tensors to ``/v1/update_opt_kv``."""
        buf = io.BytesIO()
        torch.save(kv_dict, buf)
        buf.seek(0)
        requests.post(
            f"{self.base_url}/v1/update_opt_kv",
            data=buf.read(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        ).raise_for_status()


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Poll a server until it responds or *timeout* seconds elapse.

    Checks ``GET {base_url}/v1/models`` every 2 s.  Raises
    :class:`TimeoutError` if the server does not become ready in time.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s")
