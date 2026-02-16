"""Unified training utilities for standard SFT and cache-optimized training."""

from __future__ import annotations

import math
import random
import time
import urllib.request
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import get_cosine_schedule_with_warmup

from rosetta.optimize.dataset import PackedSFTDataset, collate_padded


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
    pack: bool = True,
    seed: int = 42,
    template_kwargs: Optional[dict] = None,
    pre_processor=None,
    group_by_meta_key: bool = False,
) -> DataLoader:
    """Create a DataLoader with role-masked labels.

    Args:
        hf_dataset: HuggingFace dataset with ``messages`` and ``tools`` columns.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        pack: Whether to pack multiple samples into one sequence.
        seed: RNG seed for shuffling.
        template_kwargs: Extra kwargs for ``apply_chat_template``
            (e.g. ``enable_thinking=False``).
        pre_processor: Optional callable ``(messages) -> messages`` applied
            before tokenization (e.g. :func:`~rosetta.optimize.dataset.fill_reasoning`).
        group_by_meta_key: If ``True``, use :class:`GroupedBatchSampler` so
            each batch contains only samples with the same tool set
            (``meta_key``).  Requires ``pack=False``.
    """
    dataset = PackedSFTDataset(
        hf_dataset, tokenizer, max_length=max_length, pack=pack,
        template_kwargs=template_kwargs, pre_processor=pre_processor,
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
    eval_fn: Optional[Callable[[int], str]] = None,
    eval_step: int = 0,
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
        eval_fn: Optional ``(global_step) -> generated_text`` called every
            ``eval_step`` optimizer steps.  Logged to wandb as a versioned
            ``wandb.Table`` under ``eval/sample_output``.
        eval_step: Run ``eval_fn`` every N optimizer steps (0 = disabled).
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
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        output, n_tokens = forward_fn(batch)

        loss = output.loss * (n_tokens / NORM)
        loss.backward()
        accum_loss += output.loss.item() * n_tokens.item()
        accum_tokens += n_tokens.item()

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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            avg_loss = accum_loss / accum_tokens
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": avg_loss, "train/lr": cur_lr}, step=global_step
                )
            if save_step > 0 and global_step % save_step == 0:
                ckpt_dir = f"{output_dir}/step_{global_step}"
                save_fn(ckpt_dir)
                print(f"  Checkpoint saved to {ckpt_dir}")
            if eval_step > 0 and global_step % eval_step == 0 and eval_fn is not None:
                text = eval_fn(global_step)
                print(f"  [Eval] {text[:200]}...")
                if wandb_run is not None:
                    import wandb
                    table = wandb.Table(
                        columns=["step", "output"],
                        data=[[global_step, text]],
                    )
                    wandb_run.log({"eval/sample_output": table}, step=global_step)
            accum_loss = 0.0
            accum_tokens = 0

    if (step + 1) % grad_accum != 0 and accum_tokens > 0:
        scale = NORM / accum_tokens
        for p in trainable_params:
            if p.grad is not None:
                p.grad *= scale
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        avg_loss = accum_loss / accum_tokens
        print(f"Step {global_step}/{total_steps} (partial) | Loss: {avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"train/loss": avg_loss}, step=global_step)

    save_fn(output_dir)
    print(f"Saved to {output_dir}")


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
