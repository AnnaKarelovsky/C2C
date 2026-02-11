"""Unified training utilities for standard SFT and cache-optimized training."""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def create_dataloader(
    hf_dataset,
    tokenizer,
    *,
    batch_size: int = 2,
    max_length: int = 4096,
    pack: bool = True,
    seed: int = 42,
    template_kwargs: Optional[dict] = None,
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
    """
    dataset = PackedSFTDataset(
        hf_dataset, tokenizer, max_length=max_length, pack=pack,
        template_kwargs=template_kwargs,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_padded(b, pad_token_id=tokenizer.pad_token_id),
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
        batch = {k: v.to(device) for k, v in batch.items()}
        output, n_tokens = forward_fn(batch)

        loss = output.loss * (n_tokens / NORM)
        loss.backward()
        accum_loss += output.loss.item() * n_tokens.item()
        accum_tokens += n_tokens.item()

        if (step + 1) % grad_accum == 0:
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
            accum_loss = 0.0
            accum_tokens = 0

    if (step + 1) % grad_accum != 0:
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
