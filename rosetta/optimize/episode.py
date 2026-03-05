"""Round-by-round episode data source for multi-round OPD training.

SLIME-inspired stateful data source that replaces DataLoader for multi-round
on-policy distillation. Each ``__next__`` call:

1. Get ``pool_size`` episodes (buffer-first, fresh for remainder)
2. Generate one completion per episode via :class:`RolloutEngine`
3. Tokenize last turn → supervised batch
4. Advance episodes via :meth:`ToolEnvironment.advance`
5. Buffer unfinished, discard done
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch

from rosetta.optimize.dataset import collate_padded, tokenize_last_turn
from rosetta.workflow.basic_utils import _clean_for_api


@dataclass
class Episode:
    """Single in-flight episode."""

    task: Any
    env: Any  # ToolEnvironment
    messages: list = field(default_factory=list)


class EpisodeSource:
    """Round-by-round episode data source for multi-round OPD.

    Implements ``__iter__`` / ``__next__`` / ``__len__`` so it can be
    passed directly to :func:`~rosetta.optimize.train_utils.train_loop`
    as the ``data_source`` argument.

    Args:
        new_episode_fn: Callable that returns a fresh :class:`Episode`.
            Encapsulates all task-specific setup (env creation, user sim
            reset, initial messages).
        engine: :class:`~rosetta.optimize.train_utils.RolloutEngine`.
        tokenizer: HuggingFace tokenizer.
        pool_size: Number of episodes per batch.
        total_steps: Total training steps (batches) to produce.
        max_length: Max sequence length for tokenization.
        template_kwargs: Chat template kwargs (e.g. ``enable_thinking=False``).
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        pre_processor: Optional message pre-processor (e.g. ``fill_reasoning``).
    """

    def __init__(
        self,
        new_episode_fn: Callable[[], Episode],
        engine,
        tokenizer,
        pool_size: int,
        total_steps: int,
        max_length: int,
        template_kwargs: dict,
        *,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        pre_processor: Optional[Callable] = None,
    ):
        self._new_episode_fn = new_episode_fn
        self._engine = engine
        self._tokenizer = tokenizer
        self._pool_size = pool_size
        self._total_steps = total_steps
        self._max_length = max_length
        self._tmpl_kwargs = template_kwargs or {}
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._pre_processor = pre_processor

        self._buffer: List[Episode] = []
        self._step_count = 0

    def __len__(self):
        return self._total_steps

    def __iter__(self):
        self._step_count = 0
        return self

    def __next__(self):
        if self._step_count >= self._total_steps:
            raise StopIteration
        batch = self._step()
        self._step_count += 1
        return batch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_episodes(self, n: int) -> List[Episode]:
        """Pull up to *n* episodes from buffer, creating new ones as needed."""
        episodes = self._buffer[:n]
        self._buffer = self._buffer[n:]
        while len(episodes) < n:
            episodes.append(self._new_episode_fn())
        return episodes

    def _step(self) -> dict:
        """Execute one round: generate, tokenize, advance."""
        episodes = self._get_episodes(self._pool_size)

        # 1. Generate completions via engine
        messages_list = [_clean_for_api(ep.messages) for ep in episodes]
        tools_list = [
            [t.get_openai_tool_schema() for t in ep.env.tools]
            for ep in episodes
        ]
        extra = {"chat_template_kwargs": self._tmpl_kwargs} if self._tmpl_kwargs else {}
        completions = self._engine.generate(
            messages_list,
            tools_list=tools_list,
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            **extra,
        )

        # 2. Append completions to messages, apply pre_processor, tokenize
        items = []
        for ep, completion, tools in zip(episodes, completions, tools_list):
            ep.messages.append(completion)
            msgs = self._pre_processor(ep.messages) if self._pre_processor else ep.messages
            ids, labels = tokenize_last_turn(
                self._tokenizer, msgs, tools,
                self._max_length, self._tmpl_kwargs,
            )
            items.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
            })
        batch = collate_padded(items, pad_token_id=self._tokenizer.pad_token_id)

        # 3. Advance episodes via env interface (parallel for I/O-bound envs)
        with ThreadPoolExecutor(max_workers=len(episodes)) as pool:
            futures = [
                pool.submit(ep.env.advance, completion, ep.messages)
                for ep, completion in zip(episodes, completions)
            ]
            new_buffer: List[Episode] = []
            for ep, future in zip(episodes, futures):
                try:
                    continues = future.result(timeout=60)
                except Exception:
                    continues = False
                if continues:
                    new_buffer.append(ep)
        self._buffer = new_buffer
        return batch
