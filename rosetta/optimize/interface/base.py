"""Base class for task interfaces used in OPD / cache-optimize training.

Every task interface must implement :meth:`reward` and :meth:`eval_fn`.
Tasks that use tools may override :meth:`full_tools` to provide the
complete tool set for a domain (used for full-tool-set training where
per-trajectory tools are a subset).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class TaskInterface(ABC):
    """Abstract base for task-specific reward and evaluation.

    Subclasses must implement:

    - :meth:`reward` — per-sample scalar reward.
    - :meth:`eval_fn` — periodic evaluation with optional wandb logging.

    Subclasses may override:

    - :meth:`full_tools` — return the full tool set for a domain.

    Args:
        engine: :class:`~rosetta.optimize.train_utils.RolloutEngine`.
        eval_prompt: Message list (prompt portion only).
        eval_tools: Tool schemas for the eval sample.
        tmpl_kwargs: Extra chat-template kwargs.
        wandb_run: Optional wandb run for logging.
    """

    def __init__(self, engine=None, eval_prompt=None, eval_tools=None,
                 tmpl_kwargs=None, wandb_run=None, **kwargs):
        self.engine = engine
        self.eval_prompt = eval_prompt
        self.eval_tools = eval_tools
        self.tmpl_kwargs = tmpl_kwargs or {}
        self.wandb_run = wandb_run

    @abstractmethod
    def reward(
        self, completions: list[dict], raw_messages: list[str],
    ) -> list[float]:
        """Compute per-sample reward.

        Args:
            completions: List of generated assistant message dicts.
            raw_messages: List of raw JSON strings from the dataset's
                ``messages`` column (for extracting ground truth).

        Returns:
            List of scalar rewards, one per sample.
        """
        ...

    @abstractmethod
    def eval_fn(self, global_step: int) -> None:
        """Run evaluation and optionally log to wandb.

        Args:
            global_step: Current optimizer step (for logging).
        """
        ...

    @staticmethod
    def full_tools(**kwargs) -> Optional[List[dict]]:
        """Return the full tool set for this task, or ``None``.

        Override in subclasses that support full-tool-set training.
        The training script calls this to register all tools and render
        templates with the complete set, while only training KV params
        for the per-trajectory subset.

        Returns:
            List of tool schemas (OpenAI format), or ``None`` if the
            task has no full-tool-set concept.
        """
        return None
