"""Base class for task interfaces used in OPD / cache-optimize training.

Every task interface must implement :meth:`reward` and :meth:`eval_fn`.
Tasks that use tools may override :meth:`full_tools` to provide the
complete tool set for a domain (used for full-tool-set training where
per-trajectory tools are a subset), and :meth:`make_env` to create a
:class:`ToolEnvironment` for multi-round rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class ToolEnvironment(ABC):
    """Episode-level abstraction for tool-use environments.

    Provides wrapped tools compatible with run_with_tools,
    and tracks per-episode state (actions, done).
    """

    tools: List[Any]  # tool objects (FunctionTool-like)
    done: bool

    @abstractmethod
    def reset(self, **kwargs) -> None: ...

    @abstractmethod
    def advance(self, completion: dict, messages: list) -> bool:
        """Advance episode after an assistant completion.

        Handles task-specific logic: tool execution for tool-call
        completions, user simulation for text responses, and stop
        condition detection.  Appends follow-up messages (tool results,
        user observations) to *messages* in-place.

        Args:
            completion: Assistant message dict (may contain ``tool_calls``).
            messages: Conversation history (mutated in-place).

        Returns:
            ``True`` if the episode should continue, ``False`` if done.
        """
        ...


class TaskInterface(ABC):
    """Abstract base for task-specific reward and evaluation.

    Subclasses must implement:

    - :meth:`reward` — per-sample scalar reward.
    - :meth:`eval_fn` — periodic evaluation with optional wandb logging.

    Subclasses may override:

    - :meth:`full_tools` — return the full tool set for a domain.
    - :meth:`make_env` — create a ToolEnvironment for multi-round rollouts.

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

    @staticmethod
    def make_env(**kwargs) -> Optional[ToolEnvironment]:
        """Create a ToolEnvironment for multi-round rollouts.

        Override in subclasses whose tasks involve tool execution
        (e.g. tau-bench). Returns ``None`` by default for non-tool tasks.
        """
        return None
