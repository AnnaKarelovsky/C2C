"""Tau-bench interface for OPD training.

Implements the uniform ``TaskInterface`` for tau-bench / APIGen data:
tool-call accuracy as reward, qualitative generation as eval.
"""

from __future__ import annotations

import json

import wandb


class TauInterface:
    """Tau-bench reward and eval interface.

    Uniform API::

        interface.reward(completions, raw_messages) -> list[float]
        interface.eval_fn(global_step) -> None

    Args:
        engine: :class:`~rosetta.optimize.train_utils.RolloutEngine`.
        eval_prompt: Message list (prompt portion only).
        eval_tools: Tool schemas for the eval sample.
        tmpl_kwargs: Extra chat-template kwargs.
        wandb_run: Optional wandb run for logging.
    """

    def __init__(self, engine=None, eval_prompt=None, eval_tools=None,
                 tmpl_kwargs=None, wandb_run=None):
        self.engine = engine
        self.eval_prompt = eval_prompt
        self.eval_tools = eval_tools
        self.tmpl_kwargs = tmpl_kwargs or {}
        self.wandb_run = wandb_run

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, completions: list[dict], raw_messages: list[str]) -> list[float]:
        """Per-sample tool-call reward (1 if function names match, else 0)."""
        rewards = []
        for completion, msg_json in zip(completions, raw_messages):
            gt_turn = _extract_ground_truth(json.loads(msg_json))
            if gt_turn is None:
                rewards.append(0.0)
            else:
                rewards.append(_tool_call_reward(completion, gt_turn))
        return rewards

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_fn(self, global_step: int):
        """Generate a single completion and log to stdout / wandb."""
        extra = {"chat_template_kwargs": self.tmpl_kwargs} if self.tmpl_kwargs else {}
        completions = self.engine.generate(
            [self.eval_prompt], max_tokens=512, temperature=0,
            tools_list=[self.eval_tools], **extra,
        )
        msg = completions[0]
        text = msg.get("content", "")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                text += f"\n[tool_call: {tc['function']['name']}({tc['function']['arguments']})]"
        print(f"  [Eval] {text[:300]}...")
        if self.wandb_run is not None:
            table = wandb.Table(
                columns=["step", "output"],
                data=[[global_step, text]],
            )
            self.wandb_run.log({"eval/sample_output": table}, step=global_step)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_ground_truth(messages: list) -> dict | None:
    """Extract the ground-truth assistant turn right after the prompt."""
    last_user = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user = i
            break
    if last_user == -1 or last_user + 1 >= len(messages):
        return None
    nxt = messages[last_user + 1]
    if nxt.get("role") == "assistant":
        return nxt
    return None


def _get_call_names(msg: dict) -> set[str]:
    """Extract function names from an assistant message's tool_calls."""
    return {tc["function"]["name"] for tc in (msg.get("tool_calls") or [])}


def _tool_call_reward(generated_msg: dict, ground_truth_msg: dict) -> float:
    """1 if generated function names match ground truth, 0 otherwise."""
    gt_names = _get_call_names(ground_truth_msg)
    gen_names = _get_call_names(generated_msg)
    return 1.0 if gen_names == gt_names else 0.0
