"""AIME interface for OPD training.

Implements the uniform ``TaskInterface`` for AIME math problems:
answer accuracy as reward, qualitative generation as eval.
"""

from __future__ import annotations

import json
import re

import wandb
from camel.toolkits import FunctionTool
from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse,
    verify,
)
from latex2sympy2_extended import NormalizationConfig

SYSTEM_PROMPT = "You are a helpful math assistant. Put your final answer in \\boxed{}."


def prompt_tool() -> str:
    """This is NOT a tool to be called. It is a placeholder for instructions.

    Do NOT call this tool. Instead, follow the instructions below to guide
    your behavior when responding to user queries.

    General guidelines:
    - Think step by step before answering.
    - Be concise and accurate in your responses.
    - Use available tools when you need external information.

    Returns:
        str: A continuation signal.
    """
    return "continue"


PROMPT_TOOL = [FunctionTool(prompt_tool).get_openai_tool_schema()]


def _load_aime2025():
    """Load opencompass/AIME2025 (both parts) as list of (question, answer)."""
    from datasets import concatenate_datasets, load_dataset

    ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    ds = concatenate_datasets([ds1, ds2])
    return [(row["question"], row["answer"]) for row in ds]


class AimeInterface:
    """AIME reward and eval interface.

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
        self._eval_data = None  # lazy-loaded

    def _get_eval_data(self):
        if self._eval_data is None:
            self._eval_data = _load_aime2025()
        return self._eval_data

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, completions: list[dict], raw_messages: list[str]) -> list[float]:
        """Per-sample answer accuracy reward (1 if correct, else 0)."""
        rewards = []
        for completion, msg_json in zip(completions, raw_messages):
            gt = _extract_ground_truth(json.loads(msg_json))
            if gt is None:
                rewards.append(0.0)
            else:
                rewards.append(_answer_reward(completion, gt))
        return rewards

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_fn(self, global_step: int, n_samples: int = 4):
        """Roll out on AIME2025 (30 problems x n_samples) and log accuracy."""
        eval_data = self._get_eval_data()
        extra = {"chat_template_kwargs": self.tmpl_kwargs} if self.tmpl_kwargs else {}

        # Repeat each problem n_samples times
        prompts = []
        tools_list = []
        for question, _ in eval_data:
            for _ in range(n_samples):
                prompts.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ])
                tools_list.append(PROMPT_TOOL)

        completions = self.engine.generate(
            prompts, max_tokens=8192, temperature=0.9,
            tools_list=tools_list, **extra,
        )

        total_correct = 0
        pass_at_n = 0
        table_rows = []
        for i, (question, answer) in enumerate(eval_data):
            sample_completions = completions[i * n_samples : (i + 1) * n_samples]
            scores = [_answer_reward(c, answer) for c in sample_completions]
            n_correct = sum(scores)
            total_correct += n_correct
            passed = int(max(scores))
            pass_at_n += passed
            outputs = " | ".join(
                c.get("content", "")[-60:] for c in sample_completions
            )
            table_rows.append([
                i + 1, question[:100], answer, outputs, n_correct, passed,
            ])

        avg_pass1 = total_correct / (len(eval_data) * n_samples)
        pass_n = pass_at_n / len(eval_data)
        print(
            f"  [Eval] AIME2025 avg pass@1: {avg_pass1:.1%} | "
            f"pass@{n_samples}: {pass_at_n}/{len(eval_data)} = {pass_n:.1%}"
        )

        if self.wandb_run is not None:
            self.wandb_run.log({
                "eval/aime_pass@1": avg_pass1,
                f"eval/aime_pass@{n_samples}": pass_n,
            }, step=global_step)
            table = wandb.Table(
                columns=["#", "question", "answer", "model_outputs", "n_correct", "pass"],
                data=table_rows,
            )
            self.wandb_run.log({"eval/aime_details": table}, step=global_step)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_ground_truth(messages: list) -> str | None:
    """Extract the ground-truth answer from the assistant \\boxed{} turn."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            m = _BOXED_RE.search(msg.get("content", ""))
            if m:
                return m.group(1)
    return None


def _answer_reward(completion: dict, ground_truth: str) -> float:
    """1 if the model answer matches the ground truth, 0 otherwise."""
    model_text = completion.get("content", "")
    try:
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            model_text,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return 0.0
        return 1.0 if verify(gold, answer) else 0.0
    except Exception:
        return 0.0
