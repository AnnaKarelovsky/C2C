"""Countdown interface for OPD training.

Implements the uniform ``TaskInterface`` for the Countdown numbers game:
equation correctness as reward, accuracy on test set as eval.

The task: given a set of numbers and a target, create an arithmetic
equation (using +, -, *, /) that equals the target. Each number may
be used at most once.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List, Optional

import wandb
from camel.toolkits import FunctionTool
from transformers import AutoTokenizer

from rosetta.optimize.interface.base import TaskInterface

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning "
    "process in the mind and then provide the user with the answer."
)


def prompt_tool() -> str:
    """Deprecated tool — scheduled for removal.

    This tool is non-functional and retained only for backward compatibility.
    Do not call it. Place your final answer in <answer> tags in your response.

    Returns:
        str: Empty response.
    """
    return "continue"


PROMPT_TOOL = [FunctionTool(prompt_tool).get_openai_tool_schema()]

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TARGET_RE = re.compile(r"equals\s+(\d+)")
_NUMS_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
_NUMBER_RE = re.compile(r"\d+")


class CountdownInterface(TaskInterface):
    """Countdown reward and eval interface.

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
                 tmpl_kwargs=None, wandb_run=None, tokenizer=None):
        super().__init__(engine=engine, eval_prompt=eval_prompt,
                         eval_tools=eval_tools, tmpl_kwargs=tmpl_kwargs,
                         wandb_run=wandb_run)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.tokenizer = tokenizer
        self._eval_data = None  # lazy-loaded

    def _get_eval_data(self):
        if self._eval_data is None:
            self._eval_data = _load_countdown_test()
        return self._eval_data

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, completions: list[dict], raw_messages: list[str]) -> list[float]:
        """Per-sample equation correctness reward (1 if correct, else 0)."""
        rewards = []
        for completion, msg_json in zip(completions, raw_messages):
            msgs = json.loads(msg_json)
            target, nums = _extract_task_info(msgs)
            if target is None:
                rewards.append(0.0)
            else:
                rewards.append(_countdown_reward(completion, target, nums))
        return rewards

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_fn(self, global_step: int, n_samples: int = 4):
        """Roll out on Countdown test problems and log accuracy."""
        eval_data = self._get_eval_data()
        extra = {"chat_template_kwargs": self.tmpl_kwargs} if self.tmpl_kwargs else {}

        # Repeat each problem n_samples times
        prompts = []
        tools_list = []
        for target, nums, prompt in eval_data:
            for _ in range(n_samples):
                prompts.append(prompt)
                tools_list.append(PROMPT_TOOL)

        completions = self.engine.generate(
            prompts, max_tokens=2048, temperature=0.9,
            tools_list=tools_list, **extra,
        )

        total_correct = 0
        pass_at_n = 0
        table_rows = []
        for i, (target, nums, prompt) in enumerate(eval_data):
            sample_completions = completions[i * n_samples : (i + 1) * n_samples]
            scores = [_countdown_reward(c, target, nums) for c in sample_completions]
            n_correct = sum(scores)
            total_correct += n_correct
            passed = int(max(scores))
            pass_at_n += passed
            output = _format_completion(sample_completions[0], prompt, self.tokenizer, self.tmpl_kwargs)
            table_rows.append([
                i + 1, f"{nums}→{target}", output, n_correct, passed,
            ])

        avg_pass1 = total_correct / (len(eval_data) * n_samples)
        pass_n = pass_at_n / len(eval_data)
        print(
            f"  [Eval] Countdown avg pass@1: {avg_pass1:.1%} | "
            f"pass@{n_samples}: {pass_at_n}/{len(eval_data)} = {pass_n:.1%}"
        )

        if self.wandb_run is not None:
            self.wandb_run.log({
                "eval/countdown_pass@1": avg_pass1,
                f"eval/countdown_pass@{n_samples}": pass_n,
            }, step=global_step)
            table = wandb.Table(
                columns=["#", "problem", "model_outputs", "n_correct", "pass"],
                data=table_rows,
            )
            self.wandb_run.log({"eval/countdown_details": table}, step=global_step)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_countdown_test(n: int = 100):
    """Load a subset of Countdown test problems as list of (target, nums, prompt).

    The ``prompt`` field contains the original system + user messages from the
    dataset, so eval uses the exact same format as training.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceTB/Countdown-Task-GOLD", "test", split="test"
    )
    # Deterministic subset
    ds = ds.select(range(min(n, len(ds))))
    return [(row["target"], row["nums"], row["prompt"]) for row in ds]


def _make_user_prompt(target: int, nums: list[int]) -> str:
    """Build the user prompt for a Countdown problem.

    Matches the format used in HuggingFaceTB/Countdown-Task-GOLD.
    """
    return (
        f"Using the numbers {nums}, create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) and each "
        f"number can only be used once. Show your work in <think> </think> "
        f"tags. And return the final equation and answer in <answer> </answer> "
        f"tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
    )


def _extract_task_info(messages: list) -> tuple[int | None, list[int] | None]:
    """Extract (target, nums) from the user message text."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            t_match = _TARGET_RE.search(content)
            n_match = _NUMS_RE.search(content)
            if t_match and n_match:
                target = int(t_match.group(1))
                nums = [int(x.strip()) for x in n_match.group(1).split(",")]
                return target, nums
    return None, None


def _format_completion(completion: dict, prompt: list, tokenizer, tmpl_kwargs: dict) -> str:
    """Format a completion dict for display.

    If a tokenizer is available, uses the chat template to produce the exact
    string the model would see during training.  Otherwise falls back to
    showing raw content + tool_calls.
    """
    if tokenizer is not None:
        try:
            kwargs = dict(tmpl_kwargs) if tmpl_kwargs else {}
            prompt_text = tokenizer.apply_chat_template(
                prompt, tools=PROMPT_TOOL, tokenize=False,
                add_generation_prompt=True, **kwargs,
            )
            full_text = tokenizer.apply_chat_template(
                prompt + [completion], tools=PROMPT_TOOL, tokenize=False,
                add_generation_prompt=False, **kwargs,
            )
            return full_text[len(prompt_text):]
        except Exception:
            pass  # fall through to manual formatting

    # Fallback: manual assembly
    parts = []
    parts.append(completion.get("content", "") or "(empty)")
    if completion.get("tool_calls"):
        tc_strs = []
        for tc in completion["tool_calls"]:
            fn = tc.get("function", {})
            tc_strs.append("%s(%s)" % (fn.get("name", ""), fn.get("arguments", "")))
        parts.append("[tool_calls: " + ", ".join(tc_strs) + "]")
    return "\n".join(parts)


def _extract_answer_text(text: str) -> str | None:
    """Extract text inside <answer> tags, or return None."""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _safe_eval(expr: str) -> float | None:
    """Evaluate an arithmetic expression safely (only +, -, *, /, parens)."""
    # Strip whitespace and the "= result" suffix if present
    expr = expr.split("=")[0].strip()
    # Only allow digits, operators, parentheses, spaces, dots
    if not re.match(r"^[\d+\-*/().\s]+$", expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return float(result)
    except Exception:
        return None


def _countdown_reward(completion: dict, target: int, nums: list[int]) -> float:
    """1 if the equation is correct and uses valid numbers, 0 otherwise."""
    content = completion.get("content", "")
    if not content:
        return 0.0

    # Strip <think> blocks so we don't match <answer> inside thinking
    content_no_think = _THINK_RE.sub("", content).strip()
    answer_text = _extract_answer_text(content_no_think)
    if answer_text is None:
        return 0.0

    result = _safe_eval(answer_text)
    if result is None:
        return 0.0

    # Check result equals target (with small tolerance for float division)
    if abs(result - target) > 1e-6:
        return 0.0

    # Check that used numbers are a subset of the given numbers
    equation_part = answer_text.split("=")[0]
    used_nums = [int(x) for x in _NUMBER_RE.findall(equation_part)]
    available = Counter(nums)
    used = Counter(used_nums)
    for num, count in used.items():
        if count > available.get(num, 0):
            return 0.0

    return 1.0
