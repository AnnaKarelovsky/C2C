"""Tau-bench interface for OPD training.

Implements the uniform ``TaskInterface`` for tau-bench / APIGen data:
tool-call accuracy as reward, qualitative generation as eval.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import wandb

from rosetta.benchmark.tau.evaluate import TERMINATE_TOOLS
from rosetta.benchmark.tau.types import Action
from rosetta.optimize.interface.base import TaskInterface, ToolEnvironment
from rosetta.workflow.basic_utils import execute_tool, msg_tool, msg_user, parse_tool_arguments


class TauInterface(TaskInterface):
    """Tau-bench reward and eval interface.

    Uniform API::

        interface.reward(completions, raw_messages) -> list[float]
        interface.eval_fn(global_step) -> None
        interface.make_env(task=..., domain=...) -> TauToolEnvironment

    Args:
        engine: :class:`~rosetta.optimize.train_utils.RolloutEngine`.
        eval_prompt: Message list (prompt portion only).
        eval_tools: Tool schemas for the eval sample.
        tmpl_kwargs: Extra chat-template kwargs.
        wandb_run: Optional wandb run for logging.
    """

    @staticmethod
    def full_tools(domain: str, tool_source: str = "tau") -> Optional[List[dict]]:
        """Return the full tool set for a tau-bench domain.

        Args:
            domain: ``"airline"`` or ``"retail"``.
            tool_source: ``"tau"`` for tau1 schemas, ``"tau2"`` for tau2
                environment schemas.

        Returns:
            List of tool schemas (OpenAI format).
        """
        if tool_source == "tau2":
            from rosetta.benchmark.tau2.interface import (
                get_environment,
                get_tools_info as get_tools_info_tau2,
            )
            env = get_environment(domain)
            return get_tools_info_tau2(env)

        from rosetta.benchmark.tau.interface import get_tools_info

        return get_tools_info(domain)

    @staticmethod
    def make_env(task, domain: str, *, user_sim=None) -> "ToolEnvironment":
        """Create a TauToolEnvironment for a single tau-bench task.

        Args:
            task: Task dataclass with instruction, actions, outputs.
            domain: ``"airline"`` or ``"retail"``.
            user_sim: Optional :class:`~rosetta.benchmark.tau.evaluate.UserSimulator`.
                Needed for multi-round training via :meth:`ToolEnvironment.advance`.

        Returns:
            A ready-to-use TauToolEnvironment (call ``env.reset()``
            before the first episode).
        """
        from rosetta.benchmark.tau.interface import (
            get_data_load_func,
            get_tools_info,
            get_tools_map,
        )

        return TauToolEnvironment(
            task=task,
            tools_info=get_tools_info(domain),
            tools_map=get_tools_map(domain),
            data_load_func=get_data_load_func(domain),
            user_sim=user_sim,
        )

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


# ---------------------------------------------------------------------------
# Tau-bench tool environment
# ---------------------------------------------------------------------------


class _TauTool:
    """Wraps a tau-bench tool to match the FunctionTool interface."""

    def __init__(self, name, schema, tool_cls, data, actions, env):
        self._name = name
        self._schema = schema
        self._cls = tool_cls
        self._data = data
        self._actions = actions  # shared mutable list
        self._env = env  # for setting done

    def get_function_name(self):
        return self._name

    def get_openai_tool_schema(self):
        return self._schema

    def func(self, **kwargs):
        self._actions.append(Action(name=self._name, kwargs=kwargs))
        result = self._cls.invoke(data=self._data, **kwargs)
        if self._name in TERMINATE_TOOLS:
            self._env.done = True
        return result


class TauToolEnvironment(ToolEnvironment):

    def __init__(self, task, tools_info, tools_map, data_load_func, user_sim=None):
        self.task = task
        self.tools_info = tools_info
        self.tools_map = tools_map
        self.data_load_func = data_load_func
        self.user_sim = user_sim
        self.tools: list = []
        self.done = False
        self.actions: list = []
        self.data = None

    def reset(self, data=None):
        self.data = data if data is not None else self.data_load_func()
        self.actions = []
        self.done = False
        self.tools = [
            _TauTool(
                schema["function"]["name"],
                schema,
                self.tools_map[schema["function"]["name"]],
                self.data,
                self.actions,
                self,
            )
            for schema in self.tools_info
        ]
        self._tool_map = {t.get_function_name(): t for t in self.tools}

    def advance(self, completion: dict, messages: list) -> bool:
        """Advance episode: execute tool calls or step the user sim.

        Args:
            completion: Assistant message dict.
            messages: Conversation so far (mutated in-place).

        Returns:
            ``True`` to continue the episode, ``False`` to discard it.
        """
        tool_calls = completion.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                args = parse_tool_arguments(args_raw)
                result = execute_tool(self._tool_map, name, args)
                messages.append(msg_tool(tc["id"], result))
            return not self.done

        # Text response → user sim
        if self.user_sim is None:
            return False
        text = completion.get("content", "")
        try:
            user_obs = self.user_sim.step(text)
        except Exception:
            return False
        if "###STOP###" in user_obs:
            return False
        messages.append(msg_user(user_obs))
        return True
