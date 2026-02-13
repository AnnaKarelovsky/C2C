"""Core evaluation logic for tau-bench multi-turn tool-calling tasks.

Provides UserSimulator (LLM-simulated user), solve_task (agent loop),
and calculate_reward (data-state hash + output matching).
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple, Union

from rosetta.benchmark.tau.types import Action, RESPOND_ACTION_NAME
from rosetta.workflow.basic_utils import (
    _clean_for_api,
    msg_assistant,
    msg_system,
    msg_tool,
    msg_user,
)
from rosetta.workflow.camel_utils import model_run_sync


# Hashing utilities (copied from tau-bench's base.py to avoid import chain issues)
ToHashable = Union[str, int, float, Dict[str, Any], List[Any]]


def to_hashable(item):
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(value) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


# Terminate tools for both airline and retail domains
TERMINATE_TOOLS = ["transfer_to_human_agents"]


class UserSimulator:
    """LLM-simulated user for tau-bench multi-turn evaluation.

    Wraps create_model() + model_run_sync() to replace litellm-based
    LLMUserSimulationEnv from tau-bench.
    """

    def __init__(self, model):
        self.model = model
        self.messages: List[Dict[str, Any]] = []

    def _build_system_prompt(self, instruction: Optional[str]) -> str:
        """Build system prompt matching tau-bench's LLMUserSimulationEnv."""
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return (
            f"You are a user interacting with an agent.{instruction_display}\n"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the "
            "information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the "
            "instruction. For example, if the agent asks for the order id but "
            "it is not mentioned in the instruction, do not make up an order "
            "id, just say you do not remember or have it.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as "
            "a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, "
            "use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick "
            "to the personalities in the instruction."
        )

    def reset(self, instruction: str) -> str:
        """Initialize and get first user message."""
        self.messages = [
            {"role": "system", "content": self._build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        response = model_run_sync(self.model, self.messages)
        content = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def step(self, agent_text: str) -> str:
        """Get next user message given agent's response."""
        self.messages.append({"role": "user", "content": agent_text})
        response = model_run_sync(self.model, self.messages)
        content = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        return content


def message_to_action(response) -> Action:
    """Convert model response to Action, matching tau-bench's protocol."""
    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        name = tc.function.name if hasattr(tc.function, "name") else tc["function"]["name"]
        args_raw = tc.function.arguments if hasattr(tc.function, "arguments") else tc["function"]["arguments"]
        if isinstance(args_raw, str):
            try:
                kwargs = json.loads(args_raw)
            except json.JSONDecodeError:
                kwargs = {}
        else:
            kwargs = args_raw
        return Action(name=name, kwargs=kwargs)
    content = msg.content or ""
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})


def solve_task(
    model,
    user_sim: UserSimulator,
    task,
    tools_info: List[Dict],
    tools_map: Dict[str, Any],
    wiki: str,
    data: Dict[str, Any],
    data_load_func=None,
    max_steps: int = 30,
) -> Dict[str, Any]:
    """Run the agent loop for one tau-bench task.

    Args:
        model: Agent model backend.
        user_sim: UserSimulator instance.
        task: Task dataclass with instruction, actions, outputs.
        tools_info: OpenAI tool schemas for the agent.
        tools_map: {tool_name: tool_class} for invoking tools.
        wiki: Wiki system prompt text.
        data: Mutable data dict shared across tool calls.
        data_load_func: Callable to load fresh data for reward computation.
        max_steps: Maximum number of agent turns.

    Returns:
        Dict with reward, messages, actions.
    """
    # Get first user message from simulator
    obs = user_sim.reset(task.instruction)

    messages = [msg_system(wiki), msg_user(obs)]
    actions: List[Action] = []

    for _ in range(max_steps):
        response = model_run_sync(model, _clean_for_api(messages), tools=tools_info)
        action = message_to_action(response)
        actions.append(action)

        msg = response.choices[0].message

        if action.name != RESPOND_ACTION_NAME:
            # Tool call
            tc = msg.tool_calls[0]
            tc_id = tc.id if hasattr(tc, "id") else tc["id"]

            # Build assistant message with tool_calls
            assistant_msg = msg_assistant(
                msg.content or "",
                tool_calls=msg.tool_calls,
            )
            messages.append(assistant_msg)

            # Invoke tool
            try:
                result = tools_map[action.name].invoke(
                    data=data, **action.kwargs
                )
            except Exception as e:
                result = f"Error: {e}"
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            messages.append(msg_tool(tc_id, result))

            # Check if this is a terminate tool
            if action.name in TERMINATE_TOOLS:
                break
        else:
            # Text response -> route to user sim
            text = action.kwargs["content"]
            messages.append(msg_assistant(text))

            user_obs = user_sim.step(text)
            if "###STOP###" in user_obs:
                break
            messages.append(msg_user(user_obs))

    # Calculate reward
    reward, info = calculate_reward(
        data=data,
        data_load_func=data_load_func,
        task=task,
        actions=actions,
        tools_map=tools_map,
    )

    return {
        "reward": reward,
        "messages": messages,
        "actions": [{"name": a.name, "kwargs": a.kwargs} for a in actions],
        "info": info,
    }


def calculate_reward(
    data: Dict[str, Any],
    data_load_func,
    task,
    actions: List[Action],
    tools_map: Dict[str, Any],
) -> Tuple[float, Dict]:
    """Calculate reward via data-state hash comparison + output matching.

    Extracted from tau-bench's Env.calculate_reward().

    Args:
        data: Agent's data state after the episode.
        data_load_func: Callable to load fresh data (if None, uses
            tools_map to find the domain's load_data via interface).
        task: Task with gold actions and expected outputs.
        actions: List of Actions taken by the agent.
        tools_map: {tool_name: tool_class} for replaying gold actions.

    Returns:
        (reward, info_dict)
    """
    # Hash agent's data state
    agent_hash = consistent_hash(to_hashable(data))

    # Reload fresh data and replay gold actions
    if data_load_func is None:
        # Import here to avoid circular imports
        from rosetta.benchmark.tau.interface import get_data_load_func
        # Infer domain from tools_map keys
        if "search_direct_flight" in tools_map:
            data_load_func = get_data_load_func("airline")
        else:
            data_load_func = get_data_load_func("retail")

    gt_data = data_load_func()
    for gold_action in task.actions:
        if gold_action.name in TERMINATE_TOOLS:
            continue
        if gold_action.name == RESPOND_ACTION_NAME:
            continue
        if gold_action.name in tools_map:
            try:
                tools_map[gold_action.name].invoke(
                    data=gt_data, **gold_action.kwargs
                )
            except Exception:
                pass

    gt_hash = consistent_hash(to_hashable(gt_data))

    reward = 1.0
    info = {"r_actions": agent_hash == gt_hash, "gt_data_hash": gt_hash}

    if not info["r_actions"]:
        reward = 0.0

    # Check outputs
    if task.outputs:
        outputs_check = {}
        for output in task.outputs:
            found = False
            for action in actions:
                if (
                    action.name == RESPOND_ACTION_NAME
                    and output.lower()
                    in action.kwargs.get("content", "").lower().replace(",", "")
                ):
                    found = True
                    break
            outputs_check[output] = found
            if not found:
                reward = 0.0
        info["r_outputs"] = all(outputs_check.values())
        info["outputs"] = outputs_check

    return reward, info
