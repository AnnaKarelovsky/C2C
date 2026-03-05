"""Tau2-bench interface for C2C evaluation pipeline.

Thin wrapper around tau2's registry and environment system.
Provides loading tasks, environments, tools, and system prompts.

Usage:
    from rosetta.benchmark.tau2.interface import (
        load_tasks, get_environment, get_tools_info, get_system_prompt,
        make_function_tools,
    )

    tasks = load_tasks("airline")
    env = get_environment("airline")
    tools_info = get_tools_info(env)
    system_prompt = get_system_prompt(env)
    function_tools = make_function_tools(env)
"""

from __future__ import annotations

from typing import Any, Dict, List

from tau2.registry import registry
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT



def load_tasks(domain: str, task_set: str | None = None) -> list[Task]:
    """Load tasks for a domain.

    Args:
        domain: One of airline, retail, telecom, telecom-workflow.
        task_set: Optional task set name. Defaults to the domain name.
            For telecom, can be telecom, telecom_full, telecom_small.

    Returns:
        List of Task objects.
    """
    task_set = task_set or domain
    loader = registry.get_tasks_loader(task_set)
    return loader()


def get_environment(domain: str) -> Environment:
    """Get a fresh environment for the domain."""
    constructor = registry.get_env_constructor(domain)
    return constructor()


def get_tools_info(env: Environment) -> list[dict]:
    """OpenAI tool schemas from an environment (agent-side)."""
    return [t.openai_schema for t in env.get_tools()]


def get_user_tools_info(env: Environment) -> list[dict] | None:
    """OpenAI tool schemas for user-side device tools, or None if unavailable."""
    try:
        return [t.openai_schema for t in env.get_user_tools()]
    except (ValueError, AttributeError):
        return None


def get_system_prompt(env: Environment) -> str:
    """Agent system prompt = instructions + policy (matches tau2 LLMAgent format)."""
    return SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=env.get_policy(),
    )


def get_user_instruction(task: Task) -> str:
    """User sim instruction from task's user_scenario."""
    return str(task.user_scenario)


class ToolWrapper:
    """Wraps a tau2 environment tool for use with run_with_tools.

    Provides the same interface as CAMEL FunctionTool:
    - get_function_name() -> str
    - get_openai_tool_schema() -> dict
    - func(**kwargs) -> str
    """

    def __init__(self, name: str, func, schema: dict):
        self.func = func
        self._name = name
        self._schema = schema

    def get_function_name(self) -> str:
        return self._name

    def get_openai_tool_schema(self) -> dict:
        return self._schema


def make_function_tools(env: Environment) -> list[ToolWrapper]:
    """Create ToolWrapper objects bound to an environment.

    Each tool calls env.make_tool_call() and env.sync_tools() to maintain
    proper state, matching tau2's internal agent protocol.

    Args:
        env: A tau2 Environment instance.

    Returns:
        List of ToolWrapper objects usable with run_with_tools.
    """
    wrappers = []
    for tau2_tool in env.get_tools():
        name = tau2_tool.name
        schema = tau2_tool.openai_schema

        def make_fn(n):
            def fn(**kwargs):
                result = env.make_tool_call(n, requestor="assistant", **kwargs)
                env.sync_tools()
                return Environment.to_json_str(result)

            return fn

        wrappers.append(ToolWrapper(name, make_fn(name), schema))
    return wrappers
