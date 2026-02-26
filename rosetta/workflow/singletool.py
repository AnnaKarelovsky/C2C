import json
from typing import Any, Dict, List, Optional, Tuple

from camel.models import BaseModelBackend

from rosetta.workflow.display import ConvLogger
from rosetta.workflow.basic_utils import msg_assistant, msg_tool, execute_tool, _clean_for_api
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.contextManage import ContextManager
from rosetta.workflow.camel_utils import model_run_sync, extract_logprobs


def run_with_tools(
    messages: List[Dict[str, Any]],
    model: BaseModelBackend,
    tools: List,
    tracker: Optional[InteractionTracker] = None,
    logger: Optional[ConvLogger] = None,
    ctx_manager: Optional[ContextManager] = None,
    max_iterations: int = 10,
) -> Tuple[str, List[Dict[str, Any]], Optional[InteractionTracker]]:
    """Run model with tools, handling multiple tool calls per round.

    Args:
        messages: Conversation messages so far (caller builds these).
            Typically starts with [msg_system(...), msg_user(...)].
        model: Model backend to use.
        tools: List of tool objects (FunctionTool or ToolWrapper).
            Must support get_function_name() and get_openai_tool_schema().
        tracker: Optional interaction tracker for recording.
        logger: Optional conversation logger for display.
        ctx_manager: Manages context compression and history config.
            Pass ContextManager(model, history_config=config) to control
            what gets added to chat history.
        max_iterations: Maximum number of LLM call rounds.

    Returns:
        Tuple of (final_text, messages, tracker).
    """
    tool_map = {t.get_function_name(): t for t in tools}
    tool_schemas = [t.get_openai_tool_schema() for t in tools]

    if tracker:
        tracker.register_tools(llm_id=0, tools=tools)

    if logger:
        logger.start()

    logger and logger.update(messages)

    rounds = 0
    for _ in range(max_iterations):
        rounds += 1
        response = model_run_sync(model, _clean_for_api(messages), tools=tool_schemas)
        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        # Build full assistant message with reasoning
        reasoning = getattr(assistant_msg, 'reasoning_content', None)
        messages.append(msg_assistant(assistant_msg.content, tool_calls or None, reasoning))
        logprobs = extract_logprobs(response)
        record_interaction(tracker, messages, llm_id=0, usage=response.usage, logprobs=logprobs)

        if not tool_calls:
            logger and logger.update(messages)
            break

        # Execute all tool calls
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_map, tool_call.function.name, args)
            messages.append(msg_tool(tool_call.id, result))
        logger and logger.update(messages)

        # Let ctx_manager apply history config and compression
        if ctx_manager:
            messages = ctx_manager.apply(messages)
            logger and logger.update(messages)

    if tracker:
        tracker.final_messages = messages
        tracker.rounds = rounds
    logger and logger.stop()
    return assistant_msg.content or "", messages, tracker
