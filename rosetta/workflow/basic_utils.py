import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List


class ContentMode(Enum):
    """How to add content to chat history."""
    FULL = "full"
    NONE = "none"
    SUMMARIZED = "summarized"


@dataclass
class HistoryConfig:
    """Controls what gets added to chat history.

    Components:
        reasoning: model's reasoning/thinking content
        assistant: assistant's text response
        tool: tool execution result
        delay: number of rounds to wait before applying transformations
               (0 = immediate, 1 = apply to previous round, etc.)
    """
    reasoning: ContentMode = ContentMode.NONE
    assistant: ContentMode = ContentMode.FULL
    tool: ContentMode = ContentMode.FULL
    delay: int = 0

def msg_system(content: str) -> Dict[str, Any]:
    return {"role": "system", "content": content}

def msg_user(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}

def msg_assistant(content: str, tool_calls=None, reasoning: str = None) -> Dict[str, Any]:
    """Build assistant message dict.

    Args:
        content: Assistant's response text.
        tool_calls: Optional tool call object or list of tool call objects.
        reasoning: Optional reasoning/thinking content.

    Note:
        - `reasoning_content`: Always added when reasoning is provided.
          Chat templates (e.g. Qwen3) use this field to render ``<think>`` blocks.
        - `_reasoning`: Internal field, always preserved for tracking.
          Stripped by ``_clean_for_api`` before sending to LLM APIs.
    """
    msg = {"role": "assistant", "content": content or ""}
    if tool_calls:
        # Handle both single tool call and list of tool calls
        calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
        msg["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in calls]
    if reasoning:
        msg["reasoning_content"] = reasoning
        msg["_reasoning"] = reasoning
    return msg

def msg_tool(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

def execute_tool(tool_map: Dict, name: str, args: Dict, unknown_tool_fn=None) -> str:
    tool = tool_map.get(name)
    if tool is None:
        if unknown_tool_fn is not None:
            return unknown_tool_fn(name, args)
        return f"Error: Unknown tool '{name}'"
    try:
        result = tool.func(**args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"

def parse_tool_arguments(arguments):
    """Parse tool call arguments from string or dict.

    OpenAI-compatible APIs may return arguments as either a JSON string
    or an already-parsed dict.  This normalises to a dict.
    """
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments


def _clean_for_api(messages: List[dict]) -> List[dict]:
    """Remove internal keys (starting with _) before sending to API."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
