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

def msg_assistant(content: str, tool_call=None, reasoning: str = None) -> Dict[str, Any]:
    """Build assistant message dict.

    Args:
        content: Assistant's response text.
        tool_call: Optional tool call object.
        reasoning: Optional reasoning/thinking content.

    Note:
        - `_reasoning`: Internal field, always preserved for tracking.
        - `reasoning_content`: API field, added when tool_call exists.
    """
    msg = {"role": "assistant", "content": content or ""}
    if tool_call:
        msg["tool_calls"] = [{
            "id": tool_call.id,
            "type": "function",
            "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
        }]
        if reasoning:
            msg["reasoning_content"] = reasoning
    if reasoning:
        msg["_reasoning"] = reasoning  # Always preserve internally
    return msg

def msg_tool(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

def execute_tool(tool_map: Dict, name: str, args: Dict) -> str:
    tool = tool_map.get(name)
    if tool is None:
        return f"Error: Unknown tool '{name}'"
    try:
        result = tool.func(**args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"

def _clean_for_api(messages: List[dict]) -> List[dict]:
    """Remove internal keys (starting with _) before sending to API."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]