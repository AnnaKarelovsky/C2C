"""Shared utilities for cache-optimize training."""

import hashlib
import json
from typing import List, Optional


def tool_meta_key(tools: List[dict], system_message: Optional[dict] = None) -> str:
    """Deterministic hash for a (system_message, tools) combination.

    Args:
        tools: List of tool schemas.
        system_message: Optional system message dict.

    Returns:
        16-char hex string.
    """
    data = json.dumps({"system": system_message, "tools": tools}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]
