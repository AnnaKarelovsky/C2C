"""Lightweight shims for tau_bench.types used by ref code."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


RESPOND_ACTION_NAME = "respond"


@dataclass
class Action:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    user_id: str
    instruction: str
    actions: List[Action] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    annotator: str = ""


@dataclass
class RewardActionInfo:
    r_actions: bool = True
    gt_data_hash: str = ""


@dataclass
class RewardOutputInfo:
    r_outputs: float = 1.0
    outputs: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RewardResult:
    reward: float = 0.0
    info: Any = None
    actions: List[Action] = field(default_factory=list)


@dataclass
class EnvInfo:
    task: Optional[Task] = None
    source: str = ""
    reward_info: Optional[RewardResult] = None
    user_cost: float = 0.0


@dataclass
class EnvResetResponse:
    observation: str = ""
    info: Optional[EnvInfo] = None


@dataclass
class EnvResponse:
    observation: str = ""
    reward: float = 0.0
    done: bool = False
    info: Optional[EnvInfo] = None
