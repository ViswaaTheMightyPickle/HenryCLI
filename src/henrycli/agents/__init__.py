"""Agent implementations for HenryCLI."""

from .base import BaseAgent, TaskResult, AgentState
from .router import RouterAgent, TaskType, TaskAnalysis
from .specialist import (
    CodeAgent,
    ResearchAgent,
    WritingAgent,
    ReasoningAgent,
    create_agent_for_type,
)

__all__ = [
    "BaseAgent",
    "TaskResult",
    "AgentState",
    "RouterAgent",
    "TaskType",
    "TaskAnalysis",
    "CodeAgent",
    "ResearchAgent",
    "WritingAgent",
    "ReasoningAgent",
    "create_agent_for_type",
]
