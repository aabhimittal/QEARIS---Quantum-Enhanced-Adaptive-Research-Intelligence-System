"""
Orchestration components for multi-agent coordination

Includes:
- Multi-agent orchestrator
- Task models and data structures
- Workflow coordination
"""

from src.orchestrator.task_models import (
    Task,
    Agent,
    ResearchResult,
    SessionState,
    Priority,
    TaskState,
    AgentType,
    MemoryType,
    Memory
)

# Lazy import to avoid circular dependencies
def get_multi_agent_orchestrator():
    from src.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
    return MultiAgentOrchestrator

__all__ = [
    "get_multi_agent_orchestrator",
    "Task",
    "Agent",
    "ResearchResult",
    "SessionState",
    "Priority",
    "TaskState",
    "AgentType",
    "MemoryType",
    "Memory"
]
