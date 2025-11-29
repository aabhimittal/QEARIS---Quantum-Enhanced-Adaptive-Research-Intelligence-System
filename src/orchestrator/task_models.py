"""
Task Models - Data structures for orchestration

PURPOSE: Define all data models used across the system
DESIGN: Pydantic-style dataclasses for validation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Priority(Enum):
    """Task priority levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskState(Enum):
    """Task lifecycle states"""

    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Types of agents in the system"""

    RESEARCHER = "researcher"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


class MemoryType(Enum):
    """Types of memories stored"""

    EXPERIENCE = "experience"
    FACT = "fact"
    PROCEDURE = "procedure"
    OBSERVATION = "observation"


@dataclass(eq=False)
class Task:
    """
    Research task definition

    DESIGN:
    - Unique ID for tracking (immutable, used for hashing)
    - Priority for scheduling
    - State for lifecycle management (mutable, not used for hashing)
    - Metadata for flexibility (mutable, not used for hashing)

    Note: Hash is based on the immutable `id` field only to ensure
    consistency when the task is used in sets/dicts.
    """

    description: str
    domain: str
    priority: Priority = Priority.MEDIUM

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = field(default=TaskState.CREATED)
    assigned_agent: Optional[str] = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash based on immutable id field only"""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on id field"""
        if not isinstance(other, Task):
            return NotImplemented
        return self.id == other.id

    def assign_to(self, agent_id: str):
        """Assign task to agent"""
        self.assigned_agent = agent_id
        self.state = TaskState.ASSIGNED

    def start(self):
        """Mark task as in progress"""
        self.state = TaskState.IN_PROGRESS

    def complete(self):
        """Mark task as completed"""
        self.state = TaskState.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, reason: str = None):
        """Mark task as failed"""
        self.state = TaskState.FAILED
        self.completed_at = datetime.now()
        if reason:
            self.metadata["failure_reason"] = reason


@dataclass
class ResearchResult:
    """
    Result from research/validation/synthesis

    DESIGN:
    - Linked to task and agent
    - Contains result content
    - Tracks confidence and sources
    - Supports validation flags
    """

    task_id: str
    agent_id: str
    content: str
    sources: List[str]
    confidence: float

    validated: bool = False
    validation_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def task_description(self) -> str:
        """Get task description from metadata"""
        return self.metadata.get("task_description", "")


@dataclass
class ValidationResult:
    """
    Result from validation agent

    DESIGN:
    - References original result
    - Contains validation status
    - Provides recommendations
    """

    original_task_id: str
    validated: bool
    confidence: float

    source_score: float = 0.0
    content_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """
    Result from synthesis agent

    DESIGN:
    - Combines multiple research results
    - Tracks synthesis iterations
    - Contains final report
    """

    content: str
    confidence: float
    sources: List[str]

    iterations: int = 1
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """
    Agent metadata and state

    DESIGN: Lightweight data class for agent info
    """

    name: str
    agent_type: AgentType
    specialization: str
    max_concurrent_tasks: int = 2

    id: str = field(default_factory=lambda: f"agent_{datetime.now().timestamp()}")
    current_tasks: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if agent can accept more tasks"""
        return len(self.current_tasks) < self.max_concurrent_tasks

    def assign_task(self, task_id: str) -> bool:
        """Assign task to agent"""
        if self.is_available():
            self.current_tasks.append(task_id)
            return True
        return False

    def complete_task(self, task_id: str) -> bool:
        """Mark task as completed"""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
            return True
        return False


@dataclass
class Memory:
    """
    Memory item for experience storage

    DESIGN:
    - Typed memories (experience, fact, etc.)
    - Importance for prioritization
    - Agent attribution
    """

    content: str
    memory_type: MemoryType
    importance: float

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """
    Research session state

    DESIGN:
    - Tracks entire workflow
    - Contains all tasks and results
    - Provides session persistence
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    tasks: List[Task] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: Task):
        """Add task to session"""
        self.tasks.append(task)
        self.updated_at = datetime.now()

    def add_result(self, result: ResearchResult):
        """Add result to session"""
        self.results.append(result)
        self.updated_at = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """Get session status summary"""
        return {
            "session_id": self.session_id,
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks if t.state == TaskState.COMPLETED),
            "total_results": len(self.results),
            "avg_confidence": (
                sum(r.confidence for r in self.results) / len(self.results) if self.results else 0.0
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
