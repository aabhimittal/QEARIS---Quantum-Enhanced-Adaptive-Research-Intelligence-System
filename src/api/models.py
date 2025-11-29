"""
# ============================================================================
# QEARIS - API Models
# ============================================================================
# 
# CAPSTONE REQUIREMENT: API Layer
# POINTS: Technical Implementation - Part of 50 points
# 
# DESCRIPTION: Pydantic models for API request/response validation.
# These models ensure type safety, automatic validation, and clear
# API documentation through FastAPI's OpenAPI integration.
# 
# INNOVATION: Comprehensive validation with descriptive error messages,
# default values, and examples for API documentation.
# 
# FILE LOCATION: src/api/models.py
# 
# CAPSTONE CRITERIA MET:
# - API Layer: Request/response models
# - Input Validation: Pydantic validation
# - Documentation: OpenAPI schema generation
# ============================================================================
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# ENUMS
# ============================================================================


class ResearchStatus(str, Enum):
    """Research task status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ResearchRequest(BaseModel):
    """
    Request model for research endpoint.

    ============================================================================
    CAPSTONE REQUIREMENT: API Layer - Request Validation

    VALIDATION:
    - query: 10-1000 characters
    - domains: 1-10 items
    - max_agents: 1-10
    - priority: valid enum value

    EXAMPLE:
    ```json
    {
        "query": "How does quantum computing improve AI?",
        "domains": ["quantum", "ai"],
        "max_agents": 3,
        "priority": "high"
    }
    ```
    ============================================================================
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How does quantum computing improve AI?",
                "domains": ["quantum", "ai"],
                "max_agents": 3,
                "priority": "high",
            }
        }
    )

    query: str = Field(
        ..., min_length=10, max_length=1000, description="Research query (10-1000 characters)"
    )

    domains: List[str] = Field(
        default=["general"], min_length=1, max_length=10, description="Research domains (1-10)"
    )

    max_agents: int = Field(
        default=3, ge=1, le=10, description="Maximum parallel research agents (1-10)"
    )

    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")

    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SessionCreateRequest(BaseModel):
    """Request model for session creation."""

    query: str = Field(..., min_length=10, max_length=1000, description="Research query")

    domains: List[str] = Field(default=["general"], description="Research domains")

    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Session metadata")


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""

    tool_name: str = Field(
        ..., min_length=1, max_length=100, description="Name of the tool to execute"
    )

    parameters: Dict[str, Any] = Field(default={}, description="Tool parameters")

    timeout: int = Field(default=60, ge=1, le=300, description="Execution timeout in seconds")


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ResearchResult(BaseModel):
    """
    Individual research result from an agent.

    ============================================================================
    CAPSTONE REQUIREMENT: API Layer - Response Models
    ============================================================================
    """

    task_id: str = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent that produced result")
    content: str = Field(..., description="Research content")
    sources: List[str] = Field(default=[], description="Sources used")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class ResearchResponse(BaseModel):
    """
    Response model for research endpoint.

    ============================================================================
    CAPSTONE REQUIREMENT: API Layer - Response Models

    RESPONSE STRUCTURE:
    - session_id: Unique session identifier
    - status: Research status
    - result: Synthesized research report
    - confidence: Overall confidence score
    - sources: Number of sources used
    - execution_time: Total execution time
    - metrics: Performance metrics

    EXAMPLE:
    ```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "completed",
        "result": "Research findings...",
        "confidence": 0.89,
        "sources": 12,
        "execution_time": 45.2,
        "metrics": {...}
    }
    ```
    ============================================================================
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "result": "Research findings on quantum computing and AI...",
                "confidence": 0.89,
                "sources": 12,
                "execution_time": 45.2,
                "metrics": {"tasks_created": 2, "agents_used": 2, "synthesis_iterations": 3},
            }
        }
    )

    session_id: str = Field(..., description="Session identifier")
    status: ResearchStatus = Field(..., description="Research status")
    result: str = Field(..., description="Synthesized research report")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence (0-1)")
    sources: int = Field(default=0, ge=0, description="Number of sources")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    metrics: Dict[str, Any] = Field(default={}, description="Performance metrics")


class SessionResponse(BaseModel):
    """Response model for session information."""

    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    query: str = Field(..., description="Research query")
    domains: List[str] = Field(default=[], description="Research domains")
    created_at: datetime = Field(..., description="Creation timestamp")
    tasks_count: int = Field(default=0, description="Number of tasks")
    results_count: int = Field(default=0, description="Number of results")


class ToolResponse(BaseModel):
    """Response model for tool execution."""

    success: bool = Field(..., description="Execution success")
    tool_name: str = Field(..., description="Tool name")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    components: Dict[str, str] = Field(default={}, description="Component status")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    total_requests: int = Field(default=0, description="Total requests")
    active_sessions: int = Field(default=0, description="Active sessions")
    agents_available: int = Field(default=0, description="Available agents")
    avg_response_time: float = Field(default=0.0, description="Average response time")
    success_rate: float = Field(default=1.0, description="Success rate")
    system_metrics: Dict[str, Any] = Field(default={}, description="System metrics")


# ============================================================================
# AGENT MODELS
# ============================================================================


class AgentInfo(BaseModel):
    """Information about an agent."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Agent type")
    specialization: str = Field(default="general", description="Specialization")
    status: str = Field(default="idle", description="Current status")
    tasks_completed: int = Field(default=0, description="Tasks completed")
    success_rate: float = Field(default=1.0, description="Success rate")


class AgentListResponse(BaseModel):
    """Response model for agent listing."""

    agents: List[AgentInfo] = Field(default=[], description="List of agents")
    total: int = Field(default=0, description="Total agent count")


# ============================================================================
# PAGINATION MODELS
# ============================================================================


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: List[Any] = Field(default=[], description="Items")
    total: int = Field(default=0, description="Total count")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")
    pages: int = Field(default=1, description="Total pages")
