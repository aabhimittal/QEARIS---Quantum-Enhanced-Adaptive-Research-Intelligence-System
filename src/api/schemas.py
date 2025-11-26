"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ResearchRequest(BaseModel):
    """
    Research request schema
    
    VALIDATION:
    - Query must be non-empty
    - Domains must be valid
    - Agent count within limits
    """
    query: str = Field(..., min_length=10, description="Research question")
    domains: Optional[List[str]] = Field(None, description="Research domains")
    max_agents: Optional[int] = Field(4, ge=1, le=10, description="Max parallel agents")
    enable_validation: bool = Field(True, description="Enable result validation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does quantum computing improve AI systems?",
                "domains": ["quantum", "ai"],
                "max_agents": 4,
                "enable_validation": True
            }
        }


class ResearchResponse(BaseModel):
    """Research response schema"""
    session_id: str
    status: str
    results: Dict[str, Any]
    metrics: Dict[str, Any]


class SessionResponse(BaseModel):
    """Session information schema"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    status: str
    tasks_total: int
    tasks_completed: int
    results_count: int


class AgentEvaluationResponse(BaseModel):
    """Agent evaluation schema"""
    agent_name: str
    agent_type: str
    overall_score: float
    grade: str
    metrics: List[Dict[str, Any]]
    recommendations: List[str]


class MetricsResponse(BaseModel):
    """System metrics schema"""
    timestamp: str
    workflow_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    quantum_metrics: Dict[str, Any]
    rag_metrics: Dict[str, Any]
    memory_metrics: Dict[str, Any]
    mcp_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    environment: str
    model: str
