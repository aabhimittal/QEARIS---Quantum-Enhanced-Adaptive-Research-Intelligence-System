"""
API Routes for QEARIS
Handles all HTTP endpoints
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..config import settings
from ..orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
from .schemas import (
    AgentEvaluationResponse,
    MetricsResponse,
    ResearchRequest,
    ResearchResponse,
    SessionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global orchestrator instance (in production, use dependency injection)
orchestrator: Optional[MultiAgentOrchestrator] = None


async def get_orchestrator() -> MultiAgentOrchestrator:
    """
    Dependency to get or create orchestrator instance

    DESIGN PATTERN: Singleton pattern for orchestrator
    Ensures single instance across requests
    """
    global orchestrator

    if orchestrator is None:
        logger.info("Initializing orchestrator...")
        orchestrator = await MultiAgentOrchestrator.create(settings)

    return orchestrator


@router.post("/research", response_model=ResearchResponse)
async def create_research_task(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator),
):
    """
    Create and execute research task

    WORKFLOW:
    1. Validate request
    2. Create research tasks
    3. Execute with quantum optimization
    4. Return results

    Parameters:
    -----------
    request: Research query and parameters
    background_tasks: For async cleanup
    orch: Orchestrator instance

    Returns:
    --------
    Complete research results with metrics
    """
    logger.info(f"Research request received: {request.query}")

    try:
        # Execute workflow
        results = await orch.execute_research_workflow(
            research_query=request.query,
            domains=request.domains or ["general"],
            max_agents=request.max_agents or settings.MAX_PARALLEL_AGENTS,
            enable_validation=request.enable_validation,
        )

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_task, results["session_id"])

        return ResearchResponse(
            session_id=results["session_id"],
            status="completed",
            results={
                "final_report": results["final_report"],
                "confidence": results["final_confidence"],
                "sources": results["total_sources"],
                "execution_time": results["workflow_time_seconds"],
            },
            metrics={
                "quantum_energy_reduction": results["quantum_optimization"]["energy_reduction"],
                "agents_utilized": len(results["quantum_optimization"]["assignments"]),
                "validation_pass_rate": (
                    sum(1 for v in results["validation_results"] if v["validated"])
                    / len(results["validation_results"])
                    if results["validation_results"]
                    else 0
                ),
            },
        )

    except Exception as e:
        logger.error(f"Research task failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, orch: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """
    Retrieve session information

    FEATURE: Long-running operations support
    Enables pause/resume functionality
    """
    try:
        session = await orch.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            status=session.metadata.get("status", "unknown"),
            tasks_total=len(session.tasks),
            tasks_completed=sum(1 for t in session.tasks if t.state == "completed"),
            results_count=len(session.results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 10, offset: int = 0, orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    List all sessions with pagination

    FEATURE: State management
    """
    try:
        sessions = await orch.list_sessions(limit=limit, offset=offset)

        return [
            SessionResponse(
                session_id=s.session_id,
                created_at=s.created_at,
                updated_at=s.updated_at,
                status=s.metadata.get("status", "unknown"),
                tasks_total=len(s.tasks),
                tasks_completed=sum(1 for t in s.tasks if t.state == "completed"),
                results_count=len(s.results),
            )
            for s in sessions
        ]

    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/pause")
async def pause_session(session_id: str, orch: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """
    Pause a running session

    FEATURE: Long-running operations
    Saves state for later resumption
    """
    try:
        await orch.pause_session(session_id)
        return {"status": "paused", "session_id": session_id}

    except Exception as e:
        logger.error(f"Failed to pause session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/resume")
async def resume_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator),
):
    """
    Resume a paused session

    FEATURE: Long-running operations
    Continues from saved checkpoint
    """
    try:
        results = await orch.resume_session(session_id)

        background_tasks.add_task(cleanup_task, session_id)

        return {"status": "resumed", "session_id": session_id, "results": results}

    except Exception as e:
        logger.error(f"Failed to resume session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/{session_id}", response_model=List[AgentEvaluationResponse])
async def get_evaluation(session_id: str, orch: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """
    Get agent evaluation for session

    FEATURE: Agent evaluation
    Performance metrics and recommendations
    """
    try:
        evaluations = await orch.evaluate_agents(session_id)

        return [
            AgentEvaluationResponse(
                agent_name=e.agent_name,
                agent_type=e.agent_type.value,
                overall_score=e.overall_score,
                grade=e.grade,
                metrics=[
                    {"name": m.name, "value": m.value, "unit": m.unit, "passed": m.passed}
                    for m in e.metrics
                ],
                recommendations=e.recommendations,
            )
            for e in evaluations
        ]

    except Exception as e:
        logger.error(f"Failed to get evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(orch: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """
    Get comprehensive system metrics

    FEATURE: Observability
    Metrics, logs, and traces
    """
    try:
        metrics = await orch.get_comprehensive_metrics()

        return MetricsResponse(
            timestamp=metrics["timestamp"],
            workflow_metrics=metrics["workflow"],
            agent_metrics=metrics["agents"],
            quantum_metrics=metrics["quantum"],
            rag_metrics=metrics["rag"],
            memory_metrics=metrics["memory"],
            mcp_metrics=metrics["mcp"],
            quality_metrics=metrics["quality"],
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def cleanup_task(session_id: str):
    """
    Background task for cleanup

    Runs after request completion to:
    - Compact memory
    - Archive old sessions
    - Free resources
    """
    await asyncio.sleep(60)  # Wait before cleanup
    logger.info(f"Running cleanup for session: {session_id}")
    # Add cleanup logic here
