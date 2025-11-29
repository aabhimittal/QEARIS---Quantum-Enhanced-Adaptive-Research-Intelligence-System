"""
# ============================================================================
# QEARIS Services Module
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Sessions & Memory
# POINTS: Technical Implementation - 20 points
# 
# This module provides service layer components:
# 
# COMPONENTS:
# - Session Service: InMemorySessionService implementation
# - Task Manager: Pause/resume and state persistence
# 
# (Other services like RAG, Memory Bank, Context Manager are in core module)
# ============================================================================
"""

from src.services.session_service import (InMemorySessionService,
                                          SessionConfig, SessionState,
                                          create_session_service)
from src.services.task_manager import (ManagedTask, TaskManager, TaskState,
                                       create_task_manager)

__all__ = [
    # Session Service
    "InMemorySessionService",
    "SessionState",
    "SessionConfig",
    "create_session_service",
    # Task Manager
    "TaskManager",
    "TaskState",
    "ManagedTask",
    "create_task_manager",
]
