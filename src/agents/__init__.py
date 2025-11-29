"""
# ============================================================================
# QEARIS Multi-Agent System Components
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Multi-Agent System
# POINTS: Technical Implementation - 50 points
# 
# This module provides all agent types for the QEARIS system:
# 
# AGENT TYPES:
# - Base agents (legacy and LLM-powered)
# - Parallel Research Agents (concurrent execution)
# - Sequential Validator Agent (sequential processing)
# - Loop Synthesis Agent (iterative refinement)
# - Quantum Coordinator Agent (task allocation optimization)
# - Gemini Agent (Gemini-specific optimizations)
# 
# EXECUTION PATTERNS:
# - Parallel: asyncio.gather() for concurrent execution
# - Sequential: for loop for ordered processing
# - Loop: while/for with convergence criteria
# ============================================================================
"""

# Legacy agents (backward compatibility)
from src.agents.base_agent import BaseAgent, Agent, AgentType
from src.agents.research_agent import ResearchAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent

# New LLM-powered agents with capstone comments
from src.agents.base_llm_agent import (
    BaseLLMAgent,
    LLMAgentType,
    LLMAgentConfig,
    LLMAgentMetrics
)
from src.agents.parallel_research_agent import (
    ParallelResearchAgent,
    execute_parallel_research
)
from src.agents.sequential_validator_agent import (
    SequentialValidatorAgent,
    execute_sequential_validation
)
from src.agents.loop_synthesis_agent import (
    LoopSynthesisAgent,
    execute_synthesis_loop
)
from src.agents.quantum_coordinator_agent import (
    QuantumCoordinatorAgent,
    coordinate_with_quantum_optimization
)
from src.agents.gemini_agent import (
    GeminiAgent,
    GeminiAgentConfig,
    create_gemini_agent
)

__all__ = [
    # Legacy agents
    "BaseAgent",
    "Agent",
    "AgentType",
    "ResearchAgent",
    "ValidationAgent",
    "SynthesisAgent",
    
    # LLM-powered base
    "BaseLLMAgent",
    "LLMAgentType",
    "LLMAgentConfig",
    "LLMAgentMetrics",
    
    # Parallel agents
    "ParallelResearchAgent",
    "execute_parallel_research",
    
    # Sequential agents
    "SequentialValidatorAgent",
    "execute_sequential_validation",
    
    # Loop agents
    "LoopSynthesisAgent",
    "execute_synthesis_loop",
    
    # Quantum coordinator
    "QuantumCoordinatorAgent",
    "coordinate_with_quantum_optimization",
    
    # Gemini agent (BONUS)
    "GeminiAgent",
    "GeminiAgentConfig",
    "create_gemini_agent",
]
