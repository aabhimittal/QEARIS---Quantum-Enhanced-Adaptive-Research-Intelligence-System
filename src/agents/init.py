"""
Multi-agent system components

Includes:
- Base agent class
- Research agents (parallel)
- Validation agent (sequential)
- Synthesis agent (loop)
"""

from src.agents.base_agent import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "ValidationAgent",
    "SynthesisAgent"
]
