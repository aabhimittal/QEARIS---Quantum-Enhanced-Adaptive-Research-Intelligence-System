"""
Tests for Quantum Optimizer
"""

import pytest
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def optimizer():
    """Create optimizer instance"""
    from src.core.quantum_optimizer import QuantumOptimizer
    return QuantumOptimizer()


@pytest.fixture
def sample_tasks():
    """Create sample tasks"""
    from src.orchestrator.task_models import Task, Priority
    return [
        Task(
            description="Research quantum computing",
            domain="quantum",
            priority=Priority.HIGH
        ),
        Task(
            description="Research AI systems",
            domain="ai",
            priority=Priority.MEDIUM
        ),
        Task(
            description="Research NLP",
            domain="nlp",
            priority=Priority.HIGH
        )
    ]


@pytest.fixture
def sample_agents():
    """Create sample mock agents"""
    agents = []
    for i, spec in enumerate(["quantum", "ai"]):
        agent = Mock()
        agent.agent_id = f"agent_{i}"
        agent.agent = Mock()
        agent.agent.specialization = spec
        agent.agent.is_available = Mock(return_value=True)
        agents.append(agent)
    return agents


@pytest.mark.asyncio
async def test_optimization_convergence(optimizer, sample_tasks, sample_agents):
    """Test that optimization converges"""
    assignment, energy_history = await optimizer.optimize_assignment(
        sample_tasks,
        sample_agents
    )
    
    # Check assignment shape
    assert assignment.shape == (len(sample_tasks), len(sample_agents))
    
    # Check energy history exists
    assert len(energy_history) > 0
    
    # Check valid assignment (each task assigned to exactly one agent)
    assert np.all(assignment.sum(axis=1) == 1)


@pytest.mark.asyncio
async def test_empty_inputs(optimizer):
    """Test handling of empty inputs"""
    assignment, energy_history = await optimizer.optimize_assignment([], [])
    
    assert assignment.shape == (0, 0)
    assert energy_history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("n_tasks,n_agents", [
    (3, 2),
    (5, 3),
    (10, 4)
])
async def test_optimization_with_different_sizes(optimizer, n_tasks, n_agents):
    """Test optimization with different task/agent counts"""
    from src.orchestrator.task_models import Task, Priority
    
    tasks = [
        Task(description=f"Task {i}", domain="test", priority=Priority.MEDIUM)
        for i in range(n_tasks)
    ]
    
    agents = []
    for i in range(n_agents):
        agent = Mock()
        agent.agent_id = f"agent_{i}"
        agent.agent = Mock()
        agent.agent.specialization = "test"
        agent.agent.is_available = Mock(return_value=True)
        agents.append(agent)
    
    assignment, energy_history = await optimizer.optimize_assignment(tasks, agents)
    
    assert assignment.shape == (n_tasks, n_agents)
    assert len(energy_history) > 0


def test_assignment_stats(optimizer, sample_tasks, sample_agents):
    """Test assignment statistics calculation"""
    n_tasks = len(sample_tasks)
    n_agents = len(sample_agents)
    
    # Create test assignment
    assignment = np.zeros((n_tasks, n_agents))
    for i in range(n_tasks):
        assignment[i, i % n_agents] = 1
    
    stats = optimizer.get_assignment_stats(assignment, sample_tasks, sample_agents)
    
    assert 'total_tasks' in stats
    assert 'total_agents' in stats
    assert stats['total_tasks'] == n_tasks
    assert stats['total_agents'] == n_agents
