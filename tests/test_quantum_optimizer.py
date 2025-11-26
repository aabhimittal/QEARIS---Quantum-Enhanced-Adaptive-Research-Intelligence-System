"""
Tests for Quantum Optimizer
"""

import pytest
import numpy as np
from src.core.quantum_optimizer import QuantumOptimizer
from src.orchestrator.task_models import Task, Priority
from src.agents.base_agent import Agent, AgentType


@pytest.fixture
def optimizer():
    """Create optimizer instance"""
    from src.config import settings
    return QuantumOptimizer(settings)


@pytest.fixture
def sample_tasks():
    """Create sample tasks"""
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
    """Create sample agents"""
    return [
        Agent(
            name="Agent1",
            agent_type=AgentType.RESEARCHER,
            specialization="quantum"
        ),
        Agent(
            name="Agent2",
            agent_type=AgentType.RESEARCHER,
            specialization="ai"
        )
    ]


def test_energy_calculation(optimizer, sample_tasks, sample_agents):
    """Test energy calculation"""
    n_tasks = len(sample_tasks)
    n_agents = len(sample_agents)
    
    # Create random assignment
    assignment = np.zeros((n_tasks, n_agents))
    for i in range(n_tasks):
        assignment[i, i % n_agents] = 1
    
    energy = optimizer.calculate_energy(assignment, sample_tasks, sample_agents)
    
    assert isinstance(energy, float)
    assert energy >= 0


def test_optimization_convergence(optimizer, sample_tasks, sample_agents):
    """Test that optimization converges"""
    assignment, energy_history = optimizer.optimize_assignment(
        sample_tasks,
        sample_agents
    )
    
    # Check assignment shape
    assert assignment.shape == (len(sample_tasks), len(sample_agents))
    
    # Check energy decreases
    assert energy_history[-1] <= energy_history[0]
    
    # Check valid assignment (each task assigned to exactly one agent)
    assert np.all(assignment.sum(axis=1) == 1)


def test_quantum_fluctuation(optimizer):
    """Test quantum fluctuation creates valid assignments"""
    assignment = np.array([
        [1, 0],
        [0, 1],
        [1, 0]
    ])
    
    new_assignment = optimizer._quantum_fluctuation(assignment)
    
    # Check still valid
    assert new_assignment.shape == assignment.shape
    assert np.all(new_assignment.sum(axis=1) == 1)
    
    # Check something changed
    assert not np.array_equal(assignment, new_assignment)


def test_compatibility_calculation(optimizer):
    """Test compatibility scoring"""
    task = Task(description="Test", domain="quantum", priority=Priority.HIGH)
    agent = Agent(name="Test", agent_type=AgentType.RESEARCHER, specialization="quantum")
    
    compatibility = optimizer._calculate_compatibility(task, agent)
    
    assert 0 <= compatibility <= 1
    assert compatibility == 1.0  # Perfect match


@pytest.mark.parametrize("n_tasks,n_agents", [
    (3, 2),
    (5, 3),
    (10, 4)
])
def test_optimization_with_different_sizes(optimizer, n_tasks, n_agents):
    """Test optimization with different task/agent counts"""
    tasks = [
        Task(description=f"Task {i}", domain="test", priority=Priority.MEDIUM)
        for i in range(n_tasks)
    ]
    agents = [
        Agent(name=f"Agent {i}", agent_type=AgentType.RESEARCHER, specialization="test")
        for i in range(n_agents)
    ]
    
    assignment, energy_history = optimizer.optimize_assignment(tasks, agents)
    
    assert assignment.shape == (n_tasks, n_agents)
    assert len(energy_history) > 0
