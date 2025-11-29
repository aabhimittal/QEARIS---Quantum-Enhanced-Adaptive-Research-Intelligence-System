"""
Quantum-Inspired Optimizer - Task allocation optimization

TECHNIQUE: Simulated annealing (quantum-inspired)
PURPOSE: Optimal task-to-agent assignment
INSPIRED BY: Quantum annealing algorithms

This is a classical simulation of quantum optimization,
not actual quantum computing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for quantum-inspired optimization"""

    temperature: float = 1.0
    cooling_rate: float = 0.95
    iterations: int = 100
    min_temperature: float = 0.01


class QuantumOptimizer:
    """
    Quantum-inspired optimizer for task allocation

    MATHEMATICS:
    ------------
    Energy function: E = -Σ match(task_i, agent_j) × x_ij

    Where:
    - match() = compatibility score (0-1)
    - x_ij = 1 if task i assigned to agent j, 0 otherwise

    QUANTUM INSPIRATION:
    --------------------
    Simulated annealing mimics quantum tunneling:
    - High temperature: Explore widely (tunnel through barriers)
    - Low temperature: Settle into minimum (ground state)

    WHY this approach?
    - Avoids local minima
    - Finds good solutions quickly
    - Scales well with problem size
    """

    def __init__(self, config: Any = None):
        """
        Initialize optimizer

        PARAMETERS:
        -----------
        config: Application settings (uses defaults if None)
        """
        if config:
            self.temperature = getattr(config, "QUANTUM_TEMPERATURE", 1.0)
            self.cooling_rate = getattr(config, "COOLING_RATE", 0.95)
            self.iterations = getattr(config, "QUANTUM_ITERATIONS", 100)
        else:
            self.temperature = 1.0
            self.cooling_rate = 0.95
            self.iterations = 100

        self.min_temperature = 0.01

        logger.info(
            f"QuantumOptimizer initialized: "
            f"T={self.temperature}, rate={self.cooling_rate}, "
            f"iterations={self.iterations}"
        )

    async def optimize_assignment(
        self, tasks: List[Any], agents: List[Any]
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize task-to-agent assignment

        ALGORITHM: Simulated Annealing

        STEPS:
        1. Create random initial assignment
        2. Calculate initial energy
        3. For each iteration:
           a. Propose random swap
           b. Calculate new energy
           c. Accept or reject (Metropolis criterion)
           d. Cool temperature
        4. Return best assignment

        RETURNS:
        - Assignment matrix (tasks × agents)
        - Energy history for visualization

        COMPLEXITY: O(iterations × n_tasks × n_agents)
        """
        n_tasks = len(tasks)
        n_agents = len(agents)

        if n_tasks == 0 or n_agents == 0:
            logger.warning("Empty tasks or agents list")
            return np.zeros((n_tasks, n_agents)), []

        logger.info(f"Optimizing {n_tasks} tasks across {n_agents} agents")

        # Calculate compatibility matrix
        compatibility = self._calculate_compatibility(tasks, agents)

        # Initialize random assignment (one agent per task)
        assignment = self._initialize_assignment(n_tasks, n_agents)

        # Track best solution
        best_assignment = assignment.copy()
        best_energy = self._calculate_energy(assignment, compatibility)

        # Energy history for visualization
        energy_history = [best_energy]

        # Simulated annealing
        current_temp = self.temperature
        current_energy = best_energy

        for iteration in range(self.iterations):
            # Propose new assignment (swap)
            new_assignment = self._propose_swap(assignment.copy())
            new_energy = self._calculate_energy(new_assignment, compatibility)

            # Metropolis acceptance criterion
            delta_e = new_energy - current_energy

            if delta_e < 0:
                # Better solution - always accept
                accept = True
            else:
                # Worse solution - accept with probability
                accept_prob = np.exp(-delta_e / current_temp)
                accept = np.random.random() < accept_prob

            if accept:
                assignment = new_assignment
                current_energy = new_energy

                # Update best if better
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment.copy()

            # Cool down
            current_temp = max(self.min_temperature, current_temp * self.cooling_rate)

            energy_history.append(current_energy)

        logger.info(
            f"Optimization complete: "
            f"Initial energy={energy_history[0]:.4f}, "
            f"Final energy={best_energy:.4f}, "
            f"Improvement={(energy_history[0]-best_energy)/abs(energy_history[0]+1e-10)*100:.1f}%"
        )

        return best_assignment, energy_history

    def _calculate_compatibility(self, tasks: List[Any], agents: List[Any]) -> np.ndarray:
        """
        Calculate task-agent compatibility matrix

        FACTORS:
        - Domain match (agent specialization)
        - Priority alignment
        - Agent availability

        RETURNS:
        Matrix where [i,j] = compatibility of task i with agent j
        """
        n_tasks = len(tasks)
        n_agents = len(agents)
        compatibility = np.zeros((n_tasks, n_agents))

        for i, task in enumerate(tasks):
            for j, agent in enumerate(agents):
                score = 0.5  # Base compatibility

                # Domain match bonus
                if hasattr(task, "domain") and hasattr(agent, "agent"):
                    agent_obj = agent.agent if hasattr(agent, "agent") else agent
                    if hasattr(agent_obj, "specialization"):
                        if task.domain.lower() in agent_obj.specialization.lower():
                            score += 0.3
                        elif agent_obj.specialization.lower() == "general":
                            score += 0.1

                # Availability bonus
                if hasattr(agent, "agent"):
                    agent_obj = agent.agent if hasattr(agent, "agent") else agent
                    if hasattr(agent_obj, "is_available"):
                        if agent_obj.is_available():
                            score += 0.2

                compatibility[i, j] = min(1.0, score)

        return compatibility

    def _initialize_assignment(self, n_tasks: int, n_agents: int) -> np.ndarray:
        """
        Create initial random assignment

        CONSTRAINT: Each task assigned to exactly one agent
        (agents can have multiple tasks)
        """
        assignment = np.zeros((n_tasks, n_agents))

        for i in range(n_tasks):
            # Random agent for each task
            j = np.random.randint(n_agents)
            assignment[i, j] = 1

        return assignment

    def _calculate_energy(self, assignment: np.ndarray, compatibility: np.ndarray) -> float:
        """
        Calculate energy of assignment

        LOWER = BETTER (minimize)

        ENERGY = -Σ compatibility × assignment
                + λ × load_imbalance

        WHY negative compatibility?
        - High compatibility should give LOW energy
        - Annealing minimizes energy
        """
        # Compatibility contribution (negative = good)
        compatibility_score = -np.sum(compatibility * assignment)

        # Load balancing penalty
        load_per_agent = assignment.sum(axis=0)
        avg_load = load_per_agent.mean()
        load_imbalance = np.sum((load_per_agent - avg_load) ** 2)

        energy = compatibility_score + 0.1 * load_imbalance

        return energy

    def _propose_swap(self, assignment: np.ndarray) -> np.ndarray:
        """
        Propose new assignment by swapping

        OPERATION:
        Pick random task, reassign to different agent
        """
        n_tasks, n_agents = assignment.shape

        # Pick random task
        task_idx = np.random.randint(n_tasks)

        # Find current agent
        current_agent = np.argmax(assignment[task_idx])

        # Pick new random agent (different)
        new_agent = np.random.randint(n_agents)
        while new_agent == current_agent and n_agents > 1:
            new_agent = np.random.randint(n_agents)

        # Make swap
        assignment[task_idx, current_agent] = 0
        assignment[task_idx, new_agent] = 1

        return assignment

    def get_assignment_stats(
        self, assignment: np.ndarray, tasks: List[Any], agents: List[Any]
    ) -> Dict[str, Any]:
        """Get statistics about the assignment"""
        n_tasks, n_agents = assignment.shape
        load_per_agent = assignment.sum(axis=0)

        # Map assignments
        task_to_agent = {}
        for i in range(n_tasks):
            agent_idx = np.argmax(assignment[i])
            task_id = tasks[i].id if hasattr(tasks[i], "id") else f"task_{i}"
            agent_id = (
                agents[agent_idx].agent_id
                if hasattr(agents[agent_idx], "agent_id")
                else f"agent_{agent_idx}"
            )
            task_to_agent[task_id] = agent_id

        return {
            "total_tasks": n_tasks,
            "total_agents": n_agents,
            "load_distribution": load_per_agent.tolist(),
            "max_load": int(load_per_agent.max()),
            "min_load": int(load_per_agent.min()),
            "avg_load": float(load_per_agent.mean()),
            "assignments": task_to_agent,
        }
