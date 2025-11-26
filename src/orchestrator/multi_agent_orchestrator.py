"""
Multi-Agent Orchestrator with Gemini Integration
Coordinates all agents and manages workflow
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from ..config import settings
from ..core.quantum_optimizer import QuantumOptimizer
from ..core.rag_system import RAGSystem
from ..core.memory_bank import MemoryBank
from ..core.mcp_server import MCPServer
from ..agents.research_agent import ResearchAgent
from ..agents.validation_agent import ValidationAgent
from ..agents.synthesis_agent import SynthesisAgent
from .task_models import Task, TaskState, Priority, SessionState

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent research workflow
    
    GEMINI INTEGRATION:
    - Uses Gemini 1.5 Pro for agent reasoning
    - Supports function calling for tool use
    - Handles context management
    
    KEY FEATURES:
    - Quantum-optimized task allocation
    - Parallel, sequential, and loop execution
    - Session persistence
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        quantum_optimizer: QuantumOptimizer,
        rag_system: RAGSystem,
        memory_bank: MemoryBank,
        mcp_server: MCPServer,
        gemini_model: genai.GenerativeModel
    ):
        self.quantum_optimizer = quantum_optimizer
        self.rag_system = rag_system
        self.memory_bank = memory_bank
        self.mcp_server = mcp_server
        self.gemini_model = gemini_model
        
        # Agent pool
        self.agents: List[Any] = []
        
        # State tracking
        self.sessions: Dict[str, SessionState] = {}
        self.current_session: Optional[SessionState] = None
        
        logger.info("🎯 Multi-Agent Orchestrator initialized with Gemini")
    
    @classmethod
    async def create(cls, config: Any) -> "MultiAgentOrchestrator":
        """
        Factory method to create orchestrator
        
        INITIALIZATION:
        1. Configure Gemini API
        2. Initialize core components
        3. Create agent pool
        4. Load tools
        """
        # Configure Gemini
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Create model with function calling
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": config.GEMINI_TEMPERATURE,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": config.GEMINI_MAX_TOKENS,
            }
        )
        
        # Initialize core components
        quantum_optimizer = QuantumOptimizer(config)
        rag_system = await RAGSystem.create(config)
        memory_bank = MemoryBank(config)
        mcp_server = MCPServer(config)
        
        # Register MCP tools
        await cls._register_tools(mcp_server, rag_system, memory_bank)
        
        # Create orchestrator
        orchestrator = cls(
            quantum_optimizer=quantum_optimizer,
            rag_system=rag_system,
            memory_bank=memory_bank,
            mcp_server=mcp_server,
            gemini_model=model
        )
        
        # Create agent pool
        await orchestrator._create_agents(config)
        
        return orchestrator
    
    @staticmethod
    async def _register_tools(
        mcp_server: MCPServer,
        rag_system: RAGSystem,
        memory_bank: MemoryBank
    ):
        """Register all MCP tools"""
        
        # Knowledge base search
        async def search_kb(query: str, top_k: int = 3):
            results = await rag_system.retrieve(query, top_k)
            return {
                "results": [
                    {
                        "content": r["content"],
                        "similarity": r["similarity"],
                        "source": r["metadata"]["source"]
                    }
                    for r in results
                ]
            }
        
        await mcp_server.register_tool(
            name="search_knowledge_base",
            description="Search RAG knowledge base",
            handler=search_kb
        )
        
        # Memory retrieval
        async def search_memory(query: str, top_k: int = 3):
            memories = await memory_bank.retrieve_memories(query, top_k)
            return {
                "memories": [
                    {
                        "content": m.content,
                        "type": m.memory_type.value,
                        "importance": m.importance
                    }
                    for m in memories
                ]
            }
        
        await mcp_server.register_tool(
            name="search_memory",
            description="Search memory bank",
            handler=search_memory
        )
        
        logger.info("✅ MCP tools registered")
    
    async def _create_agents(self, config: Any):
        """Create agent pool with Gemini models"""
        
        # Research agents (parallel)
        for i in range(config.MAX_PARALLEL_AGENTS):
            agent = ResearchAgent(
                agent_id=f"researcher_{i}",
                gemini_model=self.gemini_model,
                mcp_server=self.mcp_server,
                rag_system=self.rag_system,
                memory_bank=self.memory_bank
            )
            self.agents.append(agent)
        
        # Validation agent (sequential)
        validator = ValidationAgent(
            agent_id="validator",
            gemini_model=self.gemini_model,
            mcp_server=self.mcp_server
        )
        self.agents.append(validator)
        
        # Synthesis agent (loop)
        synthesizer = SynthesisAgent(
            agent_id="synthesizer",
            gemini_model=self.gemini_model,
            mcp_server=self.mcp_server
        )
        self.agents.append(synthesizer)
        
        logger.info(f"✅ Created {len(self.agents)} agents")
    
    async def execute_research_workflow(
        self,
        research_query: str,
        domains: List[str],
        max_agents: int = 4,
        enable_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete research workflow
        
        WORKFLOW:
        1. Create tasks for each domain
        2. Quantum-optimize assignments
        3. Execute parallel research
        4. Sequential validation
        5. Loop synthesis
        6. Return results
        
        GEMINI USAGE:
        - Each agent uses Gemini for reasoning
        - Function calling for tool usage
        - Context management across steps
        """
        workflow_start = datetime.now()
        
        # Create session
        session = SessionState(
            metadata={
                "query": research_query,
                "domains": domains,
                "status": "running"
            }
        )
        self.sessions[session.session_id] = session
        self.current_session = session
        
        logger.info(f"🚀 Starting workflow: {research_query}")
        
        try:
            # Phase 1: Create research tasks
            tasks = []
            for domain in domains:
                task = Task(
                    description=f"{research_query} - Focus on {domain}",
                    domain=domain,
                    priority=Priority.HIGH
                )
                tasks.append(task)
                session.tasks.append(task)
            
            logger.info(f"📋 Created {len(tasks)} tasks")
            
            # Phase 2: Quantum optimization
            research_agents = [a for a in self.agents if isinstance(a, ResearchAgent)][:max_agents]
            
            assignment_matrix, energy_history = await self.quantum_optimizer.optimize_assignment(
                tasks=tasks,
                agents=research_agents
            )
            
            assignments = self._apply_assignments(assignment_matrix, tasks, research_agents)
            
            logger.info("✅ Quantum optimization complete")
            
            # Phase 3: Parallel research execution
            research_results = await self._execute_parallel_research(tasks, assignments)
            
            logger.info(f"✅ Research complete: {len(research_results)} results")
            
            # Phase 4: Sequential validation (if enabled)
            validation_results = []
            if enable_validation:
                validator = next((a for a in self.agents if isinstance(a, ValidationAgent)), None)
                if validator:
                    validation_results = await self._execute_sequential_validation(
                        research_results,
                        validator
                    )
                    logger.info(f"✅ Validation complete: {len(validation_results)} validated")
            
            # Phase 5: Loop synthesis
            synthesizer = next((a for a in self.agents if isinstance(a, SynthesisAgent)), None)
            if synthesizer:
                final_report = await self._execute_synthesis_loop(
                    research_results,
                    validation_results,
                    synthesizer
                )
                logger.info("✅ Synthesis complete")
            else:
                final_report = research_results[0] if research_results else None
            
            # Calculate metrics
            workflow_time = (datetime.now() - workflow_start).total_seconds()
            
            # Update session
            session.metadata["status"] = "completed"
            session.updated_at = datetime.now()
            
            # Compile results
            results = {
                "session_id": session.session_id,
                "research_query": research_query,
                "tasks_created": len(tasks),
                "workflow_time_seconds": workflow_time,
                "quantum_optimization": {
                    "energy_history": energy_history,
                    "final_energy": energy_history[-1] if energy_history else 0,
                    "energy_reduction": energy_history[0] - energy_history[-1] if len(energy_history) > 1 else 0,
                    "assignments": {
                        task.id: agent.agent_id
                        for task, agent in assignments.items()
                    }
                },
                "research_results": [
                    {
                        "task": r.task_description,
                        "confidence": r.confidence,
                        "sources": r.sources
                    }
                    for r in research_results
                ],
                "validation_results": [
                    {
                        "validated": v.validated,
                        "confidence": v.confidence
                    }
                    for v in validation_results
                ],
                "final_report": final_report.content if final_report else "",
                "final_confidence": final_report.confidence if final_report else 0.0,
                "total_sources": len(set(
                    source
                    for r in research_results
                    for source in r.sources
                ))
            }
            
            logger.info(f"🎉 Workflow complete in {workflow_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}", exc_info=True)
            session.metadata["status"] = "failed"
            session.metadata["error"] = str(e)
            raise
    
    def _apply_assignments(self, matrix, tasks, agents):
        """Apply quantum-optimized assignments"""
        assignments = {}
        for i, task in enumerate(tasks):
            agent_idx = matrix[i].argmax()
            agent = agents[agent_idx]
            assignments[task] = agent
            task.assigned_agent = agent.agent_id
            task.state = TaskState.ASSIGNED
        return assignments
    
    async def _execute_parallel_research(self, tasks, assignments):
        """Execute research tasks in parallel"""
        coroutines = [
            agent.execute_task(task)
            for task, agent in assignments.items()
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        valid_results = [r for r in results if not isinstance(r, Exception)]
        return valid_results
    
    async def _execute_sequential_validation(self, results, validator):
        """Execute validation sequentially"""
        validated = []
        for result in results:
            validation = await validator.execute_task(result)
            validated.append(validation)
        return validated
    
    async def _execute_synthesis_loop(self, research_results, validation_results, synthesizer):
        """Execute synthesis with iterative refinement"""
        max_iterations = settings.SYNTHESIS_ITERATIONS
        quality_threshold = 0.85
        
        current_synthesis = None
        for iteration in range(max_iterations):
            current_synthesis = await synthesizer.execute_task({
                "research_results": research_results,
                "validation_results": validation_results,
                "iteration": iteration
            })
            
            if current_synthesis.confidence >= quality_threshold:
                break
        
        return current_synthesis
    
    # Additional methods for session management, evaluation, metrics...
    # (Implement as needed based on routes.py requirements)
