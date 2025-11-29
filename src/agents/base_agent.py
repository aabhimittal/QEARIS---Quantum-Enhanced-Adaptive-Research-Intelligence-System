"""
Base Agent - Foundation for all agent types

DESIGN: Template pattern for agent implementation
COMPONENTS: Gemini integration, MCP tools, RAG, memory
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system"""
    RESEARCHER = "researcher"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


@dataclass
class Agent:
    """
    Agent metadata and state
    
    DESIGN: Lightweight data class for agent info
    """
    name: str
    agent_type: AgentType
    specialization: str
    max_concurrent_tasks: int = 2
    
    id: str = field(default_factory=lambda: f"agent_{datetime.now().timestamp()}")
    current_tasks: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        """Check if agent can accept more tasks"""
        return len(self.current_tasks) < self.max_concurrent_tasks
    
    def assign_task(self, task_id: str) -> bool:
        """Assign task to agent"""
        if self.is_available():
            self.current_tasks.append(task_id)
            return True
        return False
    
    def complete_task(self, task_id: str) -> bool:
        """Mark task as completed"""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
            return True
        return False


class BaseAgent:
    """
    Base class for all agents
    
    ARCHITECTURE:
    -------------
    - Gemini model for reasoning
    - MCP server for tool execution
    - RAG system for knowledge retrieval
    - Memory bank for learning
    
    PATTERNS:
    ---------
    - Template method for execute_task
    - Strategy for different agent types
    - Observer for metrics collection
    
    LIFECYCLE:
    ----------
    1. Initialize with dependencies
    2. Receive task
    3. Execute (subclass-specific)
    4. Store experience
    5. Update metrics
    """
    
    def __init__(
        self,
        agent_id: str,
        gemini_model: Any,
        mcp_server: Any = None,
        rag_system: Any = None,
        memory_bank: Any = None
    ):
        """
        Initialize base agent
        
        PARAMETERS:
        -----------
        agent_id: Unique identifier
        gemini_model: Configured Gemini model for reasoning
        mcp_server: MCP server for tool execution
        rag_system: RAG system for knowledge retrieval
        memory_bank: Memory bank for experience storage
        """
        self.agent_id = agent_id
        self.gemini_model = gemini_model
        self.mcp_server = mcp_server
        self.rag_system = rag_system
        self.memory_bank = memory_bank
        
        # Create agent metadata
        self.agent = Agent(
            name=agent_id,
            agent_type=self._determine_agent_type(),
            specialization=self._get_specialization(),
            id=agent_id
        )
        
        # Metrics tracking
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'success_rate': 1.0
        }
        
        logger.info(f"Agent {agent_id} initialized")
    
    def _determine_agent_type(self) -> AgentType:
        """Determine agent type from class name"""
        class_name = self.__class__.__name__.lower()
        if 'research' in class_name:
            return AgentType.RESEARCHER
        elif 'valid' in class_name:
            return AgentType.VALIDATOR
        elif 'synth' in class_name:
            return AgentType.SYNTHESIZER
        return AgentType.RESEARCHER
    
    def _get_specialization(self) -> str:
        """Get agent specialization"""
        return getattr(self, 'specialization', 'general')
    
    async def execute_task(self, task: Any) -> Any:
        """
        Execute task (template method)
        
        OVERRIDE: Subclasses must implement
        
        PATTERN: Template Method
        - Base class defines structure
        - Subclass provides implementation
        """
        raise NotImplementedError("Subclasses must implement execute_task")
    
    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute MCP tool
        
        PROCESS:
        1. Validate tool exists
        2. Execute via MCP server
        3. Return result
        
        ERROR HANDLING:
        - Log failures
        - Return None on error
        """
        if not self.mcp_server:
            logger.warning(f"No MCP server configured for {self.agent_id}")
            return None
        
        try:
            result = await self.mcp_server.execute_tool(tool_name, parameters)
            return result.result if hasattr(result, 'result') else result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
            return None
    
    async def retrieve_memories(self, query: str, top_k: int = 3) -> List[Any]:
        """
        Retrieve relevant memories
        
        PURPOSE: Learn from past experiences
        
        PROCESS:
        1. Search memory bank
        2. Return top-k relevant memories
        """
        if not self.memory_bank:
            return []
        
        try:
            memories = await self.memory_bank.retrieve_memories(query, top_k)
            return memories
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            return []
    
    async def store_experience(self, content: str, importance: float = 0.5) -> bool:
        """
        Store experience in memory bank
        
        PURPOSE: Learn from actions
        
        PARAMETERS:
        -----------
        content: Experience description
        importance: How important (0.0-1.0)
        
        WHY store?
        - Future task improvement
        - Pattern learning
        - Error avoidance
        """
        if not self.memory_bank:
            return False
        
        try:
            await self.memory_bank.store_memory(
                content=content,
                agent_id=self.agent_id,
                importance=importance
            )
            return True
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}")
            return False
    
    def update_metrics(self, execution_time: float, success: bool = True):
        """
        Update agent metrics
        
        TRACKED:
        - Tasks completed/failed
        - Execution time (total, average)
        - Success rate
        
        WHY track?
        - Performance monitoring
        - Load balancing
        - Quality assurance
        """
        if success:
            self.metrics['tasks_completed'] += 1
        else:
            self.metrics['tasks_failed'] += 1
        
        self.metrics['total_execution_time'] += execution_time
        
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        if total_tasks > 0:
            self.metrics['avg_execution_time'] = (
                self.metrics['total_execution_time'] / total_tasks
            )
            self.metrics['success_rate'] = (
                self.metrics['tasks_completed'] / total_tasks
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            'agent_id': self.agent_id,
            'agent_type': self.agent.agent_type.value,
            'available': self.agent.is_available()
        }
