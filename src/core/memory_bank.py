"""
Memory Bank - Experience storage for agent learning

PURPOSE: Store and retrieve agent experiences
DESIGN: Semantic memory with importance weighting
ENABLES: Learning from past interactions
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib

from src.orchestrator.task_models import Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    Memory storage for agent experiences
    
    ARCHITECTURE:
    -------------
    1. Short-term memory: Recent experiences
    2. Long-term memory: Important experiences
    3. Retrieval: Semantic similarity search
    4. Decay: Old/unimportant memories fade
    
    WHY memory?
    -----------
    - Learn from past successes/failures
    - Avoid repeating mistakes
    - Build domain expertise
    - Improve over time
    
    INSPIRED BY:
    - Human episodic memory
    - Transformer memory mechanisms
    - Experience replay in RL
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize memory bank
        
        PARAMETERS:
        -----------
        config: Application settings
        """
        if config:
            self.retention_days = getattr(config, 'MEMORY_RETENTION_DAYS', 30)
            self.max_items = getattr(config, 'MAX_MEMORY_ITEMS', 1000)
        else:
            self.retention_days = 30
            self.max_items = 1000
        
        # Memory storage
        self.memories: List[Memory] = []
        
        # Simple embedding dimension
        self.embedding_dim = 256
        
        # Embeddings for similarity search
        self._embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(
            f"MemoryBank initialized: "
            f"retention={self.retention_days}d, "
            f"max_items={self.max_items}"
        )
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EXPERIENCE,
        importance: float = 0.5,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store new memory
        
        PROCESS:
        1. Create memory object
        2. Generate embedding
        3. Store in bank
        4. Cleanup if at capacity
        
        RETURNS: Memory ID
        """
        # Handle memory_type as string (for backward compatibility)
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                memory_type = MemoryType.EXPERIENCE
        
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Generate embedding
        embedding = self._create_embedding(content)
        self._embeddings[memory.id] = embedding
        
        # Store
        self.memories.append(memory)
        
        # Cleanup if at capacity
        if len(self.memories) > self.max_items:
            await self._cleanup()
        
        logger.debug(
            f"Stored memory: type={memory_type.value}, "
            f"importance={importance:.2f}, "
            f"agent={agent_id}"
        )
        
        return memory.id
    
    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 3,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """
        Retrieve relevant memories
        
        PROCESS:
        1. Filter by type/agent/importance
        2. Calculate similarity to query
        3. Return top-k most relevant
        
        RETURNS: List of Memory objects
        """
        if not self.memories:
            return []
        
        # Filter memories
        candidates = self.memories
        
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        if agent_id:
            candidates = [m for m in candidates if m.agent_id == agent_id]
        
        if min_importance > 0:
            candidates = [m for m in candidates if m.importance >= min_importance]
        
        if not candidates:
            return []
        
        # Embed query
        query_embedding = self._create_embedding(query)
        
        # Calculate similarities
        scored_memories = []
        for memory in candidates:
            if memory.id in self._embeddings:
                sim = self._cosine_similarity(
                    query_embedding,
                    self._embeddings[memory.id]
                )
                # Weight by importance
                score = sim * (0.7 + 0.3 * memory.importance)
                scored_memories.append((memory, score))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = [m for m, _ in scored_memories[:top_k]]
        
        logger.debug(
            f"Retrieved {len(results)} memories for query: "
            f"'{query[:30]}...'"
        )
        
        return results
    
    async def _cleanup(self):
        """
        Remove old/low-importance memories
        
        STRATEGY:
        1. Remove expired memories
        2. If still over limit, remove lowest importance
        
        WHY cleanup?
        - Memory limits
        - Relevance decay
        - Quality maintenance
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove expired
        self.memories = [
            m for m in self.memories
            if m.created_at > cutoff_date
        ]
        
        # Remove embeddings for deleted memories
        valid_ids = {m.id for m in self.memories}
        self._embeddings = {
            k: v for k, v in self._embeddings.items()
            if k in valid_ids
        }
        
        # If still over limit, remove lowest importance
        if len(self.memories) > self.max_items:
            # Sort by importance
            self.memories.sort(key=lambda x: x.importance)
            
            # Keep top max_items
            remove_count = len(self.memories) - self.max_items
            removed = self.memories[:remove_count]
            self.memories = self.memories[remove_count:]
            
            # Remove embeddings
            for m in removed:
                self._embeddings.pop(m.id, None)
            
            logger.info(f"Cleaned up {remove_count} low-importance memories")
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text
        
        NOTE: Simple demo implementation
        Production should use proper embeddings
        """
        text = text.lower().strip()
        embedding = np.zeros(self.embedding_dim)
        
        # Character frequency
        for char in text:
            idx = ord(char) % self.embedding_dim
            embedding[idx] += 1
        
        # Word hashing
        for word in text.split():
            idx = hash(word) % self.embedding_dim
            embedding[idx] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        type_counts = {}
        agent_counts = {}
        
        for memory in self.memories:
            # Type counts
            type_name = memory.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Agent counts
            if memory.agent_id:
                agent_counts[memory.agent_id] = agent_counts.get(
                    memory.agent_id, 0
                ) + 1
        
        return {
            'total_memories': len(self.memories),
            'by_type': type_counts,
            'by_agent': agent_counts,
            'avg_importance': sum(m.importance for m in self.memories) / len(self.memories) if self.memories else 0,
            'max_items': self.max_items,
            'retention_days': self.retention_days
        }
    
    async def clear(self):
        """Clear all memories"""
        self.memories = []
        self._embeddings = {}
        logger.info("Memory bank cleared")
