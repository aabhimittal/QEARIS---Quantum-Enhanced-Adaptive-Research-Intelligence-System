"""
RAG System - Retrieval Augmented Generation

PURPOSE: Ground AI responses in factual knowledge
TECHNIQUE: Semantic search + vector embeddings
STORAGE: In-memory (production: use vector DB)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Document in the knowledge base
    
    DESIGN:
    - Content chunked for retrieval
    - Metadata for filtering
    - Embedding for similarity search
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    id: str = field(default="")
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class RAGSystem:
    """
    Retrieval Augmented Generation system
    
    ARCHITECTURE:
    -------------
    1. Document Store: Holds chunked documents
    2. Embedder: Converts text to vectors
    3. Index: Enables similarity search
    4. Retriever: Finds relevant documents
    
    WHY RAG?
    --------
    - Grounds responses in facts
    - Reduces hallucination
    - Enables citation
    - Keeps knowledge current
    
    LIMITATIONS:
    ------------
    - In-memory only (demo)
    - Simple cosine similarity
    - No production embeddings
    
    PRODUCTION WOULD USE:
    - ChromaDB, Pinecone, or Weaviate
    - sentence-transformers embeddings
    - Optimized indexing (HNSW)
    """
    
    def __init__(
        self,
        config: Any = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initialize RAG system
        
        PARAMETERS:
        -----------
        config: Application settings
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        top_k: Default number of results
        """
        if config:
            self.chunk_size = getattr(config, 'CHUNK_SIZE', chunk_size)
            self.chunk_overlap = getattr(config, 'CHUNK_OVERLAP', chunk_overlap)
            self.top_k = getattr(config, 'TOP_K_RETRIEVAL', top_k)
            self.embedding_model = getattr(config, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.top_k = top_k
            self.embedding_model = 'all-MiniLM-L6-v2'
        
        # Document storage
        self.documents: List[Document] = []
        
        # Simple embedding dimension (demo)
        self.embedding_dim = 384
        
        logger.info(
            f"RAGSystem initialized: "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"model={self.embedding_model}"
        )
    
    @classmethod
    async def create(cls, config: Any = None) -> "RAGSystem":
        """
        Factory method for async creation
        
        ENABLES:
        - Async initialization
        - Loading initial documents
        - Connecting to external services
        """
        instance = cls(config)
        
        # Add some initial knowledge (demo)
        await instance._load_initial_knowledge()
        
        return instance
    
    async def _load_initial_knowledge(self):
        """Load initial knowledge base"""
        initial_docs = [
            {
                "content": """Quantum computing uses quantum mechanical phenomena 
                like superposition and entanglement to perform computation. 
                Unlike classical bits that are 0 or 1, quantum bits (qubits) 
                can exist in superposition of both states simultaneously.""",
                "source": "quantum_basics.txt"
            },
            {
                "content": """Large Language Models (LLMs) are neural networks 
                trained on vast amounts of text data. They learn patterns in 
                language and can generate human-like text, answer questions, 
                and perform various NLP tasks.""",
                "source": "ai_fundamentals.txt"
            },
            {
                "content": """Multi-agent systems coordinate multiple AI agents 
                to solve complex problems. Agents can specialize in different 
                tasks and collaborate through communication protocols.""",
                "source": "multi_agent.txt"
            },
            {
                "content": """RAG (Retrieval Augmented Generation) improves AI 
                responses by retrieving relevant information from a knowledge 
                base before generating answers. This grounds responses in facts 
                and reduces hallucination.""",
                "source": "rag_overview.txt"
            }
        ]
        
        for doc in initial_docs:
            await self.add_document(
                content=doc["content"],
                metadata={"source": doc["source"]}
            )
        
        logger.info(f"Loaded {len(initial_docs)} initial documents")
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document to knowledge base
        
        PROCESS:
        1. Chunk document
        2. Generate embedding for each chunk
        3. Store in index
        
        RETURNS: Document ID
        """
        metadata = metadata or {}
        
        # Chunk content
        chunks = self._chunk_text(content)
        
        doc_ids = []
        for i, chunk in enumerate(chunks):
            # Create embedding (simple demo version)
            embedding = self._create_embedding(chunk)
            
            doc = Document(
                content=chunk,
                metadata={
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                },
                embedding=embedding
            )
            
            self.documents.append(doc)
            doc_ids.append(doc.id)
        
        logger.debug(
            f"Added document: {len(chunks)} chunks, "
            f"source={metadata.get('source', 'unknown')}"
        )
        
        return doc_ids[0] if doc_ids else ""
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents
        
        PROCESS:
        1. Embed query
        2. Calculate similarity to all documents
        3. Return top-k most similar
        
        RETURNS: List of {content, similarity, metadata}
        """
        k = top_k or self.top_k
        
        if not self.documents:
            logger.warning("No documents in knowledge base")
            return []
        
        # Embed query
        query_embedding = self._create_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            if doc.embedding is not None:
                sim = self._cosine_similarity(query_embedding, doc.embedding)
                similarities.append((doc, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for doc, sim in similarities[:k]:
            results.append({
                'content': doc.content,
                'similarity': float(sim),
                'metadata': doc.metadata
            })
        
        logger.debug(
            f"Retrieved {len(results)} documents for query: "
            f"'{query[:50]}...'"
        )
        
        return results
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces
        
        STRATEGY: Fixed size with overlap
        
        WHY overlap?
        - Preserves context at boundaries
        - Reduces information loss
        """
        chunks = []
        start = 0
        text = text.strip()
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
        
        return chunks if chunks else [text]
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text
        
        NOTE: This is a DEMO implementation using simple hashing.
        Production should use:
        - sentence-transformers
        - OpenAI embeddings
        - Gemini embeddings
        
        The simple approach here uses character-based features
        to create a pseudo-embedding for demonstration.
        """
        # Normalize text
        text = text.lower().strip()
        
        # Create simple feature vector
        embedding = np.zeros(self.embedding_dim)
        
        # Character frequency features
        for i, char in enumerate(text):
            idx = ord(char) % self.embedding_dim
            embedding[idx] += 1
        
        # Word-based features
        words = text.split()
        for i, word in enumerate(words):
            # Hash word to index
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
        """
        Calculate cosine similarity
        
        FORMULA: cos(θ) = (A · B) / (||A|| × ||B||)
        
        RANGE: -1 to 1 (higher = more similar)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        sources = set()
        for doc in self.documents:
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        
        return {
            'total_documents': len(self.documents),
            'unique_sources': len(sources),
            'sources': list(sources),
            'embedding_dim': self.embedding_dim,
            'chunk_size': self.chunk_size
        }
    
    async def clear(self):
        """Clear all documents"""
        self.documents = []
        logger.info("RAG system cleared")
