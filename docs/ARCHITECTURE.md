# 🏗️ QEARIS Architecture

Comprehensive architecture documentation for the Quantum-Enhanced Adaptive Research Intelligence System.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Agent System](#agent-system)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Scalability](#scalability)

## System Overview

QEARIS is a multi-agent research system that combines quantum-inspired optimization, retrieval-augmented generation (RAG), and the Model Context Protocol (MCP) to conduct autonomous research.

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (FastAPI)                     │
│                     Authentication & Rate Limiting               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Multi-Agent Orchestrator                       │
│                  (Coordination & Workflow)                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │         Quantum Task Allocator                          │    │
│  │  (Simulated Annealing Optimization)                     │    │
│  └────────────────────────────────────────────────────────┘    │
└──┬────────────────┬─────────────────┬─────────────────────────┘
   │                │                 │
   │  Parallel      │  Sequential     │  Loop
   │  Execution     │  Validation     │  Refinement
   │                │                 │
┌──▼──────────┐  ┌─▼──────────┐  ┌──▼──────────┐
│ Research    │  │ Validation │  │  Synthesis  │
│ Agent Pool  │  │   Agent    │  │    Agent    │
│  (3-4)      │  │            │  │             │
└──┬──────────┘  └─┬──────────┘  └──┬──────────┘
   │                │                │
   └────────────────┼────────────────┘
                    │
┌───────────────────▼────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌──────┐  ┌──────┐  ┌────────┐  ┌──────┐  ┌──────────────┐  │
│  │ RAG  │  │ MCP  │  │ Memory │  │Gemini│  │   Context    │  │
│  │System│  │Server│  │  Bank  │  │ API  │  │   Manager    │  │
│  └──────┘  └──────┘  └────────┘  └──────┘  └──────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. API Layer

**Technology:** FastAPI  
**Purpose:** HTTP interface for external clients

**Key Features:**
- RESTful endpoints
- OpenAPI documentation
- Request validation (Pydantic)
- CORS middleware
- Rate limiting
- Authentication

**Endpoints:**
```python
POST   /api/v1/research          # Create research task
GET    /api/v1/session/{id}      # Get session info
POST   /api/v1/session/{id}/pause # Pause workflow
POST   /api/v1/session/{id}/resume # Resume workflow
GET    /api/v1/evaluation/{id}   # Get agent evaluation
GET    /api/v1/metrics            # System metrics
GET    /health                    # Health check
```

### 2. Orchestrator Layer

**Core Component:** `MultiAgentOrchestrator`  
**Purpose:** Coordinates all agents and workflow execution

**Responsibilities:**
- Task creation and distribution
- Agent lifecycle management
- Workflow state management
- Session persistence
- Result aggregation

**Workflow States:**
```python
PENDING → ASSIGNED → IN_PROGRESS → VALIDATING → COMPLETED
                                  ↓
                                FAILED
```

### 3. Agent System

#### Agent Hierarchy
```
BaseAgent (Abstract)
├── ResearchAgent
│   ├── Parallel execution
│   ├── RAG integration
│   ├── MCP tool usage
│   └── Gemini reasoning
├── ValidationAgent
│   ├── Sequential processing
│   ├── Quality checks
│   └── Fact verification
└── SynthesisAgent
    ├── Loop refinement
    ├── Multi-source integration
    └── Report generation
```

#### Agent Communication

**A2A Protocol Implementation:**
```python
class A2AMessage:
    from_agent: str
    to_agent: str
    message_type: str  # REQUEST, RESPONSE, NOTIFICATION
    content: Dict[str, Any]
    timestamp: datetime

class A2AProtocol:
    async def send(message: A2AMessage) -> A2AResponse
    async def receive() -> A2AMessage
    async def broadcast(message: A2AMessage) -> List[A2AResponse]
```

### 4. Quantum Optimizer

**Algorithm:** Simulated Quantum Annealing  
**Purpose:** Optimal task-agent assignment

**Energy Function:**
```
E(assignment) = Σ cost(task_i, agent_j) * x_ij + λ * Var(load)

Where:
- cost: Task-agent compatibility cost
- x_ij: Binary assignment variable
- λ: Load balancing weight
- Var(load): Variance in agent workload
```

**Optimization Process:**

1. **Initialize:** Random assignment
2. **Calculate:** System energy
3. **Mutate:** Quantum fluctuation (reassign 1-2 tasks)
4. **Evaluate:** Accept/reject via Metropolis criterion
5. **Cool:** Reduce temperature
6. **Repeat:** Until convergence

**Convergence Criteria:**
- Energy change < threshold (0.01)
- Max iterations reached (100)

### 5. RAG System

**Components:**
- **Embedder:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB
- **Chunking:** 512 tokens with 50 token overlap
- **Retrieval:** Cosine similarity, top-k=5

**Workflow:**
```
Document → Chunk → Embed → Store
                              ↓
Query → Embed → Search → Retrieve → Augment
```

**Similarity Calculation:**
```python
similarity = 1 - cosine_distance(query_embedding, doc_embedding)
```

### 6. Memory Bank

**Memory Types:**
- **Episodic:** Task experiences, outcomes
- **Semantic:** Facts, knowledge
- **Procedural:** How-to patterns

**Consolidation:**
```python
importance_new = importance_old * exp(-decay_rate * age_days) + access_boost
access_boost = min(0.3, access_count * 0.01)
```

**Pruning Strategy:**
- Keep top 1000 memories
- Sort by importance + recency
- Remove least important

### 7. MCP Server

**Model Context Protocol Implementation:**

**Tool Structure:**
```python
class MCPTool:
    name: str
    description: str
    parameters: JSONSchema
    handler: Callable
    timeout: int
    retry_policy: Dict
```

**Registered Tools:**
- `search_knowledge_base` - RAG retrieval
- `search_memory` - Memory retrieval
- `calculate_priority` - Task prioritization
- `web_search` - External search (Gemini grounding)

### 8. Context Manager

**Purpose:** Optimize context window usage

**Strategies:**
1. **Prioritization:** Keep high-importance content
2. **Compression:** Summarize low-priority content
3. **Deduplication:** Remove redundant information
4. **Chunking:** Split into manageable pieces

**Priority Levels:**
- Critical (100): Current task
- High (80): Recent memories
- Medium (50): Historical context
- Low (20): Auxiliary info

### 9. Gemini Integration

**Model:** gemini-1.5-pro  
**Context Window:** 1M tokens (using 100K for efficiency)

**Features Used:**
- **Function Calling:** Tool integration
- **Long Context:** Extended research
- **Grounding:** Web search integration

**Generation Config:**
```python
{
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192
}
```

## Data Flow

### Research Workflow
```
1. Client Request
   ↓
2. API Validation
   ↓
3. Task Creation (per domain)
   ↓
4. Quantum Optimization
   │  - Calculate compatibility matrix
   │  - Optimize assignments
   │  - Minimize system energy
   ↓
5. Parallel Research
   │  ┌─→ Agent 1 → RAG → Memory → Gemini → Result 1
   │  ├─→ Agent 2 → RAG → Memory → Gemini → Result 2
   │  └─→ Agent 3 → RAG → Memory → Gemini → Result 3
   ↓
6. Sequential Validation
   │  Result 1 → Validator → Validated 1
   │  Result 2 → Validator → Validated 2
   │  Result 3 → Validator → Validated 3
   ↓
7. Loop Synthesis
   │  Iteration 1: Combine → Quality Check → Refine
   │  Iteration 2: Combine → Quality Check → Refine
   │  Iteration 3: Combine → Quality Check → Accept
   ↓
8. Final Report
   ↓
9. Client Response
```

### Agent Execution Flow
```
Agent.execute_task(task)
  ↓
1. Retrieve Memories (relevant past experiences)
  ↓
2. Search Knowledge Base (RAG)
  ↓
3. Build Context (prioritize + compact)
  ↓
4. Call Gemini API
  │  - System prompt with context
  │  - Function calling for tools
  │  - Generate response
  ↓
5. Process Response
  │  - Extract content
  │  - Calculate confidence
  │  - Collect sources
  ↓
6. Store Experience (memory bank)
  ↓
7. Return Result
```

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | FastAPI | Web framework |
| Agent Runtime | Google Gemini | LLM reasoning |
| Vector DB | ChromaDB | Semantic search |
| Embeddings | Sentence-Transformers | Text vectorization |
| Optimization | NumPy/SciPy | Quantum algorithms |
| Observability | OpenTelemetry | Metrics & tracing |
| Deployment | Docker + Cloud Run | Containerization |

### Dependencies
```
# Core
fastapi==0.109.0
google-generativeai==0.3.2
chromadb==0.4.22
sentence-transformers==2.3.1

# Optimization
numpy==1.26.3
scipy==1.12.0

# Observability
opentelemetry-api==1.22.0
prometheus-client==0.19.0

# Utilities
pydantic==2.5.3
aiohttp==3.9.3
tenacity==8.2.3
```

## Design Patterns

### 1. Factory Pattern

**Usage:** Agent creation
```python
class AgentFactory:
    def create_agent(type: AgentType) -> BaseAgent:
        if type == AgentType.RESEARCHER:
            return ResearchAgent(...)
        elif type == AgentType.VALIDATOR:
            return ValidationAgent(...)
```

### 2. Strategy Pattern

**Usage:** Execution patterns
```python
class ExecutionStrategy(ABC):
    @abstractmethod
    async def execute(tasks, agents):
        pass

class ParallelStrategy(ExecutionStrategy):
    async def execute(tasks, agents):
        return await asyncio.gather(...)

class SequentialStrategy(ExecutionStrategy):
    async def execute(tasks, agents):
        for task in tasks:
            await execute_one(task)
```

### 3. Observer Pattern

**Usage:** Metrics collection
```python
class Observable:
    observers: List[Observer] = []
    
    def notify(event):
        for observer in observers:
            observer.update(event)

class MetricsObserver(Observer):
    def update(event):
        metrics.record(event.name, event.value)
```

### 4. Command Pattern

**Usage:** Task execution
```python
class Command(ABC):
    @abstractmethod
    async def execute():
        pass

class ResearchCommand(Command):
    async def execute():
        return await agent.research(task)
```

### 5. Singleton Pattern

**Usage:** Orchestrator instance
```python
class MultiAgentOrchestrator:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

## Scalability

### Horizontal Scaling

**Cloud Run Auto-scaling:**
```
min_instances: 1
max_instances: 10
concurrency: 80
```

**Load Distribution:**
- Request-level parallelism
- Agent pool expansion
- Stateless design

### Vertical Scaling

**Resource Allocation:**
```
CPU: 2 cores
Memory: 2Gi
Timeout: 300s
```

**Optimization:**
- Connection pooling
- Caching (Redis)
- Lazy loading

### Database Scaling

**Vector Database:**
- Sharding by domain
- Read replicas
- Index optimization

**Memory Bank:**
- Periodic archival
- Compression
- Pruning

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Response Time (P50) | < 30s | 25s |
| Response Time (P95) | < 60s | 45s |
| Throughput | > 1 req/s | 1.3 req/s |
| Availability | > 99.5% | 99.8% |
| Error Rate | < 1% | 0.5% |

## Security Architecture

### Authentication
```
Client → API Key → Verification → JWT Token → Request
```

### Authorization
```
JWT Token → Role Check → Permission Check → Allow/Deny
```

### Data Security

- **In Transit:** TLS 1.3
- **At Rest:** AES-256
- **Secrets:** Google Secret Manager

### Rate Limiting
```python
@limiter.limit("10/minute")
async def research_endpoint():
    pass
```

## Monitoring & Observability

### Metrics
```
# System Metrics
qearis_requests_total
qearis_request_duration_seconds
qearis_errors_total

# Agent Metrics
qearis_agent_tasks_completed
qearis_agent_success_rate
qearis_agent_execution_time

# Quantum Metrics
qearis_quantum_energy
qearis_quantum_iterations

# RAG Metrics
qearis_rag_retrievals
qearis_rag_similarity_score
```

### Logging
```json
{
  "timestamp": "2025-11-23T10:30:00Z",
  "level": "INFO",
  "component": "orchestrator",
  "message": "Task completed",
  "context": {
    "task_id": "task_123",
    "session_id": "session_456",
    "execution_time": 45.2,
    "confidence": 0.92
  }
}
```

### Tracing
```
Span: research_workflow
  ├── Span: quantum_optimization
  ├── Span: parallel_execution
  │   ├── Span: agent_1_research
  │   ├── Span: agent_2_research
  │   └── Span: agent_3_research
  ├── Span: validation
  └── Span: synthesis
```

---

**Version:** 1.0.0  
**Last Updated:** November 2025  
**Maintained By:** QEARIS Team
