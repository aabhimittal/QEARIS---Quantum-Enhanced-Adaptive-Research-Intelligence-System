# QEARIS API Reference

**Version:** 1.0.0  
**Base URL:** `http://localhost:8080/api/v1`

---

## Overview

The QEARIS API provides endpoints for multi-agent research operations, session management, and system monitoring.

### Authentication

API endpoints accept requests with optional API key authentication:

```
X-API-Key: your_api_key_here
```

### Base Response Format

All responses follow this structure:

```json
{
  "success": true,
  "data": {...},
  "error": null
}
```

---

## Endpoints

### Research Operations

#### POST /research

Execute a multi-agent research operation.

**CAPSTONE REQUIREMENT:** Multi-Agent System, Gemini Integration

**Request Body:**

```json
{
  "query": "How does quantum computing improve AI?",
  "domains": ["quantum", "ai"],
  "max_agents": 3,
  "priority": "high",
  "metadata": {
    "user_id": "user123"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Research query (10-1000 chars) |
| `domains` | array | No | Research domains (default: ["general"]) |
| `max_agents` | integer | No | Max parallel agents 1-10 (default: 3) |
| `priority` | string | No | Task priority: low, medium, high, critical |
| `metadata` | object | No | Additional metadata |

**Response (200 OK):**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": "Research findings on quantum computing and AI...",
  "confidence": 0.89,
  "sources": 12,
  "execution_time": 45.2,
  "metrics": {
    "tasks_created": 2,
    "agents_used": 2,
    "research_results": 2,
    "validated_results": 2,
    "synthesis_iterations": 3,
    "total_sources": 12
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest advances in quantum machine learning?",
    "domains": ["quantum", "ai"],
    "max_agents": 3
  }'
```

---

### Session Management

#### POST /sessions

Create a new research session.

**CAPSTONE REQUIREMENT:** Sessions & Memory

**Request Body:**

```json
{
  "query": "Research quantum computing",
  "domains": ["quantum", "ai"],
  "metadata": {
    "project": "research_project_1"
  }
}
```

**Response (201 Created):**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "query": "Research quantum computing",
  "domains": ["quantum", "ai"],
  "created_at": "2024-01-15T10:30:00Z",
  "tasks_count": 0,
  "results_count": 0
}
```

#### GET /sessions/{session_id}

Get session details.

**Response (200 OK):**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "query": "Research quantum computing",
  "domains": ["quantum", "ai"],
  "created_at": "2024-01-15T10:30:00Z",
  "tasks_count": 3,
  "results_count": 3
}
```

#### GET /sessions

List all sessions.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status |
| `page` | integer | Page number (default: 1) |
| `page_size` | integer | Items per page (default: 20) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "session_id": "...",
      "status": "completed",
      "query": "...",
      "created_at": "..."
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "pages": 3
}
```

#### DELETE /sessions/{session_id}

Delete a session.

**Response (204 No Content)**

---

### Tool Execution

#### POST /tools/execute

Execute an MCP tool directly.

**CAPSTONE REQUIREMENT:** Tools Integration

**Request Body:**

```json
{
  "tool_name": "web_search",
  "parameters": {
    "query": "quantum computing",
    "max_results": 5
  },
  "timeout": 60
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "tool_name": "web_search",
  "result": {
    "query": "quantum computing",
    "results": [...]
  },
  "execution_time": 1.5
}
```

#### GET /tools

List available tools.

**Response (200 OK):**

```json
{
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web for information",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "max_results": {"type": "integer"}
        },
        "required": ["query"]
      }
    }
  ]
}
```

---

### Agent Management

#### GET /agents

List available agents.

**CAPSTONE REQUIREMENT:** Multi-Agent System

**Response (200 OK):**

```json
{
  "agents": [
    {
      "agent_id": "researcher_0",
      "agent_type": "researcher",
      "specialization": "quantum",
      "status": "idle",
      "tasks_completed": 42,
      "success_rate": 0.95
    }
  ],
  "total": 5
}
```

#### GET /agents/{agent_id}

Get agent details.

**Response (200 OK):**

```json
{
  "agent_id": "researcher_0",
  "agent_type": "researcher",
  "specialization": "quantum",
  "status": "idle",
  "tasks_completed": 42,
  "success_rate": 0.95,
  "metrics": {
    "avg_execution_time": 15.2,
    "total_tokens": 50000
  }
}
```

#### POST /agents/{agent_id}/pause

Pause an agent.

**CAPSTONE REQUIREMENT:** Long-Running Operations

**Response (200 OK):**

```json
{
  "agent_id": "researcher_0",
  "status": "paused"
}
```

#### POST /agents/{agent_id}/resume

Resume a paused agent.

**Response (200 OK):**

```json
{
  "agent_id": "researcher_0",
  "status": "running"
}
```

---

### System Operations

#### GET /health

Health check endpoint.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api": "healthy",
    "gemini": "healthy",
    "rag": "healthy",
    "memory": "healthy"
  }
}
```

#### GET /metrics

Get system metrics.

**CAPSTONE REQUIREMENT:** Observability

**Response (200 OK):**

```json
{
  "total_requests": 1000,
  "active_sessions": 5,
  "agents_available": 10,
  "avg_response_time": 32.5,
  "success_rate": 0.98,
  "system_metrics": {
    "memory_usage_mb": 512,
    "cpu_usage_percent": 45
  }
}
```

#### GET /statistics

Get detailed system statistics.

**Response (200 OK):**

```json
{
  "sessions": {
    "total": 100,
    "by_status": {
      "completed": 80,
      "in_progress": 15,
      "failed": 5
    }
  },
  "agents": {
    "total": 10,
    "by_type": {
      "researcher": 4,
      "validator": 1,
      "synthesizer": 1
    }
  },
  "memory": {
    "total_memories": 500,
    "by_type": {
      "episodic": 200,
      "semantic": 200,
      "procedural": 100
    }
  }
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "error": "ValidationError",
  "message": "Query must be at least 10 characters",
  "details": {
    "field": "query",
    "constraint": "min_length"
  }
}
```

### 404 Not Found

```json
{
  "error": "NotFound",
  "message": "Session not found",
  "details": {
    "session_id": "..."
  }
}
```

### 429 Too Many Requests

```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "details": {
    "retry_after": 60
  }
}
```

### 500 Internal Server Error

```json
{
  "error": "InternalError",
  "message": "An unexpected error occurred",
  "request_id": "req_abc123"
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/research` | 10 requests/minute |
| `/tools/execute` | 30 requests/minute |
| Other endpoints | 100 requests/minute |

---

## Webhooks

QEARIS supports webhooks for async notifications.

### Configure Webhook

```json
POST /webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["research.completed", "session.updated"]
}
```

### Event Payload

```json
{
  "event": "research.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "session_id": "...",
    "status": "completed",
    "confidence": 0.89
  }
}
```

---

## SDK Examples

### Python

```python
import requests

# Create research request
response = requests.post(
    "http://localhost:8080/api/v1/research",
    json={
        "query": "How does quantum computing work?",
        "domains": ["quantum"],
        "max_agents": 2
    },
    headers={"X-API-Key": "your_key"}
)

result = response.json()
print(f"Confidence: {result['confidence']}")
print(f"Report: {result['result']}")
```

### cURL

```bash
# Research query
curl -X POST http://localhost:8080/api/v1/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"query": "Explain quantum entanglement", "domains": ["quantum"]}'

# Health check
curl http://localhost:8080/api/v1/health

# List agents
curl http://localhost:8080/api/v1/agents
```

---

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:

```
GET /openapi.json
```

Interactive API documentation (Swagger UI):

```
GET /docs
```

Alternative documentation (ReDoc):

```
GET /redoc
```

---

*QEARIS API v1.0.0 - Built for the Kaggle Capstone Project*
