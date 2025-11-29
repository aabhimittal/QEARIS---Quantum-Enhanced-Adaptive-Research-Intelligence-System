# QEARIS Deployment Guide

Complete guide for deploying QEARIS to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Google Cloud Run](#google-cloud-run)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- **Python 3.10+**
- **Docker** (for containerized deployment)
- **Google Cloud SDK** (for Cloud Run)
- **kubectl** (for Kubernetes)

### API Keys

1. **Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com)
   - Create API key
   - Add to `.env`: `GEMINI_API_KEY=your_key_here`

2. **Project Configuration**
```bash
   GEMINI_PROJECT_ID=gen-lang-client-0472751146
   GEMINI_PROJECT_NUMBER=412097861656
```

## Local Development

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/qearis.git
cd qearis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### 2. Run Locally
```bash
# Start API server
python -m uvicorn src.api.main:app --reload --port 8080

# In another terminal, run tests
pytest tests/ -v

# Check health
curl http://localhost:8080/health
```

### 3. Test Research Endpoint
```bash
curl -X POST http://localhost:8080/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does quantum computing improve AI?",
    "domains": ["quantum", "ai"],
    "max_agents": 3
  }'
```

## Docker Deployment

### 1. Build Image
```bash
# Build
docker build -t qearis:latest .

# Verify
docker images | grep qearis
```

### 2. Run Container
```bash
# Run with environment file
docker run -d \
  --name qearis \
  -p 8080:8080 \
  --env-file .env \
  qearis:latest

# Check logs
docker logs -f qearis

# Test
curl http://localhost:8080/health
```

### 3. Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Google Cloud Run

### Method 1: Automated Script
```bash
# Make script executable
chmod +x scripts/deploy_cloud_run.sh

# Deploy
./scripts/deploy_cloud_run.sh
```

### Method 2: Manual Deployment
```bash
# 1. Authenticate
gcloud auth login
gcloud config set project gen-lang-client-0472751146

# 2. Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Build and push
docker build -t gcr.io/gen-lang-client-0472751146/qearis:latest .
docker push gcr.io/gen-lang-client-0472751146/qearis:latest

# 4. Deploy
gcloud run deploy qearis \
  --image gcr.io/gen-lang-client-0472751146/qearis:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}" 
  --memory 2Gi 
  --cpu 2
  --timeout 300 
  --max-instances 10 
  --min-instances 1
# 5. Get service URL
gcloud run services describe qearis 
--region us-central1 
--format 'value(status.url)'
### Configuration Options
```bash
# Custom domain
gcloud run services update qearis \
  --region us-central1 \
  --add-domain-mapping example.com

# Increase resources
gcloud run services update qearis \
  --region us-central1 \
  --memory 4Gi \
  --cpu 4

# Set environment variables
gcloud run services update qearis \
  --region us-central1 \
  --set-env-vars="LOG_LEVEL=DEBUG,MAX_PARALLEL_AGENTS=6"

# Enable authentication
gcloud run services update qearis \
  --region us-central1 \
  --no-allow-unauthenticated
```

### Verify Deployment
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe qearis \
  --region us-central1 \
  --format 'value(status.url)')

# Test health
curl ${SERVICE_URL}/health

# Test research
curl -X POST ${SERVICE_URL}/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum entanglement",
    "domains": ["quantum"]
  }'
```

## Kubernetes Deployment

### 1. Create Cluster
```bash
# GKE cluster
gcloud container clusters create qearis-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --region=us-central1

# Get credentials
gcloud container clusters get-credentials qearis-cluster \
  --region=us-central1
```

### 2. Create Secrets
```bash
# Create secret for API keys
kubectl create secret generic qearis-secrets \
  --from-literal=gemini-api-key=${GEMINI_API_KEY} \
  --from-literal=gemini-project-id=gen-lang-client-0472751146
```

### 3. Deploy Application
```bash
# Apply deployment
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Check status
kubectl get pods
kubectl get services

# Get external IP
kubectl get service qearis-service
```

### 4. Scale Deployment
```bash
# Manual scaling
kubectl scale deployment qearis-deployment --replicas=5

# Autoscaling
kubectl autoscale deployment qearis-deployment \
  --min=2 \
  --max=10 \
  --cpu-percent=70
```

### 5. Update Deployment
```bash
# Rolling update
kubectl set image deployment/qearis-deployment \
  qearis=gcr.io/gen-lang-client-0472751146/qearis:v1.1

# Check rollout status
kubectl rollout status deployment/qearis-deployment

# Rollback if needed
kubectl rollout undo deployment/qearis-deployment
```

## Monitoring & Observability

### Cloud Run Monitoring
```bash
# View logs
gcloud run services logs read qearis --region us-central1

# Follow logs
gcloud run services logs tail qearis --region us-central1

# Metrics
gcloud monitoring dashboards create \
  --config-from-file=deployment/cloud-run/dashboard.json
```

### Kubernetes Monitoring
```bash
# Install Prometheus & Grafana
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000

# Access at http://localhost:3000
```

### Custom Metrics

QEARIS exposes metrics at `/metrics` endpoint:
```bash
curl ${SERVICE_URL}/metrics
```

**Available Metrics:**
- `qearis_tasks_total` - Total tasks processed
- `qearis_tasks_duration_seconds` - Task execution time
- `qearis_agents_utilization` - Agent utilization percentage
- `qearis_quantum_energy` - Quantum optimization energy
- `qearis_memory_items` - Memory bank items
- `qearis_rag_retrievals` - RAG retrievals count

### Logging

**Structured logging example:**
```json
{
  "timestamp": "2025-11-23T10:30:00Z",
  "level": "INFO",
  "component": "orchestrator",
  "message": "Task completed",
  "task_id": "task_123",
  "execution_time": 45.2,
  "confidence": 0.92
}
```

### Tracing

**OpenTelemetry integration:**
```python
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Configure tracing
tracer = trace.get_tracer(__name__)

with tracer.start_span("research_workflow") as span:
    span.set_attribute("query", research_query)
    # Execute workflow
```

## Performance Optimization

### 1. Caching
```python
# Redis caching for RAG results
import redis

cache = redis.Redis(host='redis', port=6379)

def cached_retrieve(query: str):
    cached = cache.get(f"rag:{query}")
    if cached:
        return json.loads(cached)
    
    results = rag_system.retrieve(query)
    cache.setex(f"rag:{query}", 3600, json.dumps(results))
    return results
```

### 2. Connection Pooling
```python
# Database connection pool
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 3. Async Optimization
```python
# Batch processing
async def batch_process_tasks(tasks: List[Task]):
    results = await asyncio.gather(*[
        process_task(task) for task in tasks
    ])
    return results
```

## Security

### 1. API Authentication
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Security(security)):
    if not validate_token(token):
        raise HTTPException(status_code=401)
    return token
```

### 2. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/research")
@limiter.limit("10/minute")
async def research(request: Request):
    # Process request
    pass
```

### 3. Secrets Management
```bash
# Google Secret Manager
gcloud secrets create gemini-api-key --data-file=-

# Access in Cloud Run
gcloud run services update qearis \
  --update-secrets=GEMINI_API_KEY=gemini-api-key:latest
```

## Cost Optimization

### Cloud Run
```bash
# Set CPU allocation to "CPU only allocated during request"
gcloud run services update qearis \
  --region us-central1 \
  --cpu-throttling

# Reduce min instances during low traffic
gcloud run services update qearis \
  --region us-central1 \
  --min-instances 0

# Use smaller memory allocation if possible
gcloud run services update qearis \
  --region us-central1 \
  --memory 1Gi
```

### Resource Monitoring
```bash
# Check resource usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/billable_instance_time"'

# Set budget alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="QEARIS Budget" \
  --budget-amount=100USD
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
gcloud run services logs read qearis --limit 50

# Common causes:
# - Missing environment variables
# - Port mismatch (ensure PORT=8080)
# - Dependencies not installed
```

**Solution:**
```bash
# Verify environment
gcloud run services describe qearis \
  --region us-central1 \
  --format='value(spec.template.spec.containers[0].env)'

# Update if needed
gcloud run services update qearis \
  --region us-central1 \
  --set-env-vars="PORT=8080"
```

#### 2. Timeout Errors
```bash
# Increase timeout
gcloud run services update qearis \
  --region us-central1 \
  --timeout 600
```

#### 3. Memory Issues
```bash
# Check memory usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/memory/utilizations"'

# Increase memory
gcloud run services update qearis \
  --region us-central1 \
  --memory 4Gi
```

#### 4. API Rate Limits

**Gemini API rate limits:**
- Free tier: 60 requests/minute
- Paid tier: Higher limits

**Solution:**
```python
# Add exponential backoff
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
async def call_gemini_api():
    # API call
    pass
```

#### 5. Cold Start Latency
```bash
# Increase min instances
gcloud run services update qearis \
  --region us-central1 \
  --min-instances 1

# Or use warm-up requests
curl ${SERVICE_URL}/health
```

### Debug Commands
```bash
# SSH into container (if needed)
gcloud run services proxy qearis --region us-central1

# View real-time logs
gcloud run services logs tail qearis --region us-central1

# Check service configuration
gcloud run services describe qearis \
  --region us-central1 \
  --format yaml

# Test locally with exact cloud environment
docker run -p 8080:8080 \
  --env-file .env \
  --env PORT=8080 \
  gcr.io/gen-lang-client-0472751146/qearis:latest
```

### Performance Testing
```bash
# Load testing with Apache Bench
ab -n 100 -c 10 -p request.json -T application/json \
  ${SERVICE_URL}/api/v1/research

# Load testing with wrk
wrk -t12 -c400 -d30s ${SERVICE_URL}/health
```

## Maintenance

### 1. Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Rebuild and redeploy
./scripts/deploy_cloud_run.sh
```

### 2. Backups
```bash
# Backup session data
gsutil -m cp -r /app/sessions gs://qearis-backups/sessions/$(date +%Y%m%d)

# Backup memory bank
gsutil -m cp -r /app/data gs://qearis-backups/data/$(date +%Y%m%d)
```

### 3. Database Migrations
```bash
# Using Alembic
alembic revision --autogenerate -m "Add new table"
alembic upgrade head
```

## CI/CD Pipeline

### GitHub Actions Workflow

The `.github/workflows/deploy.yml` file automates:

1. **Build** - Build Docker image
2. **Test** - Run test suite
3. **Push** - Push to Container Registry
4. **Deploy** - Deploy to Cloud Run
5. **Verify** - Health check

**Trigger deployment:**
```bash
git push origin main
# Watch deployment
gh run watch
```

## Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured in Secret Manager
- [ ] Health checks passing
- [ ] Logging configured
- [ ] Monitoring dashboards created
- [ ] Alerts configured
- [ ] Rate limiting enabled
- [ ] Backups scheduled
- [ ] Documentation updated
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Cost budget set

## Support

For deployment issues:

1. Check [troubleshooting](#troubleshooting) section
2. Review [logs](#monitoring--observability)
3. Open GitHub issue
4. Contact: your.email@example.com

---

**Last Updated:** November 2025  
**Version:** 1.0.0
