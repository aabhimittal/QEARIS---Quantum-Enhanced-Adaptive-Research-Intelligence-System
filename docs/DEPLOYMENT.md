# 🚀 QEARIS Deployment Guide

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
