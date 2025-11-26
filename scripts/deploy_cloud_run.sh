#!/bin/bash

# ============================================================================
# QEARIS Cloud Run Deployment Script
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="gen-lang-client-0472751146"
SERVICE_NAME="qearis"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${GREEN}🚀 QEARIS Cloud Run Deployment${NC}"
echo "=================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}Please login to gcloud${NC}"
    gcloud auth login
fi

# Set project
echo -e "${GREEN}Setting project...${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${GREEN}Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com

# Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest .

# Tag with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${TIMESTAMP}

# Configure Docker for GCR
echo -e "${GREEN}Configuring Docker for GCR...${NC}"
gcloud auth configure-docker

# Push to Container Registry
echo -e "${GREEN}Pushing to Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:${TIMESTAMP}

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# Load environment variables
source .env

# Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},GEMINI_PROJECT_ID=${GEMINI_PROJECT_ID},GEMINI_MODEL=${GEMINI_MODEL},ENVIRONMENT=production" \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 1 \
    --concurrency 80 \
    --port 8080

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "=================================="
echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo ""
echo "Test endpoints:"
echo "  Health: ${SERVICE_URL}/health"
echo "  API Docs: ${SERVICE_URL}/docs"
echo "  Metrics: ${SERVICE_URL}/metrics"
echo ""
echo "Test research endpoint:"
echo "  curl -X POST ${SERVICE_URL}/api/v1/research \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":\"How does quantum computing improve AI?\",\"domains\":[\"quantum\",\"ai\"]}'"
echo ""

# Test health endpoint
echo -e "${GREEN}Testing health endpoint...${NC}"
if curl -f ${SERVICE_URL}/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment successful!${NC}"
