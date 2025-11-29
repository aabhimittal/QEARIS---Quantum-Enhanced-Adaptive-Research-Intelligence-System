#!/bin/bash
# ============================================================================
# QEARIS - Cloud Run Deployment Script
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Agent Deployment (BONUS - 5 points)
# 
# This script automates deployment to Google Cloud Run, including:
# - Container image building
# - Image pushing to Container Registry
# - Cloud Run service deployment
# - Environment variable configuration
# 
# SECURITY:
# - API keys loaded from environment variables (not hardcoded)
# - Uses Google Secret Manager for sensitive data
# 
# USAGE:
#   ./deploy.sh
#   ./deploy.sh --project my-project --region us-central1
# ============================================================================

set -e  # Exit on any error
set -o pipefail  # Ensure pipeline failures are caught

# ============================================================================
# Configuration (Override via environment or command line)
# ============================================================================
PROJECT_ID="${GOOGLE_PROJECT_ID:-gen-lang-client-0472751146}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-qearis}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Resource configuration
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-300}"

# ============================================================================
# Parse command line arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./deploy.sh [options]"
            echo "Options:"
            echo "  --project PROJECT_ID    Google Cloud project ID"
            echo "  --region REGION         Cloud Run region"
            echo "  --memory MEMORY         Memory allocation (e.g., 2Gi)"
            echo "  --cpu CPU               CPU allocation (e.g., 2)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update IMAGE_NAME after parsing
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo "============================================"
echo "QEARIS Cloud Run Deployment"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Project:      ${PROJECT_ID}"
echo "  Region:       ${REGION}"
echo "  Service:      ${SERVICE_NAME}"
echo "  Image:        ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Memory:       ${MEMORY}"
echo "  CPU:          ${CPU}"
echo "  Instances:    ${MIN_INSTANCES}-${MAX_INSTANCES}"
echo ""

# Check for required tools
echo "Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi

echo "✓ Prerequisites satisfied"

# ============================================================================
# Authenticate and Configure
# ============================================================================
echo ""
echo "Configuring Google Cloud..."

# Set project
gcloud config set project "${PROJECT_ID}"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet
gcloud services enable secretmanager.googleapis.com --quiet

echo "✓ APIs enabled"

# Configure Docker for GCR
echo "Configuring Docker for Container Registry..."
gcloud auth configure-docker --quiet

echo "✓ Docker configured"

# ============================================================================
# Build Container Image
# ============================================================================
echo ""
echo "Building container image..."

# Move to project root (script is in deployment/cloud_run)
cd "$(dirname "$0")/../.."

# Build image
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

echo "✓ Image built: ${IMAGE_NAME}:${IMAGE_TAG}"

# ============================================================================
# Push to Container Registry
# ============================================================================
echo ""
echo "Pushing image to Container Registry..."

docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo "✓ Image pushed"

# ============================================================================
# Deploy to Cloud Run
# ============================================================================
echo ""
echo "Deploying to Cloud Run..."

# Check if GEMINI_API_KEY is set
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "WARNING: GEMINI_API_KEY not set. Using Secret Manager reference."
    ENV_VARS="ENVIRONMENT=production,LOG_LEVEL=INFO"
    SECRET_FLAG="--set-secrets=GEMINI_API_KEY=gemini-api-key:latest"
else
    ENV_VARS="GEMINI_API_KEY=${GEMINI_API_KEY},ENVIRONMENT=production,LOG_LEVEL=INFO"
    SECRET_FLAG=""
fi

# Deploy service
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:${IMAGE_TAG}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --timeout "${TIMEOUT}" \
    --max-instances "${MAX_INSTANCES}" \
    --min-instances "${MIN_INSTANCES}" \
    --port 8080 \
    --set-env-vars="${ENV_VARS}" \
    ${SECRET_FLAG} \
    --quiet

echo "✓ Service deployed"

# ============================================================================
# Get Service URL and Verify
# ============================================================================
echo ""
echo "Getting service URL..."

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo "Service URL: ${SERVICE_URL}"

# Health check
echo ""
echo "Running health check..."

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" || echo "000")

if [ "${HEALTH_RESPONSE}" = "200" ]; then
    echo "✓ Health check passed"
else
    echo "WARNING: Health check returned ${HEALTH_RESPONSE}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Service:     ${SERVICE_NAME}"
echo "URL:         ${SERVICE_URL}"
echo "Region:      ${REGION}"
echo "Resources:   ${MEMORY} RAM, ${CPU} CPU"
echo ""
echo "Test endpoints:"
echo "  Health:    curl ${SERVICE_URL}/health"
echo "  Research:  curl -X POST ${SERVICE_URL}/api/v1/research \\"
echo "             -H 'Content-Type: application/json' \\"
echo "             -d '{\"query\": \"How does quantum computing work?\", \"domains\": [\"quantum\"]}'"
echo ""
echo "Monitor logs:"
echo "  gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
