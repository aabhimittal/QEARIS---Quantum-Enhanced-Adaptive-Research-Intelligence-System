#!/bin/bash
# ============================================================================
# QEARIS Manual Docker Deployment Script
# ============================================================================
#
# This script deploys QEARIS to Google Cloud Run using a pre-built Docker image,
# bypassing Git repository link issues that can occur with Cloud Build triggers.
#
# ERROR RESOLVED:
#   "invalid argument: git repository link name must be in the format of
#    'projects//locations//connections//gitRepositoryLinks/'"
#
# This error occurs when Cloud Build tries to connect to Git using 2nd-gen
# connections but the repository link is not properly configured. This script
# avoids that by building locally and pushing the image directly.
#
# USAGE:
#   ./scripts/deploy_docker_manual.sh
#
# REQUIREMENTS:
#   - Docker installed and running
#   - Google Cloud SDK (gcloud) installed
#   - GEMINI_API_KEY environment variable set
#   - Authenticated to Google Cloud (gcloud auth login)
#
# ============================================================================

set -e  # Exit on any error
set -o pipefail  # Ensure pipeline failures are caught

# ============================================================================
# Colors for output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ID="${GOOGLE_PROJECT_ID:-gen-lang-client-0472751146}"
REGION="${CLOUD_RUN_REGION:-europe-west1}"
SERVICE_NAME="${SERVICE_NAME:-gemini-api-service}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
IMAGE_TAG="${IMAGE_TAG:-${TIMESTAMP}}"

# Resource configuration
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-300}"

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo ""
echo "============================================"
echo -e "${GREEN}QEARIS Manual Docker Deployment${NC}"
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
log_step "Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI not found. Please install Google Cloud SDK."
    echo "  Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker."
    echo "  Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

# Check for GEMINI_API_KEY
if [ -z "${GEMINI_API_KEY}" ]; then
    log_error "GEMINI_API_KEY environment variable is not set."
    echo ""
    echo "Please set it before running this script:"
    echo "  export GEMINI_API_KEY='your-api-key'"
    echo ""
    echo "Or create a .env file and source it:"
    echo "  source .env"
    exit 1
fi

log_info "Prerequisites satisfied"

# ============================================================================
# Authenticate and Configure Google Cloud
# ============================================================================
log_step "Configuring Google Cloud..."

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
    log_warn "Not authenticated to Google Cloud. Running gcloud auth login..."
    gcloud auth login
fi

# Set project
gcloud config set project "${PROJECT_ID}" --quiet

# Enable required APIs
log_info "Enabling required APIs..."
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet

log_info "APIs enabled"

# Configure Docker for GCR
log_info "Configuring Docker for Container Registry..."
gcloud auth configure-docker --quiet

log_info "Docker configured for GCR"

# ============================================================================
# Navigate to Repository Root
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log_info "Working directory: ${REPO_ROOT}"

# Check if Dockerfile exists
if [ ! -f "dockerfile" ] && [ ! -f "Dockerfile" ]; then
    log_error "Dockerfile not found in repository root."
    exit 1
fi

# ============================================================================
# Build Docker Image
# ============================================================================
log_step "Building Docker image..."

# Determine Dockerfile path (case-insensitive check)
if [ -f "Dockerfile" ]; then
    DOCKERFILE_PATH="Dockerfile"
else
    DOCKERFILE_PATH="dockerfile"
fi

docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    -f "${DOCKERFILE_PATH}" \
    .

if [ $? -eq 0 ]; then
    log_info "Image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    log_error "Docker build failed."
    exit 1
fi

# ============================================================================
# Push to Container Registry
# ============================================================================
log_step "Pushing image to Google Container Registry..."

docker push "${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${IMAGE_NAME}:latest"

if [ $? -eq 0 ]; then
    log_info "Image pushed successfully"
else
    log_error "Failed to push image to Container Registry."
    exit 1
fi

# ============================================================================
# Deploy to Cloud Run
# ============================================================================
log_step "Deploying to Cloud Run in ${REGION}..."

# Build environment variables string
ENV_VARS="GEMINI_API_KEY=${GEMINI_API_KEY}"
ENV_VARS="${ENV_VARS},ENVIRONMENT=production"
ENV_VARS="${ENV_VARS},LOG_LEVEL=INFO"
ENV_VARS="${ENV_VARS},CLOUD_RUN_REGION=${REGION}"

# Add optional environment variables if set
if [ -n "${GEMINI_PROJECT_ID}" ]; then
    ENV_VARS="${ENV_VARS},GEMINI_PROJECT_ID=${GEMINI_PROJECT_ID}"
fi

if [ -n "${GEMINI_MODEL}" ]; then
    ENV_VARS="${ENV_VARS},GEMINI_MODEL=${GEMINI_MODEL}"
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
    --quiet

if [ $? -eq 0 ]; then
    log_info "Service deployed successfully"
else
    log_error "Deployment to Cloud Run failed."
    exit 1
fi

# ============================================================================
# Get Service URL and Verify
# ============================================================================
log_step "Retrieving service URL..."

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")

if [ -z "${SERVICE_URL}" ]; then
    log_error "Could not retrieve service URL."
    exit 1
fi

log_info "Service URL: ${SERVICE_URL}"

# ============================================================================
# Health Check
# ============================================================================
log_step "Running health check..."

# Wait a few seconds for service to stabilize
sleep 5

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" 2>/dev/null || echo "000")

if [ "${HEALTH_RESPONSE}" = "200" ]; then
    log_info "Health check passed (HTTP 200)"
else
    log_warn "Health check returned HTTP ${HEALTH_RESPONSE}"
    log_warn "Service may still be starting up. Check logs if issues persist."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo -e "${GREEN}Deployment Complete!${NC}"
echo "============================================"
echo ""
echo "Service Details:"
echo "  Name:        ${SERVICE_NAME}"
echo "  URL:         ${SERVICE_URL}"
echo "  Region:      ${REGION}"
echo "  Image:       ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Resources:   ${MEMORY} RAM, ${CPU} CPU"
echo ""
echo "Test endpoints:"
echo "  Health:      curl ${SERVICE_URL}/health"
echo "  API Docs:    ${SERVICE_URL}/docs"
echo ""
echo "Example research request:"
echo "  curl -X POST ${SERVICE_URL}/api/v1/research \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"How does quantum computing work?\", \"domains\": [\"quantum\"]}'"
echo ""
echo "Monitor logs:"
echo "  gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo -e "${GREEN}[SUCCESS]${NC} Deployment completed successfully!"
