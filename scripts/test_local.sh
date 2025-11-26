#!/bin/bash

# ============================================================================
# Local Testing Script
# ============================================================================

set -e

echo "🧪 Running QEARIS Tests"
echo "======================="

# Activate virtual environment
source venv/bin/activate

# Run linting
echo "1. Running linting..."
black --check src/ tests/
flake8 src/ tests/ --max-line-length=100

# Run type checking
echo "2. Running type checking..."
mypy src/ --ignore-missing-imports

# Run unit tests
echo "3. Running unit tests..."
pytest tests/ -v --cov=src --cov-report=term-missing

# Run integration tests
echo "4. Starting test server..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 &
SERVER_PID=$!

sleep 5

# Test health endpoint
echo "5. Testing health endpoint..."
curl -f http://localhost:8080/health || exit 1

# Test research endpoint
echo "6. Testing research endpoint..."
curl -X POST http://localhost:8080/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query":"Test query","domains":["test"]}' || exit 1

# Cleanup
echo "7. Cleaning up..."
kill $SERVER_PID

echo ""
echo "✅ All tests passed!"
