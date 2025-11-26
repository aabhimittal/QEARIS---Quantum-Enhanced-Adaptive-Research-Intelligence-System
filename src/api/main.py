"""
QEARIS FastAPI Application
Main entry point for the REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from .routes import router
from .schemas import HealthResponse
from ..config import settings
from ..observability.metrics import metrics_manager
from ..observability.logging_config import setup_logging

# Setup logging
logger = setup_logging()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    
    STARTUP:
    - Initialize connections
    - Load models
    - Start background tasks
    
    SHUTDOWN:
    - Close connections
    - Save state
    - Cleanup resources
    """
    # Startup
    logger.info("🚀 Starting QEARIS API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Using Gemini Model: {settings.GEMINI_MODEL}")
    
    # Initialize components
    try:
        # Initialize metrics
        await metrics_manager.initialize()
        logger.info("✅ Metrics manager initialized")
        
        # Add more initialization as needed
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down QEARIS API...")
    await metrics_manager.shutdown()

# Create FastAPI app
app = FastAPI(
    title="QEARIS API",
    description="Quantum-Enhanced Adaptive Research Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    
    Returns:
    --------
    Service health status and metrics
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "model": settings.GEMINI_MODEL
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "QEARIS",
        "version": "1.0.0",
        "description": "Quantum-Enhanced Adaptive Research Intelligence System",
        "docs": "/docs",
        "health": "/health",
        "competition": "Kaggle Capstone Project - Google AI"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return await metrics_manager.get_metrics_summary()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS if not settings.API_RELOAD else 1
    )
