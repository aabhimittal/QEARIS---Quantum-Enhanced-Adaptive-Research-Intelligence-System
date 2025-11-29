"""
QEARIS Configuration Management
Centralized settings using Pydantic
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings
    
    Loads from environment variables with fallback to .env file
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )
    
    # Gemini API
    GEMINI_API_KEY: str
    GEMINI_PROJECT_ID: str = "gen-lang-client-0472751146"
    GEMINI_PROJECT_NUMBER: str = "412097861656"
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_TOKENS: int = 8192
    
    # System Configuration
    MAX_PARALLEL_AGENTS: int = 4
    VALIDATION_THRESHOLD: float = 0.75
    SYNTHESIS_ITERATIONS: int = 3
    AGENT_TIMEOUT: int = 300
    
    # Quantum Optimization
    QUANTUM_TEMPERATURE: float = 1.0
    QUANTUM_ITERATIONS: int = 100
    COOLING_RATE: float = 0.95
    
    # RAG Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    
    # Memory Configuration
    MEMORY_RETENTION_DAYS: int = 30
    MAX_MEMORY_ITEMS: int = 1000
    CONTEXT_WINDOW: int = 100000
    
    # MCP Configuration
    MCP_TOOL_TIMEOUT: int = 60
    MCP_RETRY_ATTEMPTS: int = 3
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    
    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True
    ENABLE_METRICS: bool = True
    
    # Deployment
    ENVIRONMENT: str = "production"
    CLOUD_RUN_REGION: str = "us-central1"
    CLOUD_RUN_SERVICE_NAME: str = "qearis"
    
    # Optional APIs
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None


# Global settings instance
settings = Settings()
