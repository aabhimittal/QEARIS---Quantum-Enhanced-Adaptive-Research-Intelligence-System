"""
Logging Configuration

PURPOSE: Structured logging for production
FORMAT: JSON for easy parsing
LEVELS: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import json
import logging
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging

    OUTPUT FORMAT:
    --------------
    {
        "timestamp": "2025-01-01T12:00:00Z",
        "level": "INFO",
        "logger": "qearis.api",
        "message": "Request received",
        "extra": {...}
    }

    WHY JSON?
    ---------
    - Machine readable
    - Easy to parse
    - Searchable in log aggregators
    - Standard format
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Setup application logging

    PARAMETERS:
    -----------
    level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    json_format: Use JSON formatter (True for production)

    USAGE:
    ------
    # Development
    setup_logging(level="DEBUG", json_format=False)

    # Production
    setup_logging(level="INFO", json_format=True)
    """
    # Get log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logging.info(f"Logging configured: level={level}, json={json_format}")
