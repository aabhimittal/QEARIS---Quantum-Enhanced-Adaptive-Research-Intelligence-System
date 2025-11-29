"""
Metrics Collection and Reporting

PURPOSE: Track system performance and behavior
EXPORT: Prometheus-compatible metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Single metric data point"""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and expose metrics

    METRICS COLLECTED:
    ------------------
    1. Request metrics (count, duration, status)
    2. Agent metrics (tasks, success rate, avg time)
    3. System metrics (memory, CPU, errors)
    4. Business metrics (confidence, sources, quality)

    FORMAT: Prometheus-compatible
    """

    def __init__(self):
        self.metrics: Dict[str, list] = {"requests": [], "agents": [], "system": [], "business": []}

        # Counters
        self.request_count = 0
        self.request_success = 0
        self.request_failure = 0

        # Timers
        self.request_times = []

        logger.info("MetricsCollector initialized")

    def record_request(self, endpoint: str, duration: float, status: str, status_code: int = 200):
        """Record API request metrics"""
        self.request_count += 1

        if status == "success":
            self.request_success += 1
        else:
            self.request_failure += 1

        self.request_times.append(duration)

        metric = Metric(
            name="http_request",
            value=duration,
            labels={"endpoint": endpoint, "status": status, "status_code": str(status_code)},
        )

        self.metrics["requests"].append(metric)

    def record_agent_task(self, agent_id: str, agent_type: str, duration: float, success: bool):
        """Record agent task execution"""
        metric = Metric(
            name="agent_task",
            value=duration,
            labels={"agent_id": agent_id, "agent_type": agent_type, "success": str(success)},
        )

        self.metrics["agents"].append(metric)

    def record_business_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record business metric (confidence, quality, etc.)"""
        metric = Metric(name=name, value=value, labels=labels or {})

        self.metrics["business"].append(metric)

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format

        FORMAT:
        -------
        # TYPE metric_name type
        # HELP metric_name description
        metric_name{label="value"} value timestamp
        """
        lines = []

        # Request metrics
        lines.append("# TYPE http_requests_total counter")
        lines.append("# HELP http_requests_total Total HTTP requests")
        lines.append(f"http_requests_total {self.request_count}")

        lines.append("# TYPE http_requests_success counter")
        lines.append(f"http_requests_success {self.request_success}")

        lines.append("# TYPE http_requests_failure counter")
        lines.append(f"http_requests_failure {self.request_failure}")

        # Average request time
        if self.request_times:
            avg_time = sum(self.request_times) / len(self.request_times)
            lines.append("# TYPE http_request_duration_seconds gauge")
            lines.append(f"http_request_duration_seconds {avg_time:.3f}")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "requests": {
                "total": self.request_count,
                "success": self.request_success,
                "failure": self.request_failure,
                "success_rate": self.request_success / max(self.request_count, 1),
                "avg_duration": (
                    sum(self.request_times) / max(len(self.request_times), 1)
                    if self.request_times
                    else 0
                ),
            },
            "agents": {"tasks_executed": len(self.metrics["agents"])},
            "business": {"metrics_collected": len(self.metrics["business"])},
        }

    def reset(self):
        """Reset all metrics"""
        self.metrics = {"requests": [], "agents": [], "system": [], "business": []}
        self.request_count = 0
        self.request_success = 0
        self.request_failure = 0
        self.request_times = []


# Global metrics collector
metrics_collector = MetricsCollector()
