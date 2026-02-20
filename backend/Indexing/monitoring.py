"""
Indexing Agent — Monitoring & Telemetry
=======================================
Provides observability, logging, and performance tracking.
Simplified and inlined to a single file.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class _WorkflowMonitor:
    """Monitor workflow execution and performance."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize workflow monitor."""
        # Logs are stored in the Indexing/logs directory
        self.log_dir = log_dir or Path(__file__).parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_run = None
        self.start_time = None
        self.agent_timings = {}

    def start_run(self, product_type: str, run_id: Optional[str] = None):
        """Start monitoring a workflow run."""
        self.current_run = run_id or f"idx_{int(time.time())}"
        self.start_time = time.time()
        self.agent_timings = {}

        logger.info(f"Started monitoring run: {self.current_run} for {product_type}")

    def start_agent(self, agent_name: str):
        """Start timing an agent."""
        if agent_name not in self.agent_timings:
            self.agent_timings[agent_name] = {
                'start': time.time(),
                'end': None,
                'duration': None
            }

    def end_agent(self, agent_name: str, status: str = 'success', metadata: Optional[Dict] = None):
        """End timing an agent."""
        if agent_name in self.agent_timings:
            self.agent_timings[agent_name]['end'] = time.time()
            self.agent_timings[agent_name]['duration'] = (
                self.agent_timings[agent_name]['end'] -
                self.agent_timings[agent_name]['start']
            )
            self.agent_timings[agent_name]['status'] = status

            if metadata:
                self.agent_timings[agent_name]['metadata'] = metadata

            logger.info(
                f"Agent {agent_name} completed in "
                f"{self.agent_timings[agent_name]['duration']:.2f}s - {status}"
            )

    def end_run(self, final_state: Dict[str, Any]):
        """End monitoring and save report."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        report = {
            'run_id': self.current_run,
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'agent_timings': self.agent_timings,
            'final_quality_score': final_state.get('final_quality_score', 0),
            'deployment_ready': final_state.get('deployment_ready', False),
            'product_type': final_state.get('product_type', 'unknown')
        }

        # Save report
        report_path = self.log_dir / f"{self.current_run}_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Run completed in {total_duration:.2f}s - Report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save monitoring report: {e}")

        return report

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_duration': time.time() - self.start_time if self.start_time else 0,
            'agent_timings': self.agent_timings,
            'agents_completed': len([a for a in self.agent_timings.values() if a.get('status') == 'success'])
        }


class _MetricsCollector:
    """Collect and aggregate metrics across runs."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_duration': 0,
            'avg_quality_score': 0,
            'agent_success_rates': {}
        }

    def record_run(self, report: Dict[str, Any]):
        """Record a completed run."""
        self.metrics['total_runs'] += 1

        if report.get('deployment_ready', False):
            self.metrics['successful_runs'] += 1
        else:
            self.metrics['failed_runs'] += 1

        # Update averages
        self.metrics['avg_duration'] = (
            (self.metrics['avg_duration'] * (self.metrics['total_runs'] - 1) +
             report['total_duration']) / self.metrics['total_runs']
        )

        quality_score = report.get('final_quality_score', 0)
        self.metrics['avg_quality_score'] = (
            (self.metrics['avg_quality_score'] * (self.metrics['total_runs'] - 1) +
             quality_score) / self.metrics['total_runs']
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics


# Global instances (Internal use)
_workflow_monitor = None
_metrics_collector = _MetricsCollector()


def get_monitor(log_dir: Optional[Path] = None) -> _WorkflowMonitor:
    """Internal helper to get the global workflow monitor instance."""
    global _workflow_monitor
    if _workflow_monitor is None:
        _workflow_monitor = _WorkflowMonitor(log_dir)
    return _workflow_monitor


def get_collector() -> _MetricsCollector:
    """Internal helper to get the global metrics collector instance."""
    return _metrics_collector
