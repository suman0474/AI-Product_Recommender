# agentic/deep_agent/worker_runner.py
"""
Worker Runner - CLI Entry Point for Enrichment Workers

Usage:
    python -m agentic.deep_agent.worker_runner [options]

Options:
    --worker-id <id>      Worker identifier (auto-generated if not provided)
    --redis-url <url>     Redis connection URL (default: from env REDIS_URL)
    --max-tasks <n>       Maximum tasks to process (default: unlimited)
    --use-memory          Use in-memory queue/state (for testing)

Examples:
    # Run worker with Redis
    python -m agentic.deep_agent.worker_runner

    # Run worker with custom ID
    python -m agentic.deep_agent.worker_runner --worker-id worker-1

    # Run worker for testing (in-memory)
    python -m agentic.deep_agent.worker_runner --use-memory --max-tasks 10
"""

import logging
import sys
import argparse
import os
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrichment Worker - Processes tasks from distributed queue"
    )

    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker identifier (auto-generated if not provided)"
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL (default: from env REDIS_URL)"
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks to process before stopping (default: unlimited)"
    )

    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Use in-memory queue/state instead of Redis (for testing)"
    )

    parser.add_argument(
        "--queue-name",
        type=str,
        default="enrichment_tasks",
        help="Queue name (default: enrichment_tasks)"
    )

    return parser.parse_args()


def main():
    """Main entry point for worker runner."""
    args = parse_args()

    # Set Redis URL if provided
    if args.redis_url:
        os.environ["REDIS_URL"] = args.redis_url

    # Display startup info
    logger.info("="*70)
    logger.info("ENRICHMENT WORKER STARTING")
    logger.info("="*70)
    logger.info(f"Worker ID: {args.worker_id or 'auto-generated'}")
    logger.info(f"Use Redis: {not args.use_memory}")
    logger.info(f"Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379/0')}")
    logger.info(f"Queue Name: {args.queue_name}")
    logger.info(f"Max Tasks: {args.max_tasks or 'unlimited'}")
    logger.info("="*70)

    try:
        # Import components
        from .enrichment_worker import run_worker

        # Run worker
        run_worker(
            worker_id=args.worker_id,
            use_redis=not args.use_memory,
            max_tasks=args.max_tasks
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Worker failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
