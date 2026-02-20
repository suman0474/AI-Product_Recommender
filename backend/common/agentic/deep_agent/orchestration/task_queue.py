# agentic/deep_agent/task_queue.py
"""
Task Queue Layer for Distributed Enrichment

Provides distributed task queue using Redis for work distribution.
Workers pull tasks from queue and process independently.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

TaskType = Literal["initial_enrichment", "iterative_enrichment", "domain_extraction"]


class EnrichmentTask(dict):
    """
    Enrichment task structure.

    Required fields:
    - task_id: Unique task identifier
    - task_type: Type of enrichment task
    - session_id: Session this task belongs to
    - item_id: Item being enriched
    - payload: Task-specific data
    - priority: Lower = higher priority (0 is highest)
    - created_at: ISO timestamp
    - retry_count: Number of retries attempted
    """
    pass


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class EnrichmentTaskQueue(ABC):
    """
    Abstract base class for enrichment task queue.

    Provides interface for pushing/popping tasks in a distributed manner.
    """

    @abstractmethod
    def push_task(self, task: EnrichmentTask) -> None:
        """
        Add task to the queue.

        Args:
            task: EnrichmentTask dict
        """
        pass

    @abstractmethod
    def pop_task(self, timeout: int = 30) -> Optional[EnrichmentTask]:
        """
        Get next task from queue (blocking).

        Args:
            timeout: Maximum seconds to wait for a task

        Returns:
            EnrichmentTask or None if timeout
        """
        pass

    @abstractmethod
    def ack_task(self, task_id: str) -> None:
        """
        Acknowledge task completion.

        Args:
            task_id: Task identifier
        """
        pass

    @abstractmethod
    def nack_task(self, task_id: str, retry: bool = True) -> None:
        """
        Negative acknowledge (task failed).

        Args:
            task_id: Task identifier
            retry: If True, re-queue the task. If False, move to DLQ.
        """
        pass

    @abstractmethod
    def get_queue_depth(self) -> int:
        """
        Get number of pending tasks in queue.

        Returns:
            Number of tasks waiting
        """
        pass

    @abstractmethod
    def get_processing_count(self) -> int:
        """
        Get number of tasks currently being processed.

        Returns:
            Number of tasks in processing state
        """
        pass

    @abstractmethod
    def get_dlq_tasks(self) -> List[EnrichmentTask]:
        """
        Get all tasks in the dead letter queue.

        Returns:
            List of failed tasks
        """
        pass


# =============================================================================
# REDIS IMPLEMENTATION
# =============================================================================

class RedisTaskQueue(EnrichmentTaskQueue):
    """
    Redis-based task queue using sorted sets for priority queue.

    Features:
    - Priority-based processing (lower priority number = higher priority)
    - Blocking pop with timeout
    - Automatic retry with exponential backoff
    - Dead letter queue for failed tasks
    - Processing tracking
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        queue_name: str = "enrichment_tasks",
        max_retries: int = 3
    ):
        """
        Initialize Redis task queue.

        Args:
            redis_url: Redis connection URL (default: from env REDIS_URL)
            queue_name: Name of the queue (default: enrichment_tasks)
            max_retries: Maximum retry attempts (default: 3)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package not installed. Install with: pip install redis"
            )

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.queue_name = queue_name
        self.max_retries = max_retries
        self.processing_set = f"{queue_name}:processing"
        self.dlq_name = f"{queue_name}:dlq"

        try:
            self.redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis.ping()
            logger.info(f"[TaskQueue] Connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to connect to Redis: {e}")
            raise

    def push_task(self, task: EnrichmentTask) -> None:
        """Push task to Redis priority queue."""
        # Validate required fields
        required_fields = ["task_id", "task_type", "session_id", "item_id", "payload", "priority"]
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task missing required field: {field}")

        # Calculate score for priority queue
        # Lower priority = processed first
        # Within same priority, older tasks processed first
        base_score = task["priority"] * 1000000
        time_score = int(time.time())
        score = base_score + time_score

        # Add to sorted set
        task_json = json.dumps(task)
        self.redis.zadd(self.queue_name, {task_json: score})

        logger.info(
            f"[TaskQueue] Pushed task {task['task_id']} "
            f"(type: {task['task_type']}, priority: {task['priority']})"
        )

    def pop_task(self, timeout: int = 30) -> Optional[EnrichmentTask]:
        """Pop task from Redis queue (blocking)."""
        try:
            # Blocking pop from sorted set (lowest score first)
            result = self.redis.bzpopmin(self.queue_name, timeout=timeout)

            if not result:
                return None

            # Result format: (queue_name, task_json, score)
            _, task_json, _ = result
            task = json.loads(task_json)

            # Move to processing set
            task_id = task["task_id"]
            self.redis.sadd(self.processing_set, task_id)

            # Store task data with expiry (10 min timeout)
            self.redis.setex(
                f"{self.processing_set}:{task_id}",
                600,  # 10 minutes
                task_json
            )

            logger.info(f"[TaskQueue] Popped task {task_id}")
            return task

        except Exception as e:
            logger.error(f"[TaskQueue] Error popping task: {e}")
            return None

    def ack_task(self, task_id: str) -> None:
        """Acknowledge task completion."""
        # Remove from processing set
        self.redis.srem(self.processing_set, task_id)
        self.redis.delete(f"{self.processing_set}:{task_id}")

        logger.info(f"[TaskQueue] ACK task {task_id}")

    def nack_task(self, task_id: str, retry: bool = True) -> None:
        """Negative acknowledge - retry or move to DLQ."""
        # Get task data
        task_json = self.redis.get(f"{self.processing_set}:{task_id}")

        if not task_json:
            logger.warning(f"[TaskQueue] NACK failed - task {task_id} not found in processing")
            return

        task = json.loads(task_json)
        task["retry_count"] = task.get("retry_count", 0) + 1

        # Remove from processing
        self.redis.srem(self.processing_set, task_id)
        self.redis.delete(f"{self.processing_set}:{task_id}")

        if retry and task["retry_count"] < self.max_retries:
            # Re-queue with lower priority (exponential backoff)
            task["priority"] = task["priority"] + (2 ** task["retry_count"])
            task["last_error_at"] = datetime.utcnow().isoformat()

            self.push_task(task)

            logger.warning(
                f"[TaskQueue] NACK task {task_id} - retry {task['retry_count']}/{self.max_retries}"
            )
        else:
            # Move to dead letter queue
            self.redis.lpush(self.dlq_name, json.dumps(task))

            logger.error(
                f"[TaskQueue] NACK task {task_id} - moved to DLQ "
                f"(retries: {task['retry_count']})"
            )

    def get_queue_depth(self) -> int:
        """Get number of pending tasks."""
        return self.redis.zcard(self.queue_name)

    def get_processing_count(self) -> int:
        """Get number of tasks being processed."""
        return self.redis.scard(self.processing_set)

    def get_dlq_depth(self) -> int:
        """Get number of failed tasks in dead letter queue."""
        return self.redis.llen(self.dlq_name)

    def peek_dlq(self, limit: int = 10) -> List[EnrichmentTask]:
        """
        Peek at failed tasks in dead letter queue.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of failed tasks
        """
        task_jsons = self.redis.lrange(self.dlq_name, 0, limit - 1)
        return [json.loads(tj) for tj in task_jsons]

    def clear_dlq(self) -> int:
        """
        Clear all tasks from dead letter queue.

        Returns:
            Number of tasks cleared
        """
        count = self.redis.llen(self.dlq_name)
        self.redis.delete(self.dlq_name)
        logger.info(f"[TaskQueue] Cleared {count} tasks from DLQ")
        return count

    def get_dlq_tasks(self) -> List[EnrichmentTask]:
        """
        Get all tasks in the dead letter queue.

        Returns:
            List of failed tasks
        """
        dlq_count = self.redis.llen(self.dlq_name)
        if dlq_count == 0:
            return []
        task_jsons = self.redis.lrange(self.dlq_name, 0, dlq_count - 1)
        return [json.loads(tj) for tj in task_jsons]


# =============================================================================
# IN-MEMORY IMPLEMENTATION (For Testing)
# =============================================================================

class InMemoryTaskQueue(EnrichmentTaskQueue):
    """
    In-memory task queue for testing and development.

    WARNING: Queue is lost on process restart. Use only for testing.
    """

    def __init__(self, max_retries: int = 3):
        self.queue: deque = deque()  # Priority queue (sorted on pop)
        self.processing: Dict[str, EnrichmentTask] = {}
        self.dlq: List[EnrichmentTask] = []
        self.max_retries = max_retries
        logger.warning("[TaskQueue] Using InMemoryTaskQueue - not distributed!")

    def push_task(self, task: EnrichmentTask) -> None:
        # Insert task maintaining priority order
        self.queue.append(task)
        # Sort by priority (lower = higher priority)
        self.queue = deque(sorted(self.queue, key=lambda t: t["priority"]))
        logger.info(f"[TaskQueue] Pushed task {task['task_id']}")

    def pop_task(self, timeout: int = 30) -> Optional[EnrichmentTask]:
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.queue:
                task = self.queue.popleft()
                self.processing[task["task_id"]] = task
                logger.info(f"[TaskQueue] Popped task {task['task_id']}")
                return task

            time.sleep(0.1)  # Small sleep to avoid busy wait

        return None

    def ack_task(self, task_id: str) -> None:
        self.processing.pop(task_id, None)
        logger.info(f"[TaskQueue] ACK task {task_id}")

    def nack_task(self, task_id: str, retry: bool = True) -> None:
        task = self.processing.pop(task_id, None)

        if not task:
            return

        task["retry_count"] = task.get("retry_count", 0) + 1

        if retry and task["retry_count"] < self.max_retries:
            task["priority"] = task["priority"] + (2 ** task["retry_count"])
            self.push_task(task)
            logger.warning(f"[TaskQueue] NACK task {task_id} - retry {task['retry_count']}")
        else:
            self.dlq.append(task)
            logger.error(f"[TaskQueue] NACK task {task_id} - moved to DLQ")

    def get_queue_depth(self) -> int:
        return len(self.queue)

    def get_processing_count(self) -> int:
        return len(self.processing)

    def get_dlq_tasks(self) -> List[EnrichmentTask]:
        """Get all tasks in the dead letter queue."""
        return list(self.dlq)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_task_queue(use_redis: bool = True, queue_name: str = "enrichment_tasks") -> EnrichmentTaskQueue:
    """
    Factory function to get appropriate task queue.

    Args:
        use_redis: If True, use Redis (default). If False, use in-memory.
        queue_name: Name of the queue (for Redis)

    Returns:
        EnrichmentTaskQueue instance
    """
    if use_redis:
        try:
            return RedisTaskQueue(queue_name=queue_name)
        except Exception as e:
            logger.warning(f"[TaskQueue] Failed to create Redis queue: {e}")
            logger.warning("[TaskQueue] Falling back to InMemoryTaskQueue")
            return InMemoryTaskQueue()
    else:
        return InMemoryTaskQueue()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnrichmentTask",
    "EnrichmentTaskQueue",
    "RedisTaskQueue",
    "InMemoryTaskQueue",
    "get_task_queue"
]
