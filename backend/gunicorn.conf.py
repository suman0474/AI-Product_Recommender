"""
Gunicorn Production Configuration
AI Product Recommender API

SAFE for:
- Python 3.11 / 3.13
- Background threads
- Azure SDK
- LLM workloads
"""

import multiprocessing
import os

# =============================================================================
# Server Socket Configuration
# =============================================================================

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
backlog = 2048


# =============================================================================
# Worker Configuration (LLM-SAFE)
# =============================================================================

# IMPORTANT:
# - Default workers = 1 (safe)
# - Scale horizontally instead of vertically for LLM apps
workers = int(os.getenv("GUNICORN_WORKERS", 1))

# Threaded workers (safe with I/O heavy workloads)
worker_class = "gthread"

# Threads per worker
threads = int(os.getenv("GUNICORN_THREADS", 4))


# =============================================================================
# Timeouts
# =============================================================================

timeout = int(os.getenv("GUNICORN_TIMEOUT", 3600))
graceful_timeout = 60
keepalive = 5

max_requests = 1000
max_requests_jitter = 100


# =============================================================================
# Security
# =============================================================================

limit_request_line = 4096
limit_request_field_size = 8190
limit_request_fields = 100


# =============================================================================
# Logging
# =============================================================================

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
capture_output = True

access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
    '"%(f)s" "%(a)s" %(D)s'
)


# =============================================================================
# Process Settings
# =============================================================================

proc_name = "aipr-gunicorn"
daemon = False
pidfile = "/tmp/gunicorn.pid"

# IMPORTANT:
# Disable shared tmp dir to avoid fork/thread corruption
worker_tmp_dir = None


# =============================================================================
# 🚨 CRITICAL FIX
# =============================================================================

# NEVER preload app when background threads exist
# This prevents fork-after-thread crashes
preload_app = False

# Disable reload in production
reload = False


# =============================================================================
# Lifecycle Hooks
# =============================================================================

def on_starting(server):
    server.log.info("=" * 70)
    server.log.info("Starting AI Product Recommender API")
    server.log.info(f"Bind: {bind}")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Threads per worker: {threads}")
    server.log.info(f"Worker class: {worker_class}")
    server.log.info(f"Timeout: {timeout}s")
    server.log.info("Preload: DISABLED (thread-safe)")
    server.log.info("=" * 70)


def when_ready(server):
    server.log.info("✓ Gunicorn master ready")


def post_fork(server, worker):
    server.log.info(f"✓ Worker spawned (pid={worker.pid})")


def worker_abort(worker):
    worker.log.warning(f"⚠ Worker aborted (pid={worker.pid})")


def on_exit(server):
    server.log.info("Gunicorn shutting down")