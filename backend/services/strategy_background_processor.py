"""
Strategy Background Processor
==============================
Handles asynchronous processing of strategy document uploads.

Flow:
1. Upload endpoint stores raw file in MongoDB with "pending" status
2. Background task extracts vendor data using LLM
3. Updates MongoDB document with extracted data and "completed" status

Status values:
- "pending": Waiting for processing
- "processing": Currently extracting data
- "completed": Successfully extracted
- "failed": Extraction failed
"""

import logging
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from bson import ObjectId

logger = logging.getLogger(__name__)

# Thread pool for background tasks
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """
    Get or create the global thread pool executor.

    Returns:
        ThreadPoolExecutor instance
    """
    global _executor

    if _executor is None:
        with _executor_lock:
            if _executor is None:
                # Create executor with 3 worker threads
                # This allows up to 3 documents to be processed concurrently
                _executor = ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="strategy_processor"
                )
                logger.info("[STRATEGY_BG] Thread pool initialized with 3 workers")

    return _executor


def process_strategy_document_async(
    document_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: int
):
    """
    Submit a strategy document for background processing.

    Args:
        document_id: MongoDB document ID
        file_bytes: File content
        filename: Original filename
        content_type: MIME type
        user_id: User ID
    """
    executor = get_executor()

    # Submit task to thread pool
    future = executor.submit(
        _process_document_task,
        document_id=document_id,
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        user_id=user_id
    )

    logger.info(f"[STRATEGY_BG] Submitted document {document_id} for background processing")

    # Optional: Add callback for when task completes
    future.add_done_callback(
        lambda f: _on_task_complete(document_id, f)
    )


def _process_document_task(
    document_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: int
):
    """
    Background task: Extract vendor data from document.

    This runs in a separate thread and updates MongoDB when done.

    Args:
        document_id: MongoDB document ID
        file_bytes: File content
        filename: Original filename
        content_type: MIME type
        user_id: User ID
    """
    logger.info(f"[STRATEGY_BG] Starting extraction for document {document_id}")

    try:
        from core.mongodb_manager import mongodb_manager
        from services.strategy_document_extractor import get_strategy_extractor

        strategy_collection = mongodb_manager.get_collection('stratergy')

        if not strategy_collection:
            raise RuntimeError("MongoDB strategy collection not available")

        # Update status to "processing"
        strategy_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "processing",
                    "processing_started_at": datetime.utcnow().isoformat()
                }
            }
        )

        logger.info(f"[STRATEGY_BG] Extracting vendor data from {filename}...")

        # Extract vendor data using LLM (with keyword standardization)
        extractor = get_strategy_extractor()
        vendor_data = extractor.extract_from_file(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            user_id=user_id  # Pass user_id for user-scoped standardization
        )

        logger.info(f"[STRATEGY_BG] Extracted {len(vendor_data)} vendor records from {filename}")

        # Update MongoDB with extracted data and remove file_data to save space
        strategy_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "completed",
                    "data": vendor_data,
                    "vendor_count": len(vendor_data),
                    "processing_completed_at": datetime.utcnow().isoformat(),
                    "error": None
                },
                "$unset": {
                    "file_data": ""  # Remove the base64 file data to save space
                }
            }
        )

        logger.info(f"[STRATEGY_BG] ✓ Successfully processed document {document_id}")

    except Exception as e:
        logger.error(f"[STRATEGY_BG] ✗ Processing failed for document {document_id}: {e}")

        # Update MongoDB with error status
        try:
            from core.mongodb_manager import mongodb_manager
            strategy_collection = mongodb_manager.get_collection('stratergy')

            if strategy_collection:
                strategy_collection.update_one(
                    {"_id": ObjectId(document_id)},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(e),
                            "processing_completed_at": datetime.utcnow().isoformat()
                        }
                    }
                )
        except Exception as update_error:
            logger.error(f"[STRATEGY_BG] Failed to update error status: {update_error}")


def _on_task_complete(document_id: str, future):
    """
    Callback when background task completes.

    Args:
        document_id: MongoDB document ID
        future: Completed Future object
    """
    try:
        # Check if task raised an exception
        exception = future.exception()
        if exception:
            logger.error(f"[STRATEGY_BG] Task for document {document_id} raised exception: {exception}")
        else:
            logger.info(f"[STRATEGY_BG] Task for document {document_id} completed successfully")
    except Exception as e:
        logger.error(f"[STRATEGY_BG] Error in task completion callback: {e}")


def get_processing_status(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current processing status of a document.

    Args:
        document_id: MongoDB document ID

    Returns:
        Dict with status information or None if not found
    """
    try:
        from core.mongodb_manager import mongodb_manager

        strategy_collection = mongodb_manager.get_collection('stratergy')

        if not strategy_collection:
            return None

        document = strategy_collection.find_one(
            {"_id": ObjectId(document_id)},
            {
                "status": 1,
                "vendor_count": 1,
                "error": 1,
                "uploaded_at": 1,
                "processing_started_at": 1,
                "processing_completed_at": 1,
                "file_name": 1
            }
        )

        if not document:
            return None

        # Convert ObjectId to string
        document['_id'] = str(document['_id'])

        return document

    except Exception as e:
        logger.error(f"[STRATEGY_BG] Failed to get status for document {document_id}: {e}")
        return None


def shutdown_executor():
    """
    Gracefully shutdown the thread pool executor.

    Call this on application shutdown to ensure all tasks complete.
    """
    global _executor

    if _executor is not None:
        logger.info("[STRATEGY_BG] Shutting down thread pool executor...")
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("[STRATEGY_BG] Thread pool executor shut down")
