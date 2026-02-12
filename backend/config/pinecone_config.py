"""
Pinecone Configuration
"""
import os
import logging

logger = logging.getLogger(__name__)

class PineconeConfig:
    """Centralized Pinecone configuration management"""

    # API Configuration
    API_KEY = os.getenv("PINECONE_API_KEY")

    # Index Configuration
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
    DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "example-namespace")

    # Search Settings
    DEFAULT_TOP_K = int(os.getenv("PINECONE_TOP_K", "10"))
    RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")

    # Batch Settings
    BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "96"))
    BATCH_DELAY = float(os.getenv("PINECONE_BATCH_DELAY", "0.1"))

    # Retry Settings
    MAX_RETRIES = int(os.getenv("PINECONE_MAX_RETRIES", "5"))
    RETRY_BASE_DELAY = float(os.getenv("PINECONE_RETRY_BASE_DELAY", "1.0"))
    RETRY_MAX_DELAY = float(os.getenv("PINECONE_RETRY_MAX_DELAY", "60.0"))

    # Indexing Settings
    INDEXING_TIMEOUT = int(os.getenv("PINECONE_INDEXING_TIMEOUT", "60"))
    INDEXING_POLL_INTERVAL = int(os.getenv("PINECONE_INDEXING_POLL_INTERVAL", "2"))

    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.

        Raises:
            ValueError: If required configuration is missing
        """
        if not cls.API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Please set it in your .env file or environment variables."
            )

        # Validate numeric settings
        if cls.BATCH_SIZE < 1 or cls.BATCH_SIZE > 96:
            raise ValueError(
                f"PINECONE_BATCH_SIZE must be between 1 and 96 (got {cls.BATCH_SIZE})"
            )

        if cls.MAX_RETRIES < 1:
            raise ValueError(
                f"PINECONE_MAX_RETRIES must be at least 1 (got {cls.MAX_RETRIES})"
            )

        if cls.DEFAULT_TOP_K < 1:
            raise ValueError(
                f"PINECONE_TOP_K must be at least 1 (got {cls.DEFAULT_TOP_K})"
            )

    @classmethod
    def print_config(cls):
        """Print current configuration (safe - hides API key)"""
        print("=" * 60)
        print("PINECONE CONFIGURATION")
        print("=" * 60)
        print(f"API Key: {'*' * 20}{cls.API_KEY[-4:] if cls.API_KEY else 'NOT SET'}")
        print(f"Index Name: {cls.INDEX_NAME}")
        print(f"Default Namespace: {cls.DEFAULT_NAMESPACE}")
        print(f"Default Top K: {cls.DEFAULT_TOP_K}")
        print(f"Rerank Model: {cls.RERANK_MODEL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Batch Delay: {cls.BATCH_DELAY}s")
        print(f"Max Retries: {cls.MAX_RETRIES}")
        print(f"Retry Base Delay: {cls.RETRY_BASE_DELAY}s")
        print(f"Retry Max Delay: {cls.RETRY_MAX_DELAY}s")
        print(f"Indexing Timeout: {cls.INDEXING_TIMEOUT}s")
        print(f"Indexing Poll Interval: {cls.INDEXING_POLL_INTERVAL}s")
        print("=" * 60)
