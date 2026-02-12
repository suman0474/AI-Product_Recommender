"""
Agentic Configuration
"""
import os
import logging

logger = logging.getLogger(__name__)

class AgenticConfig:
    """Centralized configuration for agentic workflows and LLM operations"""

    # LLM Model Configuration (Multi-tier approach)
    # DEFAULT_MODEL: General purpose tasks (balanced speed/quality)
    # FLASH_MODEL (Specific): Quality-critical tasks where speed is preferred (ranking, complex analysis)
    # LITE_MODEL: Fast, lightweight tasks (validation, simple queries)
    # FALLBACK_MODEL: OpenAI fallback when Google quotas are exhausted
    DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
    FLASH_MODEL = os.getenv("FLASH_LLM_MODEL", "gemini-2.5-flash")
    SPEC_GEN_MODEL = os.getenv("SPEC_GEN_LLM_MODEL", "gemini-2.5-flash")  # Fast model for spec generation
    LITE_MODEL = os.getenv("LITE_LLM_MODEL", "gemini-2.0-flash-lite")
    FALLBACK_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-4o-mini")

    # Google API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Model Temperature Settings
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    CREATIVE_TEMPERATURE = float(os.getenv("CREATIVE_TEMPERATURE", "0.7"))

    # Timeout Configuration (in seconds)
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    LONG_TIMEOUT = int(os.getenv("LONG_TIMEOUT", "120"))
    PARALLEL_ANALYSIS_TIMEOUT = int(os.getenv("PARALLEL_ANALYSIS_TIMEOUT", "300"))

    # Parallel Processing Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
    MAX_VENDORS_TO_ANALYZE = int(os.getenv("MAX_VENDORS_TO_ANALYZE", "5"))

    # ChromaDB Configuration
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
    CHROMADB_TIMEOUT = int(os.getenv("CHROMADB_TIMEOUT", "30"))

    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

    # Workflow Configuration
    MAX_WORKFLOW_RETRIES = int(os.getenv("MAX_WORKFLOW_RETRIES", "2"))
    WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "300"))

    # RAG Configuration
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))
    
    # RAG Timeout Configuration (20 minutes = 1200 seconds for RAG operations)
    RAG_TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "1200"))  # 20 min
    ORCHESTRATOR_TIMEOUT_SECONDS = int(os.getenv("ORCHESTRATOR_TIMEOUT_SECONDS", "1200"))  # 20 min
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))  # 5 min
    PINECONE_TIMEOUT_SECONDS = int(os.getenv("PINECONE_TIMEOUT_SECONDS", "60"))  # 1 min
    
    # Rate Limiting Configuration
    LLM_CALLS_PER_MINUTE = int(os.getenv("LLM_CALLS_PER_MINUTE", "60"))
    LLM_RATE_LIMIT_BURST = int(os.getenv("LLM_RATE_LIMIT_BURST", "15"))
    
    # RAG Cache Configuration
    RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "300"))  # 5 min
    RAG_CACHE_MAX_SIZE = int(os.getenv("RAG_CACHE_MAX_SIZE", "500"))
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RESET_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "60"))

    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required. "
                "Please set it in your .env file or environment variables."
            )

        # Validate temperature ranges
        if not (0.0 <= cls.DEFAULT_TEMPERATURE <= 2.0):
            raise ValueError(
                f"DEFAULT_TEMPERATURE must be between 0.0 and 2.0 (got {cls.DEFAULT_TEMPERATURE})"
            )

        if not (0.0 <= cls.CREATIVE_TEMPERATURE <= 2.0):
            raise ValueError(
                f"CREATIVE_TEMPERATURE must be between 0.0 and 2.0 (got {cls.CREATIVE_TEMPERATURE})"
            )

        # Validate positive integers
        if cls.MAX_WORKERS < 1:
            raise ValueError(f"MAX_WORKERS must be at least 1 (got {cls.MAX_WORKERS})")

        if cls.DEFAULT_TIMEOUT < 1:
            raise ValueError(f"DEFAULT_TIMEOUT must be at least 1 (got {cls.DEFAULT_TIMEOUT})")

        if cls.RAG_TOP_K < 1:
            raise ValueError(f"RAG_TOP_K must be at least 1 (got {cls.RAG_TOP_K})")

    @classmethod
    def print_config(cls):
        """Print current configuration (safe - hides API key)"""
        logger.info("=" * 60)
        logger.info("AGENTIC CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Google API Key: {'*' * 20}{cls.GOOGLE_API_KEY[-4:] if cls.GOOGLE_API_KEY else 'NOT SET'}")
        logger.info(f"Default Model: {cls.DEFAULT_MODEL}")
        logger.info(f"Flash Model: {cls.FLASH_MODEL}")
        logger.info(f"Lite Model: {cls.LITE_MODEL}")
        logger.info(f"Fallback Model: {cls.FALLBACK_MODEL}")
        logger.info(f"Default Temperature: {cls.DEFAULT_TEMPERATURE}")
        logger.info(f"Creative Temperature: {cls.CREATIVE_TEMPERATURE}")
        logger.info(f"Default Timeout: {cls.DEFAULT_TIMEOUT}s")
        logger.info(f"Long Timeout: {cls.LONG_TIMEOUT}s")
        logger.info(f"Parallel Analysis Timeout: {cls.PARALLEL_ANALYSIS_TIMEOUT}s")
        logger.info(f"Max Workers: {cls.MAX_WORKERS}")
        logger.info(f"Max Vendors to Analyze: {cls.MAX_VENDORS_TO_ANALYZE}")
        logger.info(f"ChromaDB Host: {cls.CHROMADB_HOST}")
        logger.info(f"ChromaDB Port: {cls.CHROMADB_PORT}")
        logger.info(f"ChromaDB Timeout: {cls.CHROMADB_TIMEOUT}s")
        logger.info(f"RAG Top K: {cls.RAG_TOP_K}")
        logger.info(f"RAG Score Threshold: {cls.RAG_SCORE_THRESHOLD}")
        logger.info("-" * 60)
        logger.info("RAG TIMEOUT & PERFORMANCE SETTINGS")
        logger.info("-" * 60)
        logger.info(f"RAG Timeout: {cls.RAG_TIMEOUT_SECONDS}s ({cls.RAG_TIMEOUT_SECONDS // 60} min)")
        logger.info(f"Orchestrator Timeout: {cls.ORCHESTRATOR_TIMEOUT_SECONDS}s ({cls.ORCHESTRATOR_TIMEOUT_SECONDS // 60} min)")
        logger.info(f"LLM Timeout: {cls.LLM_TIMEOUT_SECONDS}s")
        logger.info(f"Pinecone Timeout: {cls.PINECONE_TIMEOUT_SECONDS}s")
        logger.info(f"LLM Calls Per Minute: {cls.LLM_CALLS_PER_MINUTE}")
        logger.info(f"LLM Rate Limit Burst: {cls.LLM_RATE_LIMIT_BURST}")
        logger.info(f"RAG Cache TTL: {cls.RAG_CACHE_TTL_SECONDS}s")
        logger.info(f"RAG Cache Max Size: {cls.RAG_CACHE_MAX_SIZE}")
        logger.info(f"Circuit Breaker Threshold: {cls.CIRCUIT_BREAKER_THRESHOLD}")
        logger.info(f"Circuit Breaker Reset Timeout: {cls.CIRCUIT_BREAKER_RESET_TIMEOUT}s")
        logger.info("=" * 60)

