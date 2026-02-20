"""
Application Initialization Module
==================================

Single point of initialization for the entire application.
Ensures environment variables are loaded once, configurations are validated,
and all singletons are properly initialized.

This module MUST be called FIRST before any other imports in main.py.
"""

import os
import logging
import atexit
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 2: APPLICATION-LEVEL CLEANUP HANDLERS
# =============================================================================

def _cleanup_bounded_caches():
    """
    Cleanup all bounded caches on application shutdown.
    Prevents memory leaks and logs final statistics.
    """
    try:
        from common.infrastructure.caching.bounded_cache import (
            cleanup_all_caches,
            get_all_cache_stats,
            get_registry_summary
        )

        logger.info("[CLEANUP] Cleaning up bounded caches...")

        # Cleanup expired entries in all caches
        cleanup_results = cleanup_all_caches()

        # Get final statistics
        stats = get_all_cache_stats()
        summary = get_registry_summary()

        logger.info(f"[CLEANUP] Cache cleanup results: {cleanup_results}")
        logger.info(f"[CLEANUP] Registry summary: {summary}")

        # Log per-cache stats
        if stats:
            logger.info("[CLEANUP] Final cache statistics:")
            for cache_name, cache_stats in stats.items():
                logger.info(f"[CLEANUP]   {cache_name}: {cache_stats}")

        logger.info("[CLEANUP] ✓ Bounded caches cleanup complete")
    except ImportError:
        logger.debug("[CLEANUP] BoundedCache module not available (not initialized)")
    except Exception as e:
        logger.error(f"[CLEANUP] Error cleaning up bounded caches: {e}", exc_info=True)


def _shutdown_global_executor():
    """
    Shutdown the global thread pool executor.
    Ensures all pending tasks complete before shutdown.
    """
    try:
        from common.infrastructure.state.execution.executor_manager import shutdown_global_executor, get_executor_stats

        logger.info("[CLEANUP] Shutting down global executor...")

        # Get stats before shutdown
        stats = get_executor_stats()
        logger.info(f"[CLEANUP] Global executor stats: {stats}")

        # Shutdown with wait=True to ensure all tasks complete
        shutdown_global_executor(wait=True)

        logger.info("[CLEANUP] ✓ Global executor shutdown complete")
    except ImportError:
        logger.debug("[CLEANUP] Global executor not available (not initialized)")
    except Exception as e:
        logger.error(f"[CLEANUP] Error shutting down global executor: {e}", exc_info=True)




def register_cleanup_handlers():
    """
    Register cleanup handlers for application shutdown.
    These run in reverse order (LIFO) when the application exits.

    Cleanup sequence:
    1. Stop background cleanup task
    2. Cleanup bounded caches
    3. Shutdown global executor
    4. Log final status
    """
    logger.info("[INIT] Step 4: Registering cleanup handlers...")

    # Register handlers in reverse order of initialization
    # They will execute in reverse order (LIFO)
    atexit.register(_shutdown_global_executor)
    atexit.register(_cleanup_bounded_caches)

    # Register final status logging
    def _log_final_status():
        logger.info("="*70)
        logger.info("[CLEANUP] Application shutdown complete")
        logger.info("="*70)

    atexit.register(_log_final_status)

    logger.info("[INIT] ✓ Cleanup handlers registered (LIFO order)")


def initialize_application() -> bool:
    """
    Single point of initialization for the entire application.
    Called ONCE at startup in main.py before any other imports.

    Returns:
        bool: True if initialization successful

    Raises:
        RuntimeError: If critical configuration is missing or invalid
    """
    logger.info("="*70)
    logger.info("[INIT] Starting application initialization")
    logger.info("="*70)

    # Step 1: Load environment variables ONCE
    logger.info("[INIT] Step 1: Loading environment variables...")
    load_dotenv()
    logger.info("[INIT] ✓ Environment variables loaded from .env")

    # Step 1.5: Configure LangSmith tracing (if API key available)
    logger.info("[INIT] Step 1.5: Configuring observability...")
    try:
        from common.config.langsmith_config import configure_langsmith
        langsmith_enabled = configure_langsmith()
        if langsmith_enabled:
            logger.info("[INIT] ✓ LangSmith tracing enabled")
        else:
            logger.info("[INIT] ⚠ LangSmith tracing not configured (set LANGSMITH_API_KEY to enable)")
    except ImportError:
        logger.warning("[INIT] ⚠ LangSmith config module not available")
    except Exception as e:
        logger.warning(f"[INIT] ⚠ LangSmith configuration failed: {e}")

    # Step 2: Validate all configurations
    logger.info("[INIT] Step 2: Validating configurations...")
    try:
        validate_all_configs()
        logger.info("[INIT] ✓ All configurations validated")
    except RuntimeError as e:
        logger.error(f"[INIT] ✗ Configuration validation failed: {e}")
        raise

    # Step 3: Initialize singletons
    logger.info("[INIT] Step 3: Initializing singletons...")
    try:
        initialize_singletons()
        logger.info("[INIT] ✓ All singletons initialized")
    except Exception as e:
        logger.error(f"[INIT] ✗ Singleton initialization failed: {e}")
        raise RuntimeError(f"Singleton initialization failed: {e}")

    # Step 4: Register cleanup handlers (Phase 2)
    try:
        register_cleanup_handlers()
        logger.info("[INIT] ✓ Cleanup handlers registered")
    except Exception as e:
        logger.error(f"[INIT] ✗ Cleanup handler registration failed: {e}")
        raise RuntimeError(f"Cleanup handler registration failed: {e}")

    logger.info("="*70)
    logger.info("[INIT] Application initialization complete")
    logger.info("="*70)

    return True


def validate_all_configs() -> None:
    """
    Validate all configuration after environment loading.
    Raises RuntimeError if any critical config is missing/invalid.

    This runs AFTER load_dotenv() to ensure all environment variables
    are available for validation.
    """
    errors = []

    # Validate Pinecone Configuration
    try:
        from common.config import PineconeConfig
        PineconeConfig.validate()
        logger.info("[INIT]   ✓ PineconeConfig validated")
    except ValueError as e:
        errors.append(f"PineconeConfig: {e}")
    except ImportError:
        logger.warning("[INIT]   ⚠ PineconeConfig not available (optional)")

    # Validate Agentic Configuration
    try:
        from common.config import AgenticConfig
        AgenticConfig.validate()
        logger.info("[INIT]   ✓ AgenticConfig validated")
    except ValueError as e:
        errors.append(f"AgenticConfig: {e}")
    except ImportError:
        logger.warning("[INIT]   ⚠ AgenticConfig not available (optional)")

    # Validate MongoDB Configuration (NEW - Phase 1 refactoring)
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb_db = os.getenv("MONGODB_DATABASE", "product_recommender")
        if mongodb_uri:
            logger.info(f"[INIT]   ✓ MongoDB configured (database: {mongodb_db})")
        else:
            logger.warning("[INIT]   ⚠ MONGODB_URI not set - using Azure Blob fallback for data queries")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ MongoDB validation warning: {e}")

    # Validate Azure Configuration (optional)
    try:
        azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if azure_conn_str:
            logger.info("[INIT]   ✓ Azure Storage configured")
        else:
            logger.warning("[INIT]   ⚠ Azure Storage not configured (optional)")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Azure validation warning: {e}")

    # Validate Google API Keys
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        logger.info("[INIT]   ✓ GOOGLE_API_KEY configured")
    else:
        logger.warning("[INIT]   ⚠ GOOGLE_API_KEY not set - some features may be unavailable")

    # Validate OpenAI API Key (fallback)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("[INIT]   ✓ OPENAI_API_KEY configured (fallback available)")
    else:
        logger.warning("[INIT]   ⚠ OPENAI_API_KEY not set - no LLM fallback available")
        
    # Phase 5: Enhanced LLM Configuration Warnings
    _log_llm_configuration_warnings()

    # Validate Google Custom Search (optional)
    google_cx = os.getenv("GOOGLE_CX")
    if google_cx:
        logger.info("[INIT]   ✓ GOOGLE_CX configured (image search available)")
    else:
        logger.warning("[INIT]   ⚠ GOOGLE_CX not set - image search may be limited")

    # If there are critical errors, raise exception
    if errors:
        error_msg = "Critical configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def _log_llm_configuration_warnings():
    """
    Phase 5: Log warnings about LLM configuration to help users avoid rate limit issues.
    """
    google_keys = []
    # Check for multiple Google API keys
    for i in range(1, 10):
        key_name = f"GOOGLE_API_KEY{i}" if i > 1 else "GOOGLE_API_KEY"
        key = os.getenv(key_name)
        if key:
            google_keys.append(key_name)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Log summary
    logger.info("="*70)
    logger.info("[INIT] 📊 LLM Configuration Summary:")
    logger.info(f"[INIT]   • Google API keys available: {len(google_keys)}")
    logger.info(f"[INIT]   • OpenAI fallback configured: {'Yes ✓' if openai_key else 'No ✗'}")
    
    # Warnings based on configuration
    warning_count = 0
    
    # Warning 1: No OpenAI fallback
    if not openai_key:
        warning_count += 1
        logger.warning(f"[INIT]   ⚠️ WARNING {warning_count}: No OpenAI fallback configured!")
        logger.warning(f"[INIT]      • Gemini free tier has strict rate limits (20 req/min)")
        logger.warning(f"[INIT]      • Add OPENAI_API_KEY to .env for automatic fallback")
    
    # Warning 2: Only one Google key
    if len(google_keys) == 1:
        warning_count += 1
        logger.warning(f"[INIT]   ⚠️ WARNING {warning_count}: Only 1 Google API key configured")
        logger.warning(f"[INIT]      • Consider adding GOOGLE_API_KEY2, GOOGLE_API_KEY3 for rotation")
        logger.warning(f"[INIT]      • Multiple keys help avoid rate limit errors")
    
    # Warning 3: No API keys at all
    if len(google_keys) == 0 and not openai_key:
        warning_count += 1
        logger.warning(f"[INIT]   ❌ CRITICAL WARNING {warning_count}: No LLM API keys configured!")
        logger.warning(f"[INIT]      • The application will not function without LLM access")
        logger.warning(f"[INIT]      • Add GOOGLE_API_KEY or OPENAI_API_KEY to .env")
    
    # Recommendations
    if warning_count > 0:
        logger.info("[INIT]   📋 Recommended configuration for production:")
        logger.info("[INIT]      • GOOGLE_API_KEY + GOOGLE_API_KEY2 + GOOGLE_API_KEY3")
        logger.info("[INIT]      • OPENAI_API_KEY (for reliable fallback)")
    else:
        logger.info("[INIT]   ✓ LLM configuration looks good!")
    
    logger.info("="*70)


def initialize_singletons() -> None:
    """
    Initialize all singleton instances at startup.
    Ensures consistent state across application.

    This should be called AFTER validate_all_configs() to ensure
    all required environment variables are present.
    """
    # Initialize MongoDB Manager (NEW - Phase 1 refactoring)
    try:
        from common.core.mongodb_manager import mongodb_manager, is_mongodb_available
        if is_mongodb_available():
            logger.info("[INIT]   ✓ MongoDB Manager connected")
        else:
            logger.warning("[INIT]   ⚠ MongoDB not available - using Azure Blob fallback")
    except ImportError:
        logger.warning("[INIT]   ⚠ MongoDB Manager not available (install pymongo)")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ MongoDB initialization failed: {e}")

    # Initialize API Key Manager
    try:
        from common.config.api_key_manager import api_key_manager
        google_count = api_key_manager.get_google_key_count()
        logger.info(f"[INIT]   ✓ API Key Manager initialized: {google_count} Google key(s) available")

        if google_count == 0:
            logger.warning("[INIT]   ⚠ No Google API keys configured")
        elif google_count > 1:
            logger.info(f"[INIT]   ✓ API key rotation enabled with {google_count} keys")

    except ImportError as e:
        logger.warning(f"[INIT]   ⚠ API Key Manager not available: {e}")
    except Exception as e:
        logger.error(f"[INIT]   ✗ API Key Manager initialization failed: {e}")
        raise

    # Initialize Azure Blob Manager (if configured) - NON-BLOCKING
    # We DON'T connect here to avoid blocking if Azure is unreachable
    try:
        from common.config.azure_blob_config import azure_blob_manager
        if azure_blob_manager.is_available:
            # OPTIMIZATION: Don't trigger connection here - lazy initialization
            # Connection will happen on first actual use
            logger.info("[INIT]   ✓ Azure Blob Manager configured (lazy connection)")
        else:
            logger.warning("[INIT]   ⚠ Azure Blob Manager not configured (optional)")
    except ImportError:
        logger.warning("[INIT]   ⚠ Azure Blob Manager not available (optional)")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Azure Blob Manager initialization failed: {e}")
        # Don't raise - Azure is optional

    # Initialize Protected Caches (REMOVED - 'search' module deprecated)
    # Caches are now managed within their respective modules (e.g. product_search)
    pass


    # Initialize Database Indexes (NEW - Phase 1 refactoring)
    try:
        from common.core.mongodb_manager import is_mongodb_available
        if is_mongodb_available():
            from common.core.db_indexes import ensure_indexes
            ensure_indexes()
            logger.info("[INIT]   ✓ MongoDB indexes created/verified")
    except ImportError:
        logger.debug("[INIT]   ⚠ Database indexes module not available")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Index creation failed: {e}")

    # Initialize Service Layer (NEW - Phase 2 refactoring)
    try:
        from common.services.schema_service import schema_service
        from common.services.vendor_service import vendor_service
        from common.services.project_service import project_service
        from common.services.document_service import document_service
        from common.services.image_service import image_service
        logger.info("[INIT]   ✓ Service layer initialized (schema, vendor, project, document, image)")
    except ImportError as e:
        logger.warning(f"[INIT]   ⚠ Some services not available: {e}")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Service initialization warning: {e}")

    # Log initialization summary
    logger.info("[INIT]   ✓ Singleton initialization complete")


def check_initialization_status() -> dict:
    """
    Check the status of all initialized components.
    Useful for health checks and debugging.

    Returns:
        dict: Status of each component
    """
    status = {
        "environment_loaded": True,  # If we got here, env is loaded
        "google_api_keys": 0,
        "openai_configured": False,
        "azure_configured": False,
        "pinecone_configured": False,
        "mongodb_configured": False,
        "mongodb_connected": False,
        "services_available": False,
    }

    # Check API keys
    try:
        from common.config.api_key_manager import api_key_manager
        status["google_api_keys"] = api_key_manager.get_google_key_count()
    except:
        pass

    status["openai_configured"] = bool(os.getenv("OPENAI_API_KEY"))
    status["azure_configured"] = bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    status["pinecone_configured"] = bool(os.getenv("PINECONE_API_KEY"))

    # Check MongoDB (NEW - Phase 1 refactoring)
    status["mongodb_configured"] = bool(os.getenv("MONGODB_URI"))
    try:
        from common.core.mongodb_manager import is_mongodb_available
        status["mongodb_connected"] = is_mongodb_available()
    except:
        pass

    # Check services availability (NEW - Phase 2 refactoring)
    try:
        from common.services.schema_service import schema_service
        from common.services.vendor_service import vendor_service
        from common.services.project_service import project_service
        status["services_available"] = all([
            schema_service is not None,
            vendor_service is not None,
            project_service is not None
        ])
    except:
        pass

    return status


def get_database_health() -> dict:
    """
    Get detailed database health status.

    Returns:
        dict: Health status for MongoDB and Azure Blob
    """
    health = {
        "timestamp": None,
        "mongodb": {"status": "unknown", "connected": False},
        "azure_blob": {"status": "unknown", "connected": False},
        "services": {}
    }

    try:
        from datetime import datetime
        health["timestamp"] = datetime.utcnow().isoformat()
    except:
        pass

    # Check MongoDB
    try:
        from common.core.mongodb_manager import mongodb_manager
        if mongodb_manager:
            health["mongodb"] = mongodb_manager.health_check()
    except ImportError:
        health["mongodb"]["status"] = "not_installed"
    except Exception as e:
        health["mongodb"]["error"] = str(e)

    # Check Azure Blob
    try:
        from common.core.azure_blob_file_manager import azure_blob_file_manager
        if azure_blob_file_manager:
            health["azure_blob"] = azure_blob_file_manager.health_check()
    except ImportError:
        health["azure_blob"]["status"] = "not_installed"
    except Exception as e:
        health["azure_blob"]["error"] = str(e)

    # Check services
    try:
        from common.services.schema_service import schema_service
        if schema_service:
            health["services"]["schema"] = schema_service.health_check()
    except:
        pass

    try:
        from common.services.vendor_service import vendor_service
        if vendor_service:
            health["services"]["vendor"] = vendor_service.health_check()
    except:
        pass

    try:
        from common.services.project_service import project_service
        if project_service:
            health["services"]["project"] = project_service.health_check()
    except:
        pass

    return health


if __name__ == "__main__":
    # Test initialization
    try:
        initialize_application()
        print("\n✓ Initialization successful")
        print("\nStatus:")
        status = check_initialization_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
