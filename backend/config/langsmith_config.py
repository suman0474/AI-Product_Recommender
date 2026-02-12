"""
LangSmith Tracing Configuration

Enables LangSmith tracing for debugging and monitoring LangChain/LangGraph workflows.
Set LANGSMITH_API_KEY environment variable to enable tracing.
"""
import os
import logging

logger = logging.getLogger(__name__)


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing if API key is available.
    
    Sets the required environment variables for LangChain to automatically
    send traces to LangSmith.
    
    Returns:
        True if tracing was enabled, False otherwise
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    
    if not api_key:
        logger.info("[LANGSMITH] No API key found (LANGSMITH_API_KEY not set), tracing disabled")
        return False
    
    # Enable LangChain v2 tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "aipr-backend")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    project_name = os.environ["LANGCHAIN_PROJECT"]
    logger.info(f"[LANGSMITH] ✅ Tracing enabled for project: {project_name}")
    
    return True


def get_langsmith_status() -> dict:
    """
    Get current LangSmith configuration status.
    
    Returns:
        Dictionary with tracing status and configuration
    """
    return {
        "enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true",
        "project": os.getenv("LANGCHAIN_PROJECT", "not set"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "not set"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY"))
    }


def disable_langsmith():
    """Disable LangSmith tracing."""
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    logger.info("[LANGSMITH] Tracing disabled")
