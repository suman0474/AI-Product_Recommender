import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def create_standalone_llm(temperature: float = 0.0, model_override: Optional[str] = None):
    """
    Creates a standalone LangChain LLM instance, decoupling Taxonomy RAG 
    from EnGenie's complex fallback and retry infrastructure for portability.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("[StandaloneLLM] GOOGLE_API_KEY environment variable not set. RAG extraction may fail.")

        llm_model = model_override or os.getenv("PRIMARY_LLM_MODEL", "gemini-2.5-flash")
        
        # Configure the primary Gemini model
        llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=False,
            max_retries=2
        )
        
        return llm
    except ImportError as e:
        logger.error(f"[StandaloneLLM] Missing required LangChain packages: {e}")
        raise
    except Exception as e:
        logger.error(f"[StandaloneLLM] Failed to initialize LLM: {e}")
        raise
