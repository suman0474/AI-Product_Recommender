import json
import os
import logging
from typing import Optional, Dict, Any

from common.agentic.deep_agent.memory import DeepAgentMemory

logger = logging.getLogger(__name__)


def load_taxonomy(filepath: Optional[str] = None) -> Dict[str, Any]:
    """Load taxonomy from JSON file."""
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        solution_dir = os.path.dirname(current_dir)
        filepath = os.path.join(solution_dir, "data", "taxonomy.json")

    try:
        if not os.path.exists(filepath):
            logger.warning(f"[Taxonomy] File not found: {filepath}")
            return {}

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            inst_count = len(data.get('instruments', []))
            acc_count = len(data.get('accessories', []))
            logger.info(f"[Taxonomy] Loaded {inst_count} instruments and {acc_count} accessories")
            return data
    except Exception as e:
        logger.error(f"[Taxonomy] Failed to load taxonomy from {filepath}: {e}")
        return {}


def inject_taxonomy_into_memory(memory: DeepAgentMemory, filepath: Optional[str] = None) -> None:
    """Load taxonomy and store it in agent memory."""
    taxonomy = load_taxonomy(filepath)

    if taxonomy and memory:
        memory.store_taxonomy(taxonomy)

    try:
        from .rag import get_taxonomy_rag
        rag = get_taxonomy_rag()
        rag.index_taxonomy(taxonomy)
    except Exception as e:
        logger.warning(f"[TaxonomyLoader] Failed to trigger RAG indexing: {e}")
