"""Prompt Loader for Deep Agent PPI — Delegation Facade
=====================================================
The canonical prompt loading implementation lives in `prompts_library`.
This module delegates to it and preserves the original API for backward compatibility.
"""

from common.prompts import INDEXING_AGENT_PROMPTS as _PROMPTS_DICT

def load_prompt(prompt_key: str, default: str = "") -> str:
    """
    Load a prompt section by key from indexing_agent_prompts.
    
    Now delegates to common.prompts.INDEXING_AGENT_PROMPTS.
    """
    # Try exact key, then UPPER, then lower to handle casing differences
    return (
        _PROMPTS_DICT.get(prompt_key)
        or _PROMPTS_DICT.get(prompt_key.upper())
        or _PROMPTS_DICT.get(prompt_key.lower())
        or default
    )


class PromptLoader:
    """
    Backward-compatible singleton facade.
    Delegates to `prompts_library.load_prompt_sections` internally.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_prompt(self, prompt_key: str, default: str = "") -> str:
        return load_prompt(prompt_key, default)

    def reload_prompts(self):
        pass  # prompts_library handles caching internally

    def list_available_prompts(self) -> list:
        return []


_prompt_loader_instance = None


def get_prompt_loader() -> PromptLoader:
    """Get the singleton PromptLoader instance."""
    global _prompt_loader_instance
    if _prompt_loader_instance is None:
        _prompt_loader_instance = PromptLoader()
    return _prompt_loader_instance
