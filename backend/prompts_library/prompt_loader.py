# prompts_library/prompt_loader.py
# =============================================================================
# PROMPT LOADER UTILITY
# =============================================================================
#
# This module provides functions to load prompts from the prompts_library folder.
# All prompts are stored in individual .txt files and loaded at runtime.
#
# File Format:
#   PROMPT NAME: <name>
#   SOURCE FILE: <source file path>
#   PURPOSE: <description>
#   
#   === PROMPT CONTENT ===
#   
#   <actual prompt content>
#   
#   === PARAMETERS ===
#   - param1: description
#   - param2: description
#
# Usage:
#   from prompts_library.prompt_loader import load_prompt
#   
#   PLANNER_PROMPT = load_prompt("deep_agent_planner_prompt")
#
# =============================================================================

import os
import re
import logging
from functools import lru_cache
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Directory containing prompt files
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=100)
def load_prompt(prompt_name: str, section: Optional[str] = None) -> str:
    """
    Load prompt content from a prompt library file.
    
    Supports two modes:
    1. Single-prompt files: Returns content between '=== PROMPT CONTENT ===' and 
       '=== PARAMETERS ===' markers.
    2. Multi-prompt files: When section is specified, returns content for that 
       specific section (e.g., section="PLANNER" loads === PROMPT: PLANNER ===).
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        section: Optional section name for multi-prompt files (e.g., "PLANNER")
        
    Returns:
        The prompt content string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If the prompt content cannot be extracted
    """
    # Construct file path
    filename = f"{prompt_name}.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Handle multi-prompt files with section parameter
        if section:
            prompt_content = _extract_section_content(content, section)
            if not prompt_content:
                raise ValueError(f"Section '{section}' not found in {filename}")
            logger.debug(f"Loaded prompt: {prompt_name}[{section}] ({len(prompt_content)} chars)")
            return prompt_content
        
        # Handle single-prompt files (backward compatible)
        prompt_content = _extract_prompt_content(content)
        
        if not prompt_content:
            logger.warning(f"Could not extract prompt content from {filename}, using full content")
            return content.strip()
        
        logger.debug(f"Loaded prompt: {prompt_name} ({len(prompt_content)} chars)")
        return prompt_content
        
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_name}: {e}")
        raise


def _extract_prompt_content(file_content: str) -> Optional[str]:
    """
    Extract the actual prompt content from a prompt file.
    
    Looks for content between '=== PROMPT CONTENT ===' and '=== PARAMETERS ===' markers.
    If PARAMETERS marker is not found, extracts everything after PROMPT CONTENT marker.
    
    Args:
        file_content: The full content of the prompt file
        
    Returns:
        The extracted prompt content, or None if markers not found
    """
    # Pattern to match content between markers
    # Handles both Windows (\r\n) and Unix (\n) line endings
    
    start_marker = "=== PROMPT CONTENT ==="
    end_marker = "=== PARAMETERS ==="
    
    start_idx = file_content.find(start_marker)
    
    if start_idx == -1:
        return None
    
    # Move past the start marker
    start_idx += len(start_marker)
    
    # Find end marker (optional)
    end_idx = file_content.find(end_marker, start_idx)
    
    if end_idx == -1:
        # No end marker, take everything after start marker
        prompt_content = file_content[start_idx:]
    else:
        # Extract content between markers
        prompt_content = file_content[start_idx:end_idx]
    
    # Clean up the content
    prompt_content = prompt_content.strip()
    
    return prompt_content


def _extract_section_content(file_content: str, section: str) -> Optional[str]:
    """
    Extract content for a specific section from a multi-prompt file.
    
    Multi-prompt files use section markers like:
        === PROMPT: SECTION_NAME ===
        PURPOSE: Description of this prompt
        
        <actual prompt content>
        
        === PROMPT: NEXT_SECTION ===
        ...
    
    Args:
        file_content: The full content of the multi-prompt file
        section: The section name to extract (e.g., "PLANNER", "WORKER")
        
    Returns:
        The extracted section content, or None if section not found
    """
    # Pattern to match section start
    section_marker = f"=== PROMPT: {section} ==="
    
    start_idx = file_content.find(section_marker)
    if start_idx == -1:
        # Try case-insensitive search using FAST string method (not character-by-character)
        # This is O(n) instead of O(n²) and fixes the critical hang issue
        section_marker_upper = section_marker.upper()
        content_upper = file_content.upper()
        start_idx = content_upper.find(section_marker_upper)
        
        if start_idx == -1:
            return None
    
    # Move past the section marker
    start_idx += len(section_marker)
    
    # Find the next section marker or end of file
    next_section_pattern = "=== PROMPT:"
    end_idx = file_content.find(next_section_pattern, start_idx)
    
    if end_idx == -1:
        # No next section, take everything to end
        section_content = file_content[start_idx:]
    else:
        section_content = file_content[start_idx:end_idx]
    
    # Clean up the content
    section_content = section_content.strip()
    
    # Remove PURPOSE line if present at the start
    lines = section_content.split('\n')
    if lines and lines[0].strip().startswith("PURPOSE:"):
        lines = lines[1:]
        section_content = '\n'.join(lines).strip()
    
    return section_content


def load_prompt_sections(prompt_name: str, default_section: Optional[str] = None) -> Dict[str, str]:
    """
    Load all sections from a multi-prompt file.

    Supports two formats:
    1. New format: [SECTION_NAME] (preferred, cleaner)
    2. Legacy format: === PROMPT: SECTION_NAME === (backwards compatible)

    Content before any section marker is captured as the default_section if provided.
    This allows prompts to start with personality text (e.g., "You are EnGenie...")
    without requiring a section header on line 1.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        default_section: Optional name for content before first section marker

    Returns:
        Dict mapping section names to their content

    Example:
        prompts = load_prompt_sections("intent_classification_prompts", default_section="CLASSIFICATION")
        # Returns: {"CLASSIFICATION": "You are EnGenie...", "QUICK_CLASSIFICATION": "..."}
    """
    filename = f"{prompt_name}.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = {}

        # Try new format first: [SECTION_NAME]
        new_pattern = r"^\[([A-Z_]+)\]$"
        new_matches = list(re.finditer(new_pattern, content, re.MULTILINE))

        if new_matches:
            # Capture content before first section marker as default_section
            if default_section and new_matches[0].start() > 0:
                pre_content = content[:new_matches[0].start()].strip()
                if pre_content:
                    sections[default_section] = pre_content

            # Use new [SECTION] format
            for i, match in enumerate(new_matches):
                section_name = match.group(1)
                start_pos = match.end()

                if i + 1 < len(new_matches):
                    end_pos = new_matches[i + 1].start()
                else:
                    end_pos = len(content)

                section_content = content[start_pos:end_pos].strip()
                if section_content:
                    sections[section_name] = section_content
        else:
            # Fallback to legacy format: === PROMPT: SECTION_NAME ===
            legacy_pattern = r"=== PROMPT: ([A-Z_]+) ==="
            legacy_matches = list(re.finditer(legacy_pattern, content))

            # Capture content before first section marker as default_section
            if default_section:
                if legacy_matches and legacy_matches[0].start() > 0:
                    pre_content = content[:legacy_matches[0].start()].strip()
                    if pre_content:
                        sections[default_section] = pre_content
                elif not legacy_matches:
                    # No section markers at all - entire content is the default section
                    if content.strip():
                        sections[default_section] = content.strip()

            for i, match in enumerate(legacy_matches):
                section_name = match.group(1)
                start_pos = match.end()

                if i + 1 < len(legacy_matches):
                    end_pos = legacy_matches[i + 1].start()
                else:
                    end_pos = len(content)

                section_content = content[start_pos:end_pos].strip()

                # Remove PURPOSE line if present at the start (legacy format)
                lines = section_content.split('\n')
                if lines and lines[0].strip().startswith("PURPOSE:"):
                    lines = lines[1:]
                    section_content = '\n'.join(lines).strip()

                if section_content:
                    sections[section_name] = section_content

        logger.debug(f"Loaded {len(sections)} sections from {prompt_name}")
        return sections

    except Exception as e:
        logger.error(f"Error loading prompt sections from {prompt_name}: {e}")
        raise


def get_prompt_metadata(prompt_name: str) -> Dict[str, str]:
    """
    Get metadata from a prompt file (name, source file, purpose).
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Dict with 'name', 'source_file', 'purpose' keys
    """
    filename = f"{prompt_name}.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    
    metadata = {
        "name": "",
        "source_file": "",
        "purpose": ""
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("PROMPT NAME:"):
                    metadata["name"] = line.replace("PROMPT NAME:", "").strip()
                elif line.startswith("SOURCE FILE:"):
                    metadata["source_file"] = line.replace("SOURCE FILE:", "").strip()
                elif line.startswith("PURPOSE:"):
                    metadata["purpose"] = line.replace("PURPOSE:", "").strip()
                elif line.startswith("==="):
                    break  # Stop at first marker
                    
    except Exception as e:
        logger.error(f"Error reading metadata from {prompt_name}: {e}")
    
    return metadata


def list_available_prompts() -> list:
    """
    List all available prompt files in the library.
    
    Returns:
        List of prompt names (without .txt extension)
    """
    prompts = []
    
    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith('.txt') and not filename.startswith('_'):
            prompt_name = filename[:-4]  # Remove .txt extension
            prompts.append(prompt_name)
    
    return sorted(prompts)


def reload_prompt(prompt_name: str) -> str:
    """
    Force reload a prompt, bypassing the cache.
    
    Useful for development when prompt files are being edited.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        The prompt content string
    """
    # Clear this specific entry from cache
    load_prompt.cache_clear()
    
    return load_prompt(prompt_name)


def clear_prompt_cache():
    """Clear all cached prompts."""
    load_prompt.cache_clear()
    logger.info("Prompt cache cleared")


# =============================================================================
# CONVENIENCE FUNCTIONS - Pre-loaded prompt groups
# =============================================================================

def get_deep_agent_prompts() -> Dict[str, str]:
    """Load all Deep Agent prompts."""
    return {
        "planner": load_prompt("deep_agent_planner_prompt"),
        "worker": load_prompt("deep_agent_worker_prompt"),
        "synthesizer": load_prompt("deep_agent_synthesizer_prompt"),
        "merger": load_prompt("deep_agent_merger_prompt"),
        "iterative_worker": load_prompt("deep_agent_iterative_worker_prompt"),
        "batch_planner": load_prompt("deep_agent_batch_planner_prompt"),
        "batch_worker": load_prompt("deep_agent_batch_worker_prompt"),
        "batch_synthesizer": load_prompt("deep_agent_batch_synthesizer_prompt"),
    }


def get_shared_agent_prompts() -> Dict[str, str]:
    """Load all shared agent prompts."""
    index_rag_prompts = load_prompt_sections("index_rag_prompts", default_section="INTENT_CLASSIFICATION")
    return {
        "chat_agent": index_rag_prompts.get("CHAT_AGENT", ""),
        "validator": load_prompt("validator_prompt"),
        "web_verifier": load_prompt("web_verifier_prompt"),
    }


def get_rag_prompts() -> Dict[str, str]:
    """Load all RAG-related prompts."""
    return {
        "strategy_rag": load_prompt("strategy_rag_prompt"),
        "standards_rag": load_prompt("standards_rag_prompt"),
        "inventory_rag": load_prompt("inventory_rag_prompt"),
        "strategy_chat": load_prompt("strategy_chat_prompt"),
        "standards_chat": load_prompt("standards_chat_prompt"),
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Log available prompts on import (debug level)
if __name__ != "__main__":
    try:
        available = list_available_prompts()
        logger.debug(f"Prompts library initialized with {len(available)} prompts")
    except Exception:
        pass


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PROMPT LOADER TEST")
    print("=" * 60)
    
    # List all prompts
    prompts = list_available_prompts()
    print(f"\nFound {len(prompts)} prompts in library:")
    for p in prompts:
        print(f"  - {p}")
    
    # Test loading a specific prompt
    if len(sys.argv) > 1:
        prompt_name = sys.argv[1]
    else:
        prompt_name = "index_rag_prompts"
    
    print(f"\n{'=' * 60}")
    print(f"Testing load_prompt('{prompt_name}')")
    print("=" * 60)
    
    try:
        content = load_prompt(prompt_name)
        print(f"\nLoaded successfully! ({len(content)} characters)")
        print("\n--- Content Preview (first 500 chars) ---")
        print(content[:500])
        print("...")
        
        metadata = get_prompt_metadata(prompt_name)
        print(f"\n--- Metadata ---")
        print(f"  Name: {metadata['name']}")
        print(f"  Source: {metadata['source_file']}")
        print(f"  Purpose: {metadata['purpose']}")
        
    except Exception as e:
        print(f"Error: {e}")
