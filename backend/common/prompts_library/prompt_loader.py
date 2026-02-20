import os
import glob
from typing import Dict, Any, List, Optional

PROMPTS_DIR = os.path.dirname(__file__)

def load_prompt_sections(file_key: str) -> Dict[str, str]:
    """Load all sections from a prompt file."""
    file_path = os.path.join(PROMPTS_DIR, f"{file_key}.txt")
    if not os.path.exists(file_path):
        # Fallback to search if extension included
        if file_key.endswith(".txt"):
             file_path = os.path.join(PROMPTS_DIR, file_key)
        
        if not os.path.exists(file_path):
            print(f"Warning: Prompt file not found: {file_path}")
            return {}
    
    sections = {}
    current_section = "DEFAULT"
    current_content = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Simple parser for [SECTION] format
        for line in f:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                # Save previous section
                if current_content:
                    sections[current_section] = "".join(current_content).strip()
                elif current_section == "DEFAULT" and not sections:
                    # Empty default section
                     pass

                current_section = stripped[1:-1]
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = "".join(current_content).strip()
            
    return sections

def load_prompt(file_key: str, section: str = "DEFAULT") -> str:
    """Load a specific prompt section."""
    sections = load_prompt_sections(file_key)
    return sections.get(section, "")

def get_prompt_metadata(file_key: str) -> Dict[str, Any]:
    return {"source": "file", "path": file_key}

def list_available_prompts() -> List[str]:
    files = glob.glob(os.path.join(PROMPTS_DIR, "*.txt"))
    return [os.path.basename(f).replace(".txt", "") for f in files]

def reload_prompt(file_key: str):
    pass # No caching implemented in this simple restore

def clear_prompt_cache():
    pass

def get_deep_agent_prompts():
    # Helper to load deep agent prompts
    # Assuming standard file names
    return {
        "search": load_prompt_sections("search_deep_agent_prompts"),
        "solution": load_prompt_sections("solution_deep_agent_prompts"),
        "standards": load_prompt_sections("standards_deep_agent_prompts"),
    }

def get_shared_agent_prompts():
    return load_prompt_sections("shared_agent_prompts")

def get_rag_prompts():
    return load_prompt_sections("rag_prompts")
