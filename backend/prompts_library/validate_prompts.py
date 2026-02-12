#!/usr/bin/env python3
"""
Prompt Library Validation Script

This script scans the backend codebase for inline prompt definitions that should be
moved to the prompts_library folder. It helps maintain consistency and prevent
hardcoded prompts.

Usage:
    python prompts_library/validate_prompts.py

Exit codes:
    0 - No inline prompts found
    1 - Inline prompts detected (fails CI if used in pipeline)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns to detect inline prompts
INLINE_PROMPT_PATTERNS = [
    # ChatPromptTemplate.from_template with long multi-line strings
    (r'ChatPromptTemplate\.from_template\(\s*"""', "ChatPromptTemplate.from_template with triple-quoted string"),
    (r"ChatPromptTemplate\.from_template\(\s*'''", "ChatPromptTemplate.from_template with triple-quoted string"),
    # Long template strings (>200 chars)
    (r'ChatPromptTemplate\.from_template\(["\'][^"\']{200,}["\']', "ChatPromptTemplate with long inline string"),
]

# Exceptions (files allowed to have inline prompts)
ALLOWED_FILES = {
    "prompts.py",  # Legacy compatibility shim
    "validate_prompts.py",  # This script
}

# Directories to exclude from scanning
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".git",
    "venv",
    ".venv",
    "prompts_library",  # The library itself can have inline usage for testing
}


def find_inline_prompts(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a Python file for inline prompt definitions.
    
    Returns:
        List of (line_number, pattern_description, code_snippet) tuples
    """
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for pattern, description in INLINE_PROMPT_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                # Get snippet (up to 100 chars)
                snippet = lines[line_num - 1][:100]
                findings.append((line_num, description, snippet))
                
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        
    return findings


def scan_directory(root_dir: Path) -> dict:
    """
    Recursively scan directory for inline prompts.
    
    Returns:
        Dict mapping file paths to list of findings
    """
    results = {}
    
    for py_file in root_dir.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in EXCLUDE_DIRS):
            continue
            
        # Skip allowed files
        if py_file.name in ALLOWED_FILES:
            continue
            
        findings = find_inline_prompts(py_file)
        if findings:
            results[py_file] = findings
            
    return results


def print_report(results: dict):
    """Print a formatted report of findings."""
    if not results:
        print("✅ No inline prompts detected!")
        print("All prompts are properly loaded from prompts_library.")
        return
        
    print("⚠️  Inline Prompts Detected\n")
    print("=" * 80)
    print("The following files contain inline prompt definitions that should be")
    print("moved to the prompts_library folder:\n")
    
    total_findings = 0
    for file_path, findings in sorted(results.items()):
        print(f"\n📄 {file_path}")
        print("-" * 80)
        for line_num, description, snippet in findings:
            total_findings += 1
            print(f"  Line {line_num}: {description}")
            print(f"  ├─ {snippet}...")
            
    print("\n" + "=" * 80)
    print(f"Total files with inline prompts: {len(results)}")
    print(f"Total inline prompt occurrences: {total_findings}")
    print("\n💡 Recommendation:")
    print("  1. Create a new section in an existing prompt file, or")
    print("  2. Create a new .txt file in prompts_library/")
    print("  3. Use load_prompt() or load_prompt_sections() to access it")


def main():
    """Main validation entry point."""
    # Determine backend directory (assume script is in prompts_library/)
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    
    print(f"Scanning backend directory: {backend_dir}\n")
    
    results = scan_directory(backend_dir)
    print_report(results)
    
    # Exit with error code if findings detected (for CI/CD)
    if results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
