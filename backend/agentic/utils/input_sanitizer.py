"""
Input Sanitization for RAG Queries

Provides security-focused input sanitization to prevent injection attacks
and ensure query quality.
"""

import re
import html
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Maximum query length (characters)
MAX_QUERY_LENGTH = 4000

# Patterns to remove (potential injection attacks)
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script tags
    r'javascript:',                 # JavaScript URLs
    r'on\w+\s*=',                   # Event handlers
    r'<iframe[^>]*>.*?</iframe>',   # Iframes
    r'<object[^>]*>.*?</object>',   # Object tags
    r'<embed[^>]*>',                # Embed tags
    r'data:text/html',              # Data URLs
    r'\{\{.*?\}\}',                 # Template injection
    r'\$\{.*?\}',                   # Expression injection
]

# Control characters to remove (except newlines and tabs)
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_query(
    query: str, 
    max_length: int = MAX_QUERY_LENGTH,
    preserve_newlines: bool = True
) -> Tuple[str, bool, List[str]]:
    """
    Sanitize user query input.
    
    Performs the following sanitization:
    1. HTML escaping
    2. Removes dangerous patterns (scripts, iframes, etc.)
    3. Removes control characters
    4. Truncates to max length
    5. Normalizes whitespace
    
    Args:
        query: The raw user query
        max_length: Maximum allowed length
        preserve_newlines: If True, keeps newlines; if False, replaces with spaces
        
    Returns:
        Tuple of (sanitized_query, was_modified, modifications_list)
        
    Example:
        >>> query, modified, mods = sanitize_query("<script>alert('xss')</script>Hello")
        >>> print(query)
        Hello
        >>> print(modified)
        True
        >>> print(mods)
        ['removed_script_tag']
    """
    if not query:
        return "", False, []
    
    original = query
    modifications = []
    
    # 1. HTML escape (prevent XSS)
    escaped = html.escape(query)
    if escaped != query:
        modifications.append("html_escaped")
        query = escaped
    
    # 2. Remove dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
            query = re.sub(pattern, '', query, flags=re.IGNORECASE | re.DOTALL)
            modifications.append(f"removed_pattern_{pattern[:20]}")
    
    # 3. Remove control characters
    cleaned = CONTROL_CHAR_PATTERN.sub('', query)
    if cleaned != query:
        modifications.append("removed_control_chars")
        query = cleaned
    
    # 4. Handle newlines
    if not preserve_newlines:
        if '\n' in query or '\r' in query:
            query = query.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
            modifications.append("normalized_newlines")
    
    # 5. Truncate if too long
    if len(query) > max_length:
        query = query[:max_length].rsplit(' ', 1)[0] + "..."
        modifications.append(f"truncated_to_{max_length}")
    
    # 6. Normalize whitespace (collapse multiple spaces)
    normalized = ' '.join(query.split())
    if normalized != query.strip():
        modifications.append("normalized_whitespace")
        query = normalized
    else:
        query = query.strip()
    
    was_modified = query != original.strip()
    
    if modifications:
        logger.debug(f"[SANITIZE] Query modified: {modifications}")
    
    return query, was_modified, modifications


def validate_query(query: str, min_length: int = 3) -> Tuple[bool, str]:
    """
    Validate a query meets minimum requirements.
    
    Args:
        query: The query to validate
        min_length: Minimum required length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    if len(query.strip()) < min_length:
        return False, f"Query must be at least {min_length} characters"
    
    # Check if query is all punctuation/symbols
    if not re.search(r'[a-zA-Z0-9]', query):
        return False, "Query must contain alphanumeric characters"
    
    # Check for excessive repetition (spam detection)
    words = query.lower().split()
    if len(words) > 5:
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.3:  # Less than 30% unique words
            return False, "Query appears to contain excessive repetition"
    
    return True, ""


def sanitize_session_id(session_id: str) -> str:
    """
    Sanitize session ID to prevent injection.
    
    Args:
        session_id: Raw session ID
        
    Returns:
        Sanitized session ID (alphanumeric and hyphens only)
    """
    if not session_id:
        return "default"
    
    # Allow only alphanumeric, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', session_id)
    
    # Truncate to reasonable length
    return sanitized[:64] if sanitized else "default"


def extract_safe_keywords(query: str, max_keywords: int = 20) -> List[str]:
    """
    Extract safe keywords from a query for search purposes.
    
    Args:
        query: The user query
        max_keywords: Maximum keywords to extract
        
    Returns:
        List of sanitized keywords
    """
    # Sanitize first
    sanitized, _, _ = sanitize_query(query)
    
    # Remove common stop words
    stop_words = {
        'what', 'is', 'are', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 'of',
        'and', 'or', 'but', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them',
        'this', 'that', 'these', 'those', 'am', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'might', 'must',
        'tell', 'please', 'help', 'need', 'want', 'give', 'get', 'find', 'show',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z0-9]+\b', sanitized.lower())
    
    # Filter and deduplicate
    keywords = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen and len(word) > 1:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= max_keywords:
                break
    
    return keywords
