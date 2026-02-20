"""
SAS URL Utilities
==================
Utility functions for generating Shared Access Signature (SAS) URLs.

SAS URLs provide time-limited access to private Azure Blob resources.
Instead of making blobs public, generate a temporary SAS URL that expires.

Usage:
    from common.core.sas_utils import generate_sas_url, add_sas_to_document

    # Generate SAS URL from blob URL
    sas_url = generate_sas_url(
        blob_url="https://account.blob.core.windows.net/container/file.pdf",
        expiry_hours=1
    )
    # Returns: "https://account.blob.core.windows.net/container/file.pdf?sv=...&sig=..."

    # Add SAS URL to MongoDB document
    doc = add_sas_to_document(
        document={"storage": {"blobUrl": "https://..."}},
        blob_url_field="storage.blobUrl"
    )
    # Returns document with added 'sas_url' field
"""
from typing import Optional, Dict, List, Any, Union
from common.core.azure_blob_file_manager import azure_blob_file_manager


def generate_sas_url(
    blob_url: str,
    expiry_hours: int = 1,
    permissions: str = 'r'
) -> Optional[str]:
    """
    Generate time-limited SAS URL for secure file access.

    Args:
        blob_url: Full blob URL (https://account.blob.core.windows.net/container/path/file.pdf)
        expiry_hours: Hours until SAS token expires (default: 1 hour)
        permissions: SAS permissions ('r'=read, 'w'=write, 'd'=delete, 'l'=list)

    Returns:
        URL with SAS token appended for secure access, or None on error

    Example:
        >>> url = generate_sas_url("https://account.blob.core.windows.net/docs/file.pdf")
        >>> print(url)
        https://account.blob.core.windows.net/docs/file.pdf?sv=2021-08-06&st=...&se=...&sr=b&sp=r&sig=...
    """
    return azure_blob_file_manager.generate_sas_url(blob_url, expiry_hours, permissions)


def generate_sas_url_for_path(
    blob_path: str,
    container_name: str = None,
    expiry_hours: int = 1
) -> Optional[str]:
    """
    Generate SAS URL from blob path (not full URL).

    Args:
        blob_path: Path within container (e.g., "Product-Recommender/files/doc.pdf")
        container_name: Container name (defaults to main container)
        expiry_hours: Hours until expiry

    Returns:
        Full SAS URL or None on error
    """
    return azure_blob_file_manager.generate_sas_url_for_path(
        blob_path, container_name, expiry_hours
    )


def get_nested_value(obj: Dict, field_path: str) -> Any:
    """
    Get nested value from dict using dot notation.

    Args:
        obj: Dictionary to search
        field_path: Dot-separated path (e.g., "storage.blobUrl")

    Returns:
        Value at path or None if not found

    Example:
        >>> doc = {"storage": {"blobUrl": "https://..."}}
        >>> get_nested_value(doc, "storage.blobUrl")
        "https://..."
    """
    parts = field_path.split('.')
    value = obj

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None

    return value


def set_nested_value(obj: Dict, field_path: str, value: Any) -> Dict:
    """
    Set nested value in dict using dot notation.

    Args:
        obj: Dictionary to modify
        field_path: Dot-separated path
        value: Value to set

    Returns:
        Modified dictionary
    """
    parts = field_path.split('.')
    current = obj

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
    return obj


def add_sas_to_document(
    document: Dict,
    blob_url_field: str = "storage.blobUrl",
    expiry_hours: int = 1,
    sas_url_field: str = "sas_url"
) -> Dict:
    """
    Add SAS URL to a MongoDB document for file access.

    Args:
        document: MongoDB document containing blob URL
        blob_url_field: Nested field path to blob URL (e.g., "storage.blobUrl")
        expiry_hours: Hours until SAS token expires
        sas_url_field: Field name for SAS URL in document

    Returns:
        Document with added SAS URL field

    Example:
        >>> doc = {"filename": "test.pdf", "storage": {"blobUrl": "https://..."}}
        >>> doc = add_sas_to_document(doc, "storage.blobUrl")
        >>> print(doc.get("sas_url"))
        "https://...?sv=...&sig=..."
    """
    # Get blob URL from nested path
    blob_url = get_nested_value(document, blob_url_field)

    if blob_url and isinstance(blob_url, str) and 'blob.core.windows.net' in blob_url:
        sas_url = generate_sas_url(blob_url, expiry_hours=expiry_hours)
        if sas_url:
            document[sas_url_field] = sas_url

    return document


def add_sas_to_documents(
    documents: List[Dict],
    blob_url_field: str = "storage.blobUrl",
    expiry_hours: int = 1,
    sas_url_field: str = "sas_url"
) -> List[Dict]:
    """
    Add SAS URLs to multiple documents.

    Args:
        documents: List of MongoDB documents
        blob_url_field: Nested field path to blob URL
        expiry_hours: Hours until SAS token expires
        sas_url_field: Field name for SAS URL

    Returns:
        Documents with added SAS URL fields

    Example:
        >>> docs = [{"storage": {"blobUrl": "https://..."}}, ...]
        >>> docs = add_sas_to_documents(docs)
        >>> print(docs[0].get("sas_url"))
        "https://...?sv=...&sig=..."
    """
    return [
        add_sas_to_document(doc, blob_url_field, expiry_hours, sas_url_field)
        for doc in documents
    ]


def add_sas_to_image_document(
    document: Dict,
    expiry_hours: int = 24
) -> Dict:
    """
    Add SAS URL to image document (checks image.blobUrl field).

    Images typically need longer expiry for caching/display.

    Args:
        document: Image document from MongoDB
        expiry_hours: Hours until SAS token expires (default: 24 for images)

    Returns:
        Document with added 'sas_url' field
    """
    return add_sas_to_document(document, "image.blobUrl", expiry_hours)


def add_sas_to_project_document(
    document: Dict,
    expiry_hours: int = 1
) -> Dict:
    """
    Add SAS URL to project document (checks blob_info.blob_url field).

    Args:
        document: Project document from MongoDB
        expiry_hours: Hours until SAS token expires

    Returns:
        Document with added 'sas_url' field
    """
    return add_sas_to_document(document, "blob_info.blob_url", expiry_hours)


def batch_generate_sas_urls(
    blob_urls: List[str],
    expiry_hours: int = 1
) -> Dict[str, Optional[str]]:
    """
    Generate SAS URLs for multiple blobs.

    Args:
        blob_urls: List of blob URLs
        expiry_hours: Hours until expiry

    Returns:
        Dict mapping original URLs to SAS URLs

    Example:
        >>> urls = ["https://.../file1.pdf", "https://.../file2.pdf"]
        >>> sas_urls = batch_generate_sas_urls(urls)
        >>> print(sas_urls)
        {"https://.../file1.pdf": "https://.../file1.pdf?sv=...", ...}
    """
    return {
        url: generate_sas_url(url, expiry_hours)
        for url in blob_urls
    }


def is_sas_url(url: str) -> bool:
    """
    Check if URL already contains SAS token.

    Args:
        url: URL to check

    Returns:
        True if URL contains SAS parameters
    """
    if not url:
        return False

    sas_indicators = ['sv=', 'sig=', 'se=', 'sp=']
    return any(indicator in url for indicator in sas_indicators)


def strip_sas_from_url(url: str) -> str:
    """
    Remove SAS token from URL.

    Args:
        url: URL with potential SAS token

    Returns:
        URL without SAS parameters
    """
    if not url or '?' not in url:
        return url

    return url.split('?')[0]


def refresh_sas_url(
    sas_url: str,
    expiry_hours: int = 1
) -> Optional[str]:
    """
    Generate new SAS URL from existing (possibly expired) SAS URL.

    Args:
        sas_url: Existing SAS URL (may be expired)
        expiry_hours: Hours for new token

    Returns:
        Fresh SAS URL
    """
    base_url = strip_sas_from_url(sas_url)
    return generate_sas_url(base_url, expiry_hours)


# ============================================================
# Convenience Constants
# ============================================================

# Common expiry durations
EXPIRY_SHORT = 1       # 1 hour - for immediate use
EXPIRY_MEDIUM = 4      # 4 hours - for session use
EXPIRY_LONG = 24       # 24 hours - for downloads
EXPIRY_EXTENDED = 168  # 7 days - for sharing


# ============================================================
# Type Aliases
# ============================================================

SASUrl = str
BlobUrl = str
