"""
PDF Utilities for Deep Agent PPI
=================================
Utilities for PDF downloading, processing, and text extraction.
"""

import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
from urllib.parse import urlparse
import socket

logger = logging.getLogger(__name__)

# DNS validation for faster failure detection
def validate_dns(url: str, timeout: int = 3) -> bool:
    """
    Validate that URL hostname resolves via DNS.
    
    Args:
        url: URL to validate
        timeout: DNS timeout in seconds
        
    Returns:
        True if DNS resolves, False otherwise
    """
    try:
        hostname = urlparse(url).hostname
        if not hostname:
            return False
        
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        return True
        
    except (socket.gaierror, socket.timeout, Exception) as e:
        logger.debug(f"DNS validation failed for {url}: {e}")
        return False

def download_pdf(
    url: str,
    save_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 60,
    max_size_mb: int = 50,
    validate_dns_first: bool = True
) -> Optional[Path]:
    """
    Download PDF from URL with validation and error handling.
    
    Args:
        url: PDF URL
        save_dir: Directory to save PDF
        filename: Optional custom filename
        timeout: Download timeout in seconds
        max_size_mb: Maximum file size in MB
        validate_dns_first: Whether to validate DNS before downloading
        
    Returns:
        Path to downloaded PDF or None if failed
    """
    try:
        # DNS validation (FIX #4)
        if validate_dns_first and not validate_dns(url, timeout=3):
            logger.warning(f"DNS validation failed for: {url}")
            return None
        
        # Generate filename if not provided
        if not filename:
            filename = Path(urlparse(url).path).name
            if not filename.endswith('.pdf'):
                filename += '.pdf'
        
        save_path = save_dir / filename
        
        # Download with streaming to check size
        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logger.warning(f"PDF too large: {int(content_length) / 1024 / 1024:.2f} MB")
            return None
        
        # Save PDF
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded PDF: {save_path}")
        return save_path
        
    except requests.Timeout:
        logger.warning(f"Download timeout for: {url}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Download failed for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return None

def extract_text_from_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract
        
    Returns:
        Extracted text
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        text_parts = []
        
        page_limit = min(len(doc), max_pages) if max_pages else len(doc)
        
        for page_num in range(page_limit):
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        return '\n\n'.join(text_parts)
        
    except ImportError:
        logger.warning("PyMuPDF not available, falling back to pypdf")
        return _extract_text_pypdf(pdf_path, max_pages)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def _extract_text_pypdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """Fallback text extraction using pypdf."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        text_parts = []
        
        page_limit = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
        
        for page_num in range(page_limit):
            page = reader.pages[page_num]
            text_parts.append(page.extract_text())
        
        return '\n\n'.join(text_parts)
        
    except Exception as e:
        logger.error(f"Fallback extraction failed for {pdf_path}: {e}")
        return ""

def rank_pdfs_by_relevance(
    pdfs: List[Dict[str, Any]],
    product_type: str,
    vendor_name: str
) -> List[Dict[str, Any]]:
    """
    Rank PDFs by relevance to product type and vendor.
    
    Args:
        pdfs: List of PDF metadata dicts with 'title' and 'url'
        product_type: Product type name
        vendor_name: Vendor name
        
    Returns:
        Sorted list of PDFs by relevance score
    """
    def calculate_relevance_score(pdf: Dict[str, Any]) -> float:
        """Calculate relevance score for a PDF."""
        score = 0.0
        title = pdf.get('title', '').lower()
        url = pdf.get('url', '').lower()
        
        # Product type keywords
        product_keywords = product_type.lower().split()
        for keyword in product_keywords:
            if keyword in title:
                score += 3.0
            if keyword in url:
                score += 1.0
        
        # Vendor name
        if vendor_name.lower() in title:
            score += 2.0
        if vendor_name.lower() in url:
            score += 1.0
        
        # Prefer specification sheets
        spec_keywords = ['specification', 'datasheet', 'technical', 'spec', 'manual']
        for keyword in spec_keywords:
            if keyword in title:
                score += 2.0
                break
        
        # Penalize marketing materials
        marketing_keywords = ['brochure', 'catalog', 'overview']
        for keyword in marketing_keywords:
            if keyword in title:
                score -= 1.0
                break
        
        return score
    
    # Calculate scores and sort
    for pdf in pdfs:
        pdf['relevance_score'] = calculate_relevance_score(pdf)
    
    return sorted(pdfs, key=lambda x: x.get('relevance_score', 0), reverse=True)

def get_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from PDF.
    
    Args:
        pdf_path: Path to PDF
        
    Returns:
        Dictionary with metadata
    """
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        metadata = {
            'page_count': len(doc),
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'file_size_mb': pdf_path.stat().st_size / 1024 / 1024
        }
        doc.close()
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
        return {'file_size_mb': pdf_path.stat().st_size / 1024 / 1024}
