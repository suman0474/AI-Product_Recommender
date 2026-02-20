"""
Indexing Agent — PDF Tools
======================
Reusable PDF download, extraction, and ranking functions.

Delegates to the canonical utilities in ``utils.pdf_utils`` and adds
parallel-download orchestration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.pdf_utils import (
    download_pdf as _download_pdf,
    extract_text_from_pdf as _extract_text,
    rank_pdfs_by_relevance,
    validate_dns,
    get_pdf_metadata,
)
from ..config import MAX_DOWNLOAD_WORKERS, PDF_DOWNLOAD_TIMEOUT

logger = logging.getLogger(__name__)

# Re-export the canonical functions unchanged
download_pdf = _download_pdf
extract_text_from_pdf = _extract_text


def download_pdfs_parallel(
    pdfs: List[Dict[str, Any]],
    download_dir: Path,
    max_workers: int = MAX_DOWNLOAD_WORKERS,
    timeout: int = PDF_DOWNLOAD_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
    Download a list of PDFs in parallel using a thread pool.

    Each PDF dict is expected to have a ``url`` key.  On success, the dict is
    enriched with ``download_path`` and ``download_status = 'success'``.

    Args:
        pdfs: List of PDF metadata dicts.
        download_dir: Directory to save downloaded files.
        max_workers: Thread-pool size.
        timeout: Per-download timeout in seconds.

    Returns:
        List of successfully downloaded PDF dicts.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(
                download_pdf,
                pdf["url"],
                download_dir,
                timeout=timeout,
                validate_dns_first=True,
            ): pdf
            for pdf in pdfs
        }

        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                path = future.result()
                if path:
                    pdf["download_path"] = str(path)
                    pdf["download_status"] = "success"
                    downloaded.append(pdf)
                    logger.info(f"Downloaded: {pdf.get('title', '')[:50]}")
                else:
                    pdf["download_status"] = "failed"
                    logger.warning(f"Download failed: {pdf['url']}")
            except Exception as e:
                pdf["download_status"] = "error"
                logger.error(f"Download error for {pdf['url']}: {e}")

    logger.info(f"Downloaded {len(downloaded)}/{len(pdfs)} PDFs")
    return downloaded
