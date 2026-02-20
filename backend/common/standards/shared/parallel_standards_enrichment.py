"""
common/standards/shared/parallel_standards_enrichment.py
=========================================================

Provides ``ParallelStandardsEnrichment`` — a thin wrapper around
``ParallelSchemaEnricher`` (defined in enrichment.py) that exposes the
``enrich_schema_in_parallel`` method expected by
``product_search_workflow.enrich_schema_parallel()``.

The actual parallel logic lives in ``ParallelSchemaEnricher``.  This file
only exists so callers can do:

    from common.standards.shared.parallel_standards_enrichment import (
        ParallelStandardsEnrichment,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from common.standards.shared.enrichment import ParallelSchemaEnricher

logger = logging.getLogger(__name__)


class ParallelStandardsEnrichment:
    """
    Parallel standards enrichment for product schemas.

    Thin facade over ``ParallelSchemaEnricher`` that matches the calling
    convention used in ``product_search_workflow.enrich_schema_parallel()``.

    Usage::

        enricher = ParallelStandardsEnrichment(max_workers=5)
        enriched_schema = enricher.enrich_schema_in_parallel(product_type, schema)
    """

    def __init__(self, max_workers: int = 5, top_k: int = 3) -> None:
        self._enricher = ParallelSchemaEnricher(
            max_workers=max_workers, top_k=top_k
        )
        self.max_workers = max_workers
        self.top_k = top_k

    def enrich_schema_in_parallel(
        self,
        product_type: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich *schema* with standards data for *product_type* using a
        thread pool so that different field groups are queried simultaneously.

        Args:
            product_type: Human-readable product type, e.g. "pressure transmitter".
            schema: The existing schema dict (will not be mutated).

        Returns:
            A new schema dict with standards values filled in where available.
        """
        logger.info(
            "[ParallelStandardsEnrichment] enrich_schema_in_parallel for '%s'",
            product_type,
        )
        try:
            enriched = self._enricher.enrich_schema(
                product_type=product_type,
                schema=schema,
            )
            logger.info(
                "[ParallelStandardsEnrichment] ✓ Done for '%s'", product_type
            )
            return enriched
        except Exception as exc:
            logger.error(
                "[ParallelStandardsEnrichment] Failed for '%s': %s — returning schema unchanged",
                product_type,
                exc,
                exc_info=True,
            )
            return schema


__all__ = ["ParallelStandardsEnrichment"]
