# agentic/standards/enrichment.py
# =============================================================================
# STANDARDS MODULE - PARALLEL ENRICHMENT ENGINE
# =============================================================================
#
# This file consolidates enrichment functions from:
# - standards_rag_enrichment.py (enrich_identified_items_with_standards, etc.)
# - parallel_standards_enrichment.py (ParallelStandardsEnrichment class)
# - optimized_agent.py (is_valid_spec_value, deduplicate_and_merge_specifications)
# - parallel_enrichment.py (run_parallel_3_source_enrichment)
#
# =============================================================================

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from .constants import (
    MIN_STANDARDS_SPECS_COUNT,
    MAX_SPECS_PER_ITEM,
    CHUNK_SIZE,
)
from .keywords import FIELD_GROUPS
from .detector import classify_domain, get_routed_documents
from .cache import get_cached_standards, cache_standards

logger = logging.getLogger(__name__)


from common.agentic.deep_agent.memory.memory import SpecificationSource
from common.agentic.deep_agent.specifications.aggregator import SpecificationAggregator

# Import validation/normalization from consolidated module
from common.infrastructure.normalization import (
    is_valid_spec_value,
    is_valid_spec_key,
    normalize_spec_key,
    clean_and_flatten_specs,
    deduplicate_and_merge_list,
)


# =============================================================================
# VALIDATION FUNCTIONS - Now imported from infrastructure.normalization
# Local definitions removed to avoid duplication
# =============================================================================

# _is_valid_spec_value_DEPRECATED - removed, using import from infrastructure.normalization


# is_valid_spec_key - now imported from common.infrastructure.normalization
# normalize_spec_key - now imported from common.infrastructure.normalization


# clean_and_flatten_specs - removed, using import from infrastructure.normalization





# deduplicate_and_merge_list - removed, using import from infrastructure.normalization


# =============================================================================
# CATEGORY NORMALIZATION (from standards_rag_enrichment.py)
# =============================================================================

def normalize_category(product_type: str, standards: List[str]) -> str:
    """
    Normalize product category based on standards terminology.

    Args:
        product_type: Raw product type string
        standards: List of applicable standards

    Returns:
        Normalized category string
    """
    normalizations = {
        "temperature transmitter": "TT",
        "pressure transmitter": "PT",
        "flow meter": "FT",
        "level transmitter": "LT",
        "control valve": "CV",
        "isolation valve": "XV",
        "pressure gauge": "PG",
        "temperature sensor": "TE",
        "thermocouple": "TE",
        "rtd": "TE",
        "flow switch": "FS",
        "pressure switch": "PS",
        "level switch": "LS",
        "temperature switch": "TS",
    }

    product_lower = product_type.lower()
    for pattern, code in normalizations.items():
        if pattern in product_lower:
            return code

    return product_type.upper()[:2]


# =============================================================================
# SINGLE ITEM ENRICHMENT (from standards_rag_enrichment.py)
# =============================================================================

def enrich_single_item(
    item: Dict[str, Any],
    product_type_context: Optional[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Enrich a single item with standards. Used by parallel executor.

    OPTIMIZATION: Uses domain-based document routing to query only relevant
    standards documents instead of all 12+ documents.

    Args:
        item: Item to enrich
        product_type_context: Context product type
        top_k: Number of standards documents to retrieve

    Returns:
        Enriched item with standards_info
    """
    try:
        from common.tools.standards_enrichment_tool import get_applicable_standards

        # Determine the product type for this item
        item_product_type = (
            item.get("category") or
            item.get("product_name") or
            item.get("name") or
            product_type_context or
            "industrial instrument"
        )

        # Domain-based routing
        user_specs = str(item.get("specifications", {}))
        domains = classify_domain(user_specs, item_product_type, max_domains=3)
        routed_documents = get_routed_documents(domains)

        logger.info(
            f"[StandardsEnrichment] Domain routing for '{item_product_type}': "
            f"domains={[d.value for d in domains]}, "
            f"docs={routed_documents}"
        )

        # Check cache first
        source_filter = ",".join(sorted(routed_documents)) if routed_documents else None
        cached_result = get_cached_standards(item_product_type, source_filter)
        if cached_result is not None:
            logger.info(f"[StandardsEnrichment] Cache HIT for: {item_product_type}")
            item["standards_info"] = cached_result.copy()
            item["standards_info"]["from_cache"] = True
            if cached_result.get("applicable_standards"):
                item["normalized_category"] = normalize_category(
                    item_product_type,
                    cached_result.get("applicable_standards", [])
                )
            return item

        logger.info(f"[StandardsEnrichment] Querying standards for: {item_product_type}")

        # Call standards RAG
        standards_result = get_applicable_standards(
            product_type=item_product_type,
            top_k=top_k,
            source_filter=routed_documents if routed_documents else None
        )

        if standards_result.get("success"):
            standards_info = {
                "applicable_standards": standards_result.get("applicable_standards", []),
                "certifications": standards_result.get("certifications", []),
                "safety_requirements": standards_result.get("safety_requirements", {}),
                "communication_protocols": standards_result.get("communication_protocols", []),
                "environmental_requirements": standards_result.get("environmental_requirements", {}),
                "confidence": standards_result.get("confidence", 0.0),
                "sources": standards_result.get("sources", []),
                "enrichment_status": "success",
                "from_cache": False
            }

            item["standards_info"] = standards_info
            cache_standards(item_product_type, standards_info, source_filter)

            if standards_result.get("applicable_standards"):
                item["normalized_category"] = normalize_category(
                    item_product_type,
                    standards_result.get("applicable_standards", [])
                )

            logger.info(
                f"[StandardsEnrichment] Enriched {item_product_type}: "
                f"{len(standards_result.get('applicable_standards', []))} standards"
            )
        else:
            item["standards_info"] = {
                "enrichment_status": "failed",
                "error": standards_result.get("error", "Unknown error"),
                "from_cache": False
            }
            logger.warning(f"[StandardsEnrichment] Failed to enrich {item_product_type}")

    except Exception as e:
        logger.error(f"[StandardsEnrichment] Error enriching item: {e}")
        item["standards_info"] = {
            "enrichment_status": "error",
            "error": str(e),
            "from_cache": False
        }

    return item


# =============================================================================
# PARALLEL ENRICHMENT (from standards_rag_enrichment.py)
# =============================================================================

def enrich_items_parallel(
    items: List[Dict[str, Any]],
    product_type: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 3,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Enriches identified instruments/accessories with Standards RAG data.

    PERFORMANCE OPTIMIZATIONS:
    - Uses parallel processing (ThreadPoolExecutor) for multiple items
    - Caches results by product type to avoid duplicate queries
    - Uses singleton agent for LLM/vector store connections

    Args:
        items: List of identified instruments/accessories
        product_type: Optional product type context
        domain: Optional industry domain
        top_k: Number of standards documents to retrieve per query
        max_workers: Maximum parallel threads for enrichment

    Returns:
        List of enriched items with standards_info field added
    """
    if not items:
        logger.info("[StandardsEnrichment] No items to enrich")
        return items

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("PARALLEL STANDARDS ENRICHMENT STARTING")
    logger.info(f"   Items to enrich: {len(items)}")
    logger.info(f"   Max workers: {max_workers}")
    logger.info("=" * 70)

    try:
        actual_workers = min(len(items), max_workers)

        if len(items) == 1:
            enriched_items = [enrich_single_item(items[0], product_type, top_k)]
        else:
            enriched_items = []

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_item = {
                    executor.submit(enrich_single_item, item.copy(), product_type, top_k): idx
                    for idx, item in enumerate(items)
                }

                results = [None] * len(items)
                for future in as_completed(future_to_item):
                    idx = future_to_item[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"[StandardsEnrichment] Future error for item {idx}: {e}")
                        original_item = items[idx].copy()
                        original_item["standards_info"] = {
                            "enrichment_status": "error",
                            "error": str(e)
                        }
                        results[idx] = original_item

                enriched_items = results

        elapsed_time = time.time() - start_time
        cache_hits = sum(1 for item in enriched_items if item.get("standards_info", {}).get("from_cache", False))

        logger.info("=" * 70)
        logger.info("PARALLEL STANDARDS ENRICHMENT COMPLETED")
        logger.info(f"   Total time: {elapsed_time:.2f}s")
        logger.info(f"   Items enriched: {len(enriched_items)}")
        logger.info(f"   Cache hits: {cache_hits}")
        logger.info("=" * 70)

        return enriched_items

    except Exception as e:
        logger.error(f"[StandardsEnrichment] Unexpected error: {e}", exc_info=True)
        for item in items:
            item["standards_info"] = {
                "enrichment_status": "error",
                "error": str(e)
            }
        return items


# =============================================================================
# VALIDATION AGAINST STANDARDS (from standards_rag_enrichment.py)
# =============================================================================

def validate_items_against_standards(
    items: List[Dict[str, Any]],
    domain: str,
    safety_requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validates identified items against domain-specific standards.

    Uses cached standards_specifications from Phase 3 enrichment instead of
    re-running RAG when possible.

    Args:
        items: List of identified instruments/accessories
        domain: Industry domain
        safety_requirements: Safety requirements from solution analysis

    Returns:
        Validation result with compliance status and issues
    """
    logger.info(f"[StandardsValidation] Validating {len(items)} items for domain: {domain}")

    validation_result = {
        "is_compliant": True,
        "compliance_issues": [],
        "recommendations": [],
        "domain": domain,
        "cache_hits": 0,
        "cache_misses": 0
    }

    try:
        from common.tools.standards_enrichment_tool import validate_requirements_against_standards

        for item in items:
            item_product_type = item.get("category") or item.get("name") or "instrument"
            item_name = item.get("name") or item.get("product_name") or "Unknown"

            # Check if item was enriched in phase3 with sufficient specs
            if item.get("enrichment_source") == "phase3_optimized":
                total_specs_count = item.get("standards_info", {}).get("total_specs_count", 0)

                if total_specs_count == 0:
                    combined_specs = item.get("combined_specifications", {})
                    total_specs_count = len([v for v in combined_specs.values()
                                            if v and str(v).lower() not in ["null", "none", ""]])

                if total_specs_count >= MIN_STANDARDS_SPECS_COUNT:
                    logger.info(
                        f"[StandardsValidation] SKIPPING RAG for {item_name} - "
                        f"enriched via phase3 with {total_specs_count} specs"
                    )
                    validation_result["cache_hits"] += 1
                    continue

            # Check cached standards_specifications
            if item.get("standards_specifications"):
                total_specs_count = item.get("standards_info", {}).get("total_specs_count", 0)
                cached_specs = item.get("standards_specifications", {})

                if total_specs_count == 0:
                    combined_specs = item.get("combined_specifications", {})
                    if combined_specs:
                        total_specs_count = len([v for v in combined_specs.values()
                                                if v and str(v).lower() not in ["null", "none", ""]])
                    else:
                        total_specs_count = len([v for v in cached_specs.values()
                                                if v and str(v).lower() not in ["null", "none", ""]])

                if total_specs_count >= MIN_STANDARDS_SPECS_COUNT:
                    logger.info(
                        f"[StandardsValidation] Using CACHED standards for {item_name} - "
                        f"{total_specs_count} specs"
                    )
                    validation_result["cache_hits"] += 1
                    continue

            # Cache miss - run RAG
            logger.info(f"[StandardsValidation] Cache MISS for {item_name} - running RAG")
            validation_result["cache_misses"] += 1

            requirements = {
                "product_type": item_product_type,
                "specifications": item.get("specifications", {}),
            }

            if safety_requirements:
                if safety_requirements.get("sil_level"):
                    requirements["sil_level"] = safety_requirements["sil_level"]
                if safety_requirements.get("hazardous_area"):
                    requirements["hazardous_area"] = True
                if safety_requirements.get("atex_zone"):
                    requirements["atex_zone"] = safety_requirements["atex_zone"]

            try:
                val_result = validate_requirements_against_standards(
                    product_type=item_product_type,
                    requirements=requirements
                )

                if not val_result.get("is_compliant"):
                    validation_result["is_compliant"] = False
                    validation_result["compliance_issues"].extend([
                        {"item": item_name, **issue}
                        for issue in val_result.get("compliance_issues", [])
                    ])

                validation_result["recommendations"].extend([
                    {"item": item_name, **rec}
                    for rec in val_result.get("recommendations", [])
                ])

            except Exception as e:
                logger.warning(f"[StandardsValidation] Validation failed for {item_product_type}: {e}")

    except ImportError:
        logger.warning("[StandardsValidation] Standards validation tool not available")
        validation_result["validation_status"] = "unavailable"
    except Exception as e:
        logger.error(f"[StandardsValidation] Validation error: {e}")
        validation_result["validation_status"] = "error"
        validation_result["error"] = str(e)

    logger.info(
        f"[StandardsValidation] Complete - Cache hits: {validation_result['cache_hits']}, "
        f"Misses: {validation_result['cache_misses']}"
    )

    return validation_result


# =============================================================================
# 3-SOURCE ENRICHMENT (from parallel_enrichment.py)
# =============================================================================

def run_3_source_enrichment(
    product_type: str,
    user_input: Optional[str] = None,
    user_specs: Optional[Dict[str, Any]] = None,
    standards_context: Optional[str] = None,
    vendor_data: Optional[List[Dict[str, Any]]] = None,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Run enrichment from User, Standards, and LLM sources in parallel.

    Args:
        product_type: Type of product
        user_input: Raw user text
        user_specs: Structured user specs
        standards_context: Context for standards search
        vendor_data: Vendor data (not yet fully parallelized)
        max_workers: Thread pool size

    Returns:
        Merged dictionary of specifications
    """
    start_time = time.time()
    logger.info(f"[ParallelEnrichment] Starting 3-source enrichment for {product_type}")

    results = {
        "user": {},
        "standards": {},
        "llm": {}
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # 1. User Specs Task
        if user_input or user_specs:
            futures["user"] = executor.submit(
                _extract_user_specs_task,
                product_type,
                user_input,
                user_specs
            )

        # 2. Standards Task
        if standards_context:
            futures["standards"] = executor.submit(
                _extract_standards_specs_task,
                product_type,
                standards_context
            )

        # 3. LLM Generation Task
        futures["llm"] = executor.submit(
            _generate_llm_specs_task,
            product_type,
            user_input or ""
        )

        # Wait for all
        for source, future in futures.items():
            try:
                res = future.result(timeout=60)
                results[source] = res
                logger.info(f"[ParallelEnrichment] {source} task completed with {len(res)} specs")
            except Exception as e:
                logger.error(f"[ParallelEnrichment] {source} task failed: {e}")

    # Merge results
    merged_specs = deduplicate_and_merge_list([
        results["user"],
        results["standards"],
        results["llm"]
    ])

    elapsed = time.time() - start_time
    logger.info(f"[ParallelEnrichment] Completed in {elapsed:.2f}s. Total specs: {len(merged_specs)}")
    return merged_specs


def _extract_user_specs_task(product_type, user_input, user_specs):
    """Worker task for extracting user-specified specs."""
    specs = {}
    if user_specs:
        specs.update(user_specs)
    if user_input:
        try:
            from common.agentic.deep_agent.specifications.generation.llm_generator import extract_user_specified_specs
            extracted = extract_user_specified_specs(user_input, product_type)
            for k, v in extracted.items():
                if k not in specs:
                    specs[k] = v
        except Exception as e:
            logger.warning(f"User spec task error: {e}")
    return specs


def _extract_standards_specs_task(product_type, context):
    """Worker task for extracting standards specs."""
    try:
        from common.tools.standards_enrichment_tool import get_applicable_standards
        result = get_applicable_standards(product_type)

        specs = {}
        if result.get("applicable_standards"):
            specs["Applicable Standards"] = ", ".join(result["applicable_standards"])
        if result.get("certifications"):
            specs["Certifications"] = ", ".join(result["certifications"])
        return specs
    except Exception as e:
        logger.warning(f"Standards task error: {e}")
        return {}


def _generate_llm_specs_task(product_type, context):
    """Worker task for generating LLM specs."""
    try:
        from common.agentic.deep_agent.specifications.generation.llm_generator import generate_llm_specs
        result = generate_llm_specs(product_type, product_type, f"Context: {context}")
        return result.get("specifications", {})
    except Exception as e:
        logger.warning(f"LLM task error: {e}")
        return {}


# =============================================================================
# SCHEMA FIELD EXTRACTION (from standards_enrichment_tool.py)
# =============================================================================

def extract_all_schema_fields(schema: Dict[str, Any]) -> List[str]:
    """
    Extract all field names from a schema recursively.

    Args:
        schema: Schema dictionary

    Returns:
        List of field names
    """
    fields = []

    def extract_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("_"):
                    continue
                full_key = f"{prefix}.{key}" if prefix else key
                fields.append(full_key)
                if isinstance(value, dict):
                    extract_recursive(value, full_key)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    extract_recursive(value[0], full_key)

    extract_recursive(schema)
    return fields


def extract_field_value_from_answer(field_name: str, answer: str) -> Optional[str]:
    """
    Extract a field value from a RAG answer.

    Args:
        field_name: Name of field to extract
        answer: RAG answer text

    Returns:
        Extracted value or None
    """
    if not answer:
        return None

    # Try various patterns
    patterns = [
        rf"{field_name}[:\s]+([^\n,]+)",
        rf"{field_name.replace('_', ' ')}[:\s]+([^\n,]+)",
        rf"{field_name.replace('_', '-')}[:\s]+([^\n,]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if is_valid_spec_value(value):
                return value

    return None


def apply_fields_to_schema(
    schema: Dict[str, Any],
    field_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply extracted field values to schema.

    Args:
        schema: Original schema
        field_values: Dict of field -> value

    Returns:
        Updated schema
    """
    updated = schema.copy()

    for field, value in field_values.items():
        if not is_valid_spec_value(value):
            continue

        # Handle dot notation for nested fields
        parts = field.split(".")
        current = updated
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return updated


# =============================================================================
# PARALLEL SCHEMA ENRICHER CLASS (from parallel_standards_enrichment.py)
# =============================================================================

class ParallelSchemaEnricher:
    """
    Parallel schema enrichment by field groups.

    Processes different field groups in parallel for faster enrichment.
    """

    def __init__(self, max_workers: int = 5, top_k: int = 3):
        self.max_workers = max_workers
        self.top_k = top_k

    def enrich_schema(
        self,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich schema by processing field groups in parallel.

        Args:
            product_type: Product type
            schema: Schema to enrich

        Returns:
            Enriched schema
        """
        start_time = time.time()
        logger.info(f"[ParallelSchemaEnricher] Starting enrichment for {product_type}")

        # Extract all fields from schema
        all_fields = extract_all_schema_fields(schema)
        logger.info(f"[ParallelSchemaEnricher] Found {len(all_fields)} fields")

        # Group fields
        field_groups = self._group_fields(all_fields)

        # Process groups in parallel
        all_values = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for group_name, fields in field_groups.items():
                if fields:
                    futures[group_name] = executor.submit(
                        self._process_field_group,
                        product_type,
                        group_name,
                        fields
                    )

            for group_name, future in futures.items():
                try:
                    result = future.result(timeout=60)
                    all_values.update(result)
                    logger.info(f"[ParallelSchemaEnricher] {group_name}: {len(result)} values")
                except Exception as e:
                    logger.warning(f"[ParallelSchemaEnricher] {group_name} failed: {e}")

        # Apply values to schema
        enriched = apply_fields_to_schema(schema, all_values)

        elapsed = time.time() - start_time
        logger.info(f"[ParallelSchemaEnricher] Completed in {elapsed:.2f}s")

        return enriched

    def _group_fields(self, fields: List[str]) -> Dict[str, List[str]]:
        """Group fields by category."""
        grouped = {name: [] for name in FIELD_GROUPS.keys()}
        grouped["other"] = []

        for field in fields:
            field_lower = field.lower()
            matched = False
            for group_name, group_keywords in FIELD_GROUPS.items():
                if any(kw in field_lower for kw in group_keywords):
                    grouped[group_name].append(field)
                    matched = True
                    break
            if not matched:
                grouped["other"].append(field)

        return grouped

    def _process_field_group(
        self,
        product_type: str,
        group_name: str,
        fields: List[str]
    ) -> Dict[str, Any]:
        """Process a single field group."""
        values = {}

        try:
            query = self._build_query_for_fields(product_type, group_name, fields)
            answer = self._run_standards_rag_query(query)

            if answer:
                for field in fields:
                    value = extract_field_value_from_answer(field, answer)
                    if value:
                        values[field] = value

        except Exception as e:
            logger.warning(f"[ParallelSchemaEnricher] Field group {group_name} error: {e}")

        return values

    def _build_query_for_fields(
        self,
        product_type: str,
        group_name: str,
        fields: List[str]
    ) -> str:
        """Build RAG query for a field group."""
        field_list = ", ".join(fields[:10])  # Limit fields in query
        return (
            f"For a {product_type}, what are the standard values for these "
            f"{group_name} parameters: {field_list}? "
            f"Provide specific values with units where applicable."
        )

    def _run_standards_rag_query(self, query: str) -> Optional[str]:
        """Run a RAG query."""
        try:
            from common.standards.rag import run_standards_rag_workflow

            result = run_standards_rag_workflow(
                question=query,
                top_k=self.top_k
            )

            if result.get("status") == "success":
                return result.get("answer", "")
            return None
        except Exception as e:
            logger.warning(f"[ParallelSchemaEnricher] RAG query failed: {e}")
            return None


def enrich_schema_parallel(
    product_type: str,
    schema: Dict[str, Any],
    max_workers: int = 5,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Convenience function for parallel schema enrichment.

    Args:
        product_type: Product type
        schema: Schema to enrich
        max_workers: Max parallel workers
        top_k: Documents to retrieve per query

    Returns:
        Enriched schema
    """
    enricher = ParallelSchemaEnricher(max_workers=max_workers, top_k=top_k)
    return enricher.enrich_schema(product_type, schema)


# =============================================================================
# ASYNC WRAPPER FOR BACKWARD COMPATIBILITY
# =============================================================================

async def enrich_schema_async(
    product_type: str,
    schema: Dict[str, Any],
    max_workers: int = 5,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Async wrapper for parallel schema enrichment.

    This provides backward compatibility for async/await usage patterns
    while using the same parallel implementation under the hood.

    Args:
        product_type: Product type
        schema: Schema to enrich
        max_workers: Max parallel workers
        top_k: Documents to retrieve per query

    Returns:
        Enriched schema
    """
    import asyncio

    loop = asyncio.get_event_loop()
    enricher = ParallelSchemaEnricher(max_workers=max_workers, top_k=top_k)

    # Run the enrichment in a thread pool executor to avoid blocking
    enriched = await loop.run_in_executor(
        None,
        enricher.enrich_schema,
        product_type,
        schema
    )

    return enriched


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "SpecificationSource",
    # Validation
    "is_valid_spec_value",
    "is_valid_spec_key",
    "normalize_spec_key",
    "clean_and_flatten_specs",
    # Merge
    "merge_specifications",
    "deduplicate_and_merge_list",
    # Category
    "normalize_category",
    # Single item
    "enrich_single_item",
    # Parallel enrichment
    "enrich_items_parallel",
    "validate_items_against_standards",
    # 3-source
    "run_3_source_enrichment",
    # Schema
    "extract_all_schema_fields",
    "extract_field_value_from_answer",
    "apply_fields_to_schema",
    "ParallelSchemaEnricher",
    "enrich_schema_parallel",
    "enrich_schema_async",
]
