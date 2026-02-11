# advanced_parameters.py
# Advanced parameter discovery using pure LLM-based approach.
# Focuses on latest parameters added in the past 6 months for the detected product type.

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# Use Azure Blob for advanced parameters storage
from azure_blob_config import get_azure_blob_connection
from config import AgenticConfig
from services.llm.fallback import create_llm_with_fallback
from prompts_library import load_prompt_sections

# Load prompts at module level
# default_section captures content before first section marker as "ADVANCED_PARAMS"
_ADV_PARAMS_PROMPTS = load_prompt_sections("advanced_parameters_prompts", default_section="ADVANCED_PARAMS")

# MONGO_TTL_DAYS removed - advanced parameters persist indefinitely
IN_MEMORY_CACHE_TTL_MINUTES = 10
SCHEMA_CACHE_TTL_MINUTES = 30
COLLECTION_CACHE = None


class BoundedTTLCache:
    """
    Thread-safe cache with TTL expiration and LRU eviction when max size is reached.
    Prevents unbounded memory growth while maintaining recent entries.

    PHASE 2 FIX: Added optional admin token protection to prevent accidental clears.
    """

    def __init__(self, max_size: int = 500, ttl_minutes: int = 10, name: str = "BoundedTTLCache", require_admin: bool = False):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl_minutes = ttl_minutes
        self.name = name  # For logging
        self._lock = threading.Lock()

        # PHASE 2 FIX: Protection against accidental cache clears
        self._require_admin = require_admin
        self._admin_token = None

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            # Check TTL expiration
            if entry["expires_at"] <= datetime.utcnow():
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        """Set item in cache with TTL, evicting oldest if at capacity."""
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Evict oldest entries if at capacity
                while len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)

            self._cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(minutes=self._ttl_minutes),
            }

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return item from cache."""
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                return entry.get("value")
            return default

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self, admin_token: Optional[str] = None) -> int:
        """
        Clear all entries from the cache.

        PHASE 2 FIX: If cache is protected, requires admin token.

        Args:
            admin_token: Admin token (required if cache is protected)

        Returns:
            Number of entries cleared

        Raises:
            PermissionError: If cache is protected and token is invalid
        """
        # Check admin token if cache is protected
        if self._require_admin:
            if admin_token != self._admin_token:
                raise PermissionError(
                    f"Cannot clear protected cache '{self.name}' without valid admin token"
                )

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logging.warning(f"[{self.name}] Cleared {count} entries (admin: {bool(admin_token)})")
            return count

    def set_admin_token(self, token: str) -> None:
        """
        Set admin token required for protected operations.

        PHASE 2 FIX: Call this during initialization to enable cache protection.

        Args:
            token: Admin token (should come from environment variable)
        """
        with self._lock:
            self._admin_token = token
            logging.info(f"[{self.name}] Admin token set (protection enabled)")


# Bounded caches with max 500 items each to prevent OOM
# PHASE 2 FIX: Added names and enabled admin token protection
IN_MEMORY_ADVANCED_SPEC_CACHE = BoundedTTLCache(
    max_size=500,
    ttl_minutes=IN_MEMORY_CACHE_TTL_MINUTES,
    name="IN_MEMORY_ADVANCED_SPEC_CACHE",
    require_admin=True  # Protected - requires admin token to clear
)
SCHEMA_PARAM_CACHE = BoundedTTLCache(
    max_size=500,
    ttl_minutes=SCHEMA_CACHE_TTL_MINUTES,
    name="SCHEMA_PARAM_CACHE",
    require_admin=True  # Protected - requires admin token to clear
)

# Load environment variables
def _normalize_product_type(product_type: str) -> str:
    return re.sub(r"[^a-z0-9]", "", product_type.lower()) if product_type else ""


def _get_in_memory_specs(product_type: str) -> Optional[List[Dict[str, Any]]]:
    if not product_type:
        return None

    normalized = _normalize_product_type(product_type)
    return IN_MEMORY_ADVANCED_SPEC_CACHE.get(normalized)


def _set_in_memory_specs(product_type: str, specs: List[Dict[str, Any]]) -> None:
    if not product_type:
        return

    normalized = _normalize_product_type(product_type)
    IN_MEMORY_ADVANCED_SPEC_CACHE.set(normalized, specs)


class AdvancedParametersDiscovery:
    """Discovers generic advanced specifications for fallback scenarios with OpenAI fallback."""

    def __init__(self):
        self.llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def get_generic_specifications(self, product_type: str) -> List[str]:
        """Get generic advanced specifications with series numbers for product type."""
        prompt = ChatPromptTemplate.from_template(_ADV_PARAMS_PROMPTS["GENERIC_SPECIFICATIONS"])
        try:
            response = (prompt | self.llm | StrOutputParser()).invoke({"product_type": product_type})
            params = json.loads(response.strip().replace('```json', '').replace('```', ''))
            return [str(p) for p in params][:15] if isinstance(params, list) else []
        except:
            return []

def get_existing_parameters(product_type: str) -> set:
    """Get existing parameters from schema with in-memory caching to speed up LLM requests."""
    if not product_type:
        return set()

    normalized = _normalize_product_type(product_type)

    cached = SCHEMA_PARAM_CACHE.get(normalized)
    if cached is not None:
        return cached

    try:
        from core.loading import load_requirements_schema

        schema = load_requirements_schema(product_type)
        if not schema:
            return set()

        params = set()
        for req_type in ["mandatory_requirements", "optional_requirements"]:
            for fields in schema.get(req_type, {}).values():
                if isinstance(fields, dict):
                    params.update(fields.keys())

        SCHEMA_PARAM_CACHE.set(normalized, params)
        return params
    except Exception as exc:
        logging.warning(f"Failed to load schema for {product_type}: {exc}")
        return set()

def convert_specifications_to_human_readable(specs: List[str]) -> Dict[str, str]:
    """Convert specification keys to readable names without extra LLM calls."""
    readable: Dict[str, str] = {}
    for spec in specs:
        readable[spec] = spec.replace("_", " ").title()
    return readable


def _get_advanced_param_collection():
    global COLLECTION_CACHE
    if COLLECTION_CACHE is not None:
        return COLLECTION_CACHE

    try:
        conn = get_azure_blob_connection()
        COLLECTION_CACHE = conn["collections"].get("advanced_parameters")
    except Exception as exc:
        logging.warning(f"Azure advanced parameters collection unavailable: {exc}")
        COLLECTION_CACHE = None

    return COLLECTION_CACHE


def _load_cached_specifications(product_type: str) -> Optional[List[Dict[str, Any]]]:
    in_memory = _get_in_memory_specs(product_type)
    if in_memory is not None:
        return in_memory

    collection = _get_advanced_param_collection()
    if collection is None or not product_type:
        return None

    normalized = _normalize_product_type(product_type)

    # No TTL for advanced parameters - retrieve regardless of age
    try:
        # AzureBlobCollection.find_one returns a dict or None
        doc = collection.find_one(
            {
                "normalized_product_type": normalized,
            }
        )
        if doc:
            specs = doc.get("unique_specifications")
            if isinstance(specs, list) and specs:
                logging.info(f"Advanced parameters cache hit for {product_type}")
                _set_in_memory_specs(product_type, specs)
                return specs
            elif isinstance(specs, list) and not specs:
                logging.info(f"Advanced parameters cache hit (empty) for {product_type} - ignoring to allow retry")
    except Exception as exc:
        logging.warning(f"Unable to read advanced parameters cache for {product_type}: {exc}")

    return None


def _persist_specifications(
    product_type: str,
    unique_specifications: List[Dict[str, Any]],
    existing_snapshot: List[str],
) -> None:
    collection = _get_advanced_param_collection()
    if collection is None or not product_type:
        return

    normalized = _normalize_product_type(product_type)
    now = datetime.utcnow()

    document = {
        "product_type": product_type,
        "normalized_product_type": normalized,
        "unique_specifications": unique_specifications,
        "existing_parameters_snapshot": existing_snapshot,
        "created_at": now,
        "updated_at": now,
    }

    try:
        # AzureBlobCollection.update_one supports simple $set and upsert=True
        collection.update_one(
            {"normalized_product_type": normalized},
            {"$set": document},
            upsert=True,
        )
        logging.info(
            f"Cached advanced parameters in Azure for {product_type} ({len(unique_specifications)} specs)"
        )
        _set_in_memory_specs(product_type, unique_specifications)
    except Exception as exc:
        logging.warning(f"Failed to cache advanced parameters for {product_type}: {exc}")
        # Still keep in-memory cache even if upsert fails
        _set_in_memory_specs(product_type, unique_specifications)


def _build_response(product_type: str, unique_specifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "product_type": product_type,
        "vendor_specifications": [],
        "vendor_parameters": [],
        "unique_specifications": unique_specifications,
        "unique_parameters": unique_specifications,
        "total_vendors_searched": 0,
        "total_unique_specifications": len(unique_specifications),
        "total_unique_parameters": len(unique_specifications),
        "existing_specifications_filtered": 0,
    }


def _extract_json_object(raw_response: str) -> Dict[str, Any]:
    """Extract strict JSON even if the LLM emits conversational text."""
    if not raw_response:
        return {}
    # Remove common markdown fences and surrounding text
    cleaned = raw_response.strip()
    cleaned = cleaned.replace('```json', '').replace('```', '').strip()

    # Try to parse the whole cleaned text first (covers well-formed responses)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find the first JSON object in the text
    obj_start = cleaned.find('{')
    obj_end = cleaned.rfind('}')
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        snippet = cleaned[obj_start:obj_end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            logging.warning('Failed to decode JSON object from LLM response')

    # Fallback: try to find a JSON array (some responses return an array directly)
    arr_start = cleaned.find('[')
    arr_end = cleaned.rfind(']')
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        snippet = cleaned[arr_start:arr_end + 1]
        try:
            arr = json.loads(snippet)
            # Wrap array into object for backward compatibility
            return {"parameters": arr} if isinstance(arr, list) else {}
        except Exception:
            logging.warning('Failed to decode JSON array from LLM response')

    logging.warning('LLM response missing usable JSON')

    # Last-resort attempt: try to extract a simple newline-separated list of items
    lines = [l.strip(' -*.\t') for l in cleaned.splitlines() if l.strip()]
    # Filter out lines that look like non-content
    candidate_items = []
    for line in lines:
        # Skip very short lines or lines that look like instructions
        if len(line) < 3:
            continue
        if line.lower().startswith(('return', 'task', 'rules', 'rules:')):
            continue
        # Remove numbering like '1.' or '- ' prefixes
        candidate_items.append(line.strip())

    if candidate_items:
        logging.info('Parsed %d candidate items from LLM free-text response', len(candidate_items))
        return {"parameters": candidate_items}

    # If nothing usable, log the full raw response for diagnostics and return empty
    logging.warning('[AdvancedParams] LLM response missing usable JSON or parameters!')
    logging.warning(f'[AdvancedParams] Raw LLM response (first 1000 chars): {str(raw_response)[:1000]}')
    logging.debug('Full raw LLM response (no JSON found): %s', raw_response)
    return {}

def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """Discover latest advanced parameters for product type using a single LLM call.

    The LLM is responsible for:
    - Looking at the detected product type.
    - Considering latest advanced specifications from top vendors.
    - Comparing against the existing schema parameters.
    - Returning only **new** advanced specifications in a readable format.

    The public return structure stays compatible with existing callers:
    - "unique_specifications": list of {"key", "name"[, "description"]}
    - "existing_specifications_filtered": kept for logging (approximate)
    - "vendor_specifications": returned as an empty list (no per-vendor breakdown now)
    """

    try:
        if not product_type:
            return {
                "product_type": product_type,
                "vendor_specifications": [],
                "unique_specifications": [],
                "total_vendors_searched": 0,
                "total_unique_specifications": 0,
                "existing_specifications_filtered": 0,
            }

        # We always run the LLM for latest results but attempt to use cache for speed
        cached = _load_cached_specifications(product_type)
        if cached is not None:
            # return cached immediately to avoid extra LLM calls in high-load scenarios
            # but still proceed to refresh the cache in background could be implemented later
            return _build_response(product_type, cached)

        # Load existing parameters from schema so we can ask the LLM for *only new* ones
        existing = get_existing_parameters(product_type)
        existing_list = sorted(list(existing))

        logging.info(f"Starting single-call advanced specification discovery for: {product_type}")

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


        # Request only human-readable parameter names from the LLM.
        # We'll deterministically create a snake_case key from each returned name.
        prompt = ChatPromptTemplate.from_template(_ADV_PARAMS_PROMPTS["PARAMETER_DISCOVERY"])

        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"product_type": product_type, "existing_parameters": json.dumps(existing_list)})

        # Log the raw LLM response for debugging
        logging.info(f"[AdvancedParams] LLM response type: {type(raw_response)}, length: {len(str(raw_response))}")
        logging.info(f"[AdvancedParams] LLM response (first 500 chars): {str(raw_response)[:500]}")

        payload = _extract_json_object(raw_response)

        # FIX: The prompt asks for "advanced_parameters" but code was looking for "parameters"
        # Try both keys to support different prompt versions
        if isinstance(payload, dict):
            raw_params = payload.get("parameters") or payload.get("advanced_parameters") or []
        else:
            raw_params = []

        # Log extracted parameters
        logging.info(f"[AdvancedParams] Extracted payload type: {type(payload)}")
        logging.info(f"[AdvancedParams] Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'N/A'}")
        logging.info(f"[AdvancedParams] Extracted {len(raw_params)} raw parameters: {raw_params[:5] if raw_params else '[]'}")

        # Normalize existing keys for comparison
        existing_norm = {p.lower().replace("_", "") for p in existing}
        seen_norm = set()
        unique_specifications: List[Dict[str, Any]] = []

        for name in raw_params:
            if not isinstance(name, str):
                continue
            human_name = name.strip()
            if not human_name:
                continue

            # Generate a deterministic snake_case key from the human name
            key_candidate = human_name.lower()
            key_candidate = re.sub(r"[^a-z0-9 ]", "", key_candidate)
            key_candidate = re.sub(r"\s+", "_", key_candidate).strip("_")
            norm = key_candidate.replace("_", "")

            if not key_candidate or norm in existing_norm or norm in seen_norm:
                continue

            unique_specifications.append({"key": key_candidate, "name": human_name})
            seen_norm.add(norm)

        logging.info(
            f"Advanced specifications (single-call) discovery complete: {len(unique_specifications)} new specifications returned"
        )

        if unique_specifications:
            _persist_specifications(product_type, unique_specifications, existing_list)
        else:
             logging.info(f"No unique specifications found for {product_type}. Skipping persistence to allow retry.")

        return _build_response(product_type, unique_specifications)
    except Exception as e:
        logging.error(f"Discovery failed (single-call): {e}")
        # Fallback to generic specs path to avoid breaking the UI completely
        discovery = AdvancedParametersDiscovery()
        generic_specs = discovery.get_generic_specifications(product_type)
        
        human_readable_map = convert_specifications_to_human_readable(generic_specs)
        
        fallback_specs = [
            {"key": p, "name": human_readable_map.get(p, p.replace('_', ' ').title())}
            for p in generic_specs
        ]
        fallback_response = _build_response(product_type, fallback_specs)
        fallback_response["fallback"] = True
        return fallback_response
