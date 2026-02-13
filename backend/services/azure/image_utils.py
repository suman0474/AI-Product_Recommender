"""
Generic Product Type Image Utilities
Handles fetching, caching, and retrieving generic product type images
using ONLY Azure Blob Storage with LLM-based generation via Gemini Imagen 4.0.

ARCHITECTURE (Azure-Only):
1. Check Azure Blob Storage for cached image
2. If not found, check if generation is in-progress (request deduplication)
3. Generate via Gemini Imagen 4.0 LLM (NO rate limiting between requests)
4. Store result in Azure Blob Storage ONLY
5. Return URL/metadata to caller

RATE LIMIT HANDLING: Exponential backoff for 429 errors.
REQUEST DEDUPLICATION: Only one LLM call per product type for concurrent requests.
STORAGE: Azure Blob Storage (primary and only storage).
"""

import logging
import os
import io
import json
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image
import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)

# Issue-specific debug logging for terminal log analysis
try:
    from debug_flags import issue_debug
except ImportError:
    issue_debug = None  # Fallback if debug_flags not available

# Align environment keys with main.py
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini client for image generation
_gemini_client = None
if GOOGLE_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("[INIT] Gemini client initialized for image generation")
    except Exception as e:
        logger.warning(f"[INIT] Failed to initialize Gemini client: {e}")

# =============================================================================
# REQUEST DEDUPLICATION SYSTEM & THROTTLING
# =============================================================================
# Prevents duplicate concurrent LLM calls for the same product type.
# Only the first request generates; subsequent concurrent requests wait for the result.

from agentic.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

_generation_lock = threading.Lock()  # Lock for single execution

# FIX: Bounded caches with automatic TTL expiration and LRU eviction
# Prevents unbounded memory growth in long-running sessions
_global_generation_cache: BoundedCache = get_or_create_cache(
    name="image_generation",
    max_size=500,        # Limit to 500 product types
    ttl_seconds=3600     # 1 hour expiry
)

_pending_requests_cache: BoundedCache = get_or_create_cache(
    name="image_pending_requests",
    max_size=100,        # Limit pending requests
    ttl_seconds=300      # 5 minute timeout for pending requests
)

_pending_results_cache: BoundedCache = get_or_create_cache(
    name="image_pending_results",
    max_size=100,
    ttl_seconds=300
)

_CACHE_EXPIRY_SECONDS = 3600  # 1 hour (kept for backwards compatibility)

# FIX #3: Rate limiting throttling - prevents thundering herd
_last_request_time = 0  # Timestamp of last LLM request
_throttle_lock = threading.Lock()
_MIN_REQUEST_INTERVAL = 2.0  # Minimum 2 seconds between requests to avoid rate limits


# FIX #4: Failure Tracking & Retry Persistence
class FailedGenerationTracker:
    """
    Tracks failed image generation attempts to support robust retries.
    Persists data to disk to survive application restarts.
    """
    def __init__(self, persistence_file: str = "failed_image_generations.json"):
        self.persistence_file = os.path.join(os.getcwd(), persistence_file)
        self.failures: Dict[str, Dict[str, Any]] = {}  # product_type -> metadata
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        """Load failure registry from disk."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    self.failures = json.load(f)
                logger.info(f"[TRACKER] Loaded {len(self.failures)} failed image generations from disk")
            except Exception as e:
                logger.warning(f"[TRACKER] Failed to load failure registry: {e}")
                self.failures = {}

    def _save(self):
        """Save failure registry to disk."""
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(self.failures, f, indent=2)
        except Exception as e:
            logger.error(f"[TRACKER] Failed to save failure registry: {e}")

    def record_failure(self, product_type: str, error_reason: str = "unknown"):
        """Record a failure for a product type."""
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        with self.lock:
            if normalized_type not in self.failures:
                self.failures[normalized_type] = {
                    "product_type": product_type,
                    "first_failure": datetime.utcnow().isoformat(),
                    "retry_count": 0,
                    "last_attempt": datetime.utcnow().isoformat(),
                    "error_reason": error_reason,
                    "status": "pending_retry"
                }
            else:
                entry = self.failures[normalized_type]
                # Increment retry count
                entry["retry_count"] = entry.get("retry_count", 0) + 1
                entry["last_attempt"] = datetime.utcnow().isoformat()
                entry["error_reason"] = error_reason
                
                # Check max retries (User requested max 2 total attempts -> 1 retry)
                if entry["retry_count"] >= 1: # 0 = first failure, 1 = first retry failure
                    entry["status"] = "permanently_failed"
                    logger.warning(f"[TRACKER] Product '{product_type}' reached max retries (2 total attempts). Marking permanently failed.")
            
            self._save()
            logger.info(f"[TRACKER] Recorded failure for '{product_type}' (Total attempts: {self.failures[normalized_type]['retry_count'] + 1})")

    def get_pending_retries(self) -> List[Dict[str, Any]]:
        """Get list of items that need retry."""
        pending = []
        with self.lock:
            for key, data in self.failures.items():
                if data.get("status") == "pending_retry":
                    pending.append(data)
        return pending

    def mark_success(self, product_type: str):
        """Remove item from failure tracker upon success."""
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        with self.lock:
            if normalized_type in self.failures:
                del self.failures[normalized_type]
                self._save()
                logger.info(f"[TRACKER] Cleared failure record for '{product_type}' (Success)")

# Global tracker instance
_failure_tracker = FailedGenerationTracker()


def _apply_request_throttle():
    """
    FIX #3: Apply request throttling with exponential backoff and jitter.

    Ensures minimum delay between LLM requests to avoid rate limiting.
    Uses jitter to prevent thundering herd when multiple requests arrive simultaneously.

    Returns:
        float: Actual wait time applied (seconds)
    """
    import random

    global _last_request_time
    current_time = time.time()

    with _throttle_lock:
        time_since_last = current_time - _last_request_time

        if time_since_last < _MIN_REQUEST_INTERVAL:
            # Add jitter (±25% of wait time) to prevent synchronization
            wait_time = _MIN_REQUEST_INTERVAL - time_since_last
            jitter = random.uniform(0, 0.25 * wait_time)
            actual_wait = wait_time + jitter

            logger.info(
                f"[FIX3] Request throttling: waiting {actual_wait:.2f}s "
                f"(jitter: {jitter:.2f}s) to maintain {_MIN_REQUEST_INTERVAL}s interval"
            )
            time.sleep(actual_wait)
            _last_request_time = time.time()
            return actual_wait
        else:
            _last_request_time = current_time
            return 0.0



def _get_visual_description(product_type: str) -> Optional[str]:
    """
    Get a visual description of the product using a text LLM.
    This helps ground the image generation for obscure industrial parts.
    """
    global _gemini_client
    if not _gemini_client:
        return None

    try:
        logger.info(f"[LLM_DESC] Generating visual description for '{product_type}'...")
        prompt = (
            f"Describe the **isolated physical form** of an industrial {product_type} for a 3D artist. "
            f"Focus on what makes a {product_type} VISUALLY DISTINCT from other industrial instruments. "
            "Describe it as a standalone component floating in whitespace. "
            "Focus on shape, materials (e.g. stainless steel, plastic), and key visible components (e.g. LCD screen, flanges, tubes, U-shaped pipes). "
            "Do not mention usage. Start with 'A ...'"
        )
        
        # Use gemini-2.0-flash for speed/cost
        response = _gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=120
            )
        )
        
        if response.text:
            description = response.text.strip()
            logger.info(f"[LLM_DESC] Description: {description}")
            return description
            
    except Exception as e:
        logger.warning(f"[LLM_DESC] Failed to generate description: {e}")
    
    return None


def _generate_image_with_llm(product_type: str, retry_count: int = 0, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Generate a generic product image using Gemini's Imagen model
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
        retry_count: Current retry attempt (for internal use)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing generated image data or None if generation failed
    """
    global _gemini_client
    
    if not _gemini_client:
        logger.error("[LLM_IMAGE_GEN] Gemini client not initialized. Check GOOGLE_API_KEY.")
        return None
    
    try:
        logger.info(f"[LLM_IMAGE_GEN] Generating image for '{product_type}' using Gemini Imagen model...")

        # FIX #3: Apply request throttling to prevent rate limiting
        throttle_wait = _apply_request_throttle()
        if throttle_wait > 0:
            logger.info(f"[FIX3] Applied throttling: {throttle_wait:.2f}s")

        # Step 1: Get Visual Description (Dynamic Prompt Enrichment)
        visual_description = _get_visual_description(product_type)

        # Step 2: Construct the prompt for image generation
        # Enforce industrial context to avoid ambiguous interpretations (e.g. "Terminal Head" -> Robot)
        if visual_description:
            # Use the LLM-generated description
            clean_desc = visual_description.rstrip('.')
            prompt = (
                f"A professional product catalog shot of {clean_desc}. "
                f"(Context: {product_type} - industrial automation component). "
                "Style: Industrial product photography, flat lighting, soft even illumination, "
                "white background, high resolution. "
                "Composition: Floating object, fully isolated. "
                "Exclude: shadows, cast shadows, drop shadows, floor, surface, table, reflection, "
                "dramatic lighting, high contrast, cinematic, "
                "faces, characters, robots, organic, sci-fi."
            )
        else:
            # Fallback to template if description fails
            prompt = (
                f"A professional product catalog shot of a {product_type}. "
                f"(Context: {product_type} - industrial automation component). "
                "Style: Industrial product photography, flat lighting, soft even illumination, "
                "white background, high resolution. "
                "Composition: Floating object, fully isolated. "
                "Exclude: shadows, cast shadows, drop shadows, floor, surface, table, reflection, "
                "dramatic lighting, high contrast, cinematic, "
                "faces, characters, robots, organic, sci-fi."
            )

        logger.info(f"[LLM_IMAGE_GEN] Prompt: {prompt}")

        # Use Gemini's Imagen 4.0 model for image generation
        response = _gemini_client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                # aspect_ratio can be "1:1", "3:4", "4:3", "9:16", "16:9"
                # For generic product images, square is best
            )
        )
        
        if not response.generated_images:
            logger.warning(f"[LLM_IMAGE_GEN] No images generated for '{product_type}'")
            return None
        
        # Get the first generated image
        generated_image = response.generated_images[0]
        
        # The image bytes are directly available from the response
        # The google.genai library provides the image as bytes
        image_bytes = generated_image.image.image_bytes
        
        logger.info(f"[LLM_IMAGE_GEN] ✓ Successfully generated image for '{product_type}' (size: {len(image_bytes)} bytes)")
        if issue_debug:
            issue_debug.image_llm_generation(product_type, success=True, time_ms=0)
        
        return {
            'image_bytes': image_bytes,
            'content_type': 'image/png',
            'file_size': len(image_bytes),
            'source': 'gemini_imagen',
            'prompt': prompt
        }
        
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a rate limit error (429)
        if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
            if retry_count < max_retries:
                # FIX3: Retry logic for 429 errors
                # The original max_retries is passed as an argument, use that.
                # The original retry_count is passed as an argument, use that.
                retry_delay = 2  # Start with 2 seconds
                
                # Calculate wait time based on current retry_count
                wait_time = retry_delay * (2 ** retry_count)  # 2s, 4s, 8s
                import random
                jitter = random.uniform(0.1, 1.0)
                total_wait = wait_time + jitter
                
                logger.warning(
                    f"[FIX3] Rate limit hit for '{product_type}'. "
                    f"Retry {retry_count + 1}/{max_retries} after {total_wait:.1f}s "
                    f"(base: {wait_time:.1f}s, jitter: {jitter:.1f}s)..."
                )
                time.sleep(total_wait)
                
                # Retry the request
                return _generate_image_with_llm(product_type, retry_count + 1, max_retries)
            else:
                logger.error(f"[LLM_IMAGE_GEN] Max retries ({max_retries}) exceeded for '{product_type}' due to rate limiting")
                return None
        
        # FIX: Retry on network errors (Network is unreachable, Connection refused)
        if 'Network is unreachable' in error_str or 'Connection refused' in error_str or 'ConnectError' in error_str:
            if retry_count < max_retries:
                wait_time = 2 ** (retry_count + 1)  # 2s, 4s, 8s exponential backoff
                logger.warning(
                    f"[LLM_IMAGE_GEN] Network error for '{product_type}'. "
                    f"Retry {retry_count + 1}/{max_retries} after {wait_time}s..."
                )
                time.sleep(wait_time)
                return _generate_image_with_llm(product_type, retry_count + 1, max_retries)
            else:
                logger.error(f"[LLM_IMAGE_GEN] Max retries ({max_retries}) exceeded for '{product_type}' due to network errors")
                return None
        
        logger.error(f"[LLM_IMAGE_GEN] Failed to generate image with Gemini: {e}")
        logger.exception(e)
        return None


# =============================================================================
# FAST-FAIL LLM GENERATION (For UI responsiveness)
# =============================================================================
def _generate_image_with_llm_fast(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Attempt LLM image generation with FAST-FAIL on rate limit.
    
    Instead of waiting 60-240 seconds on 429 errors, returns None immediately.
    This allows the UI to show a placeholder while background retries can happen later.
    
    Returns:
        Image data dict if successful, None if rate-limited or failed
    """
    global _gemini_client
    
    if not _gemini_client:
        logger.warning("[LLM_FAST] Gemini client not initialized")
        return None
    
    try:
        logger.info(f"[LLM_FAST] Attempting fast image generation for '{product_type}'...")
        
        # Apply minimal throttle
        throttle_wait = _apply_request_throttle()
        if throttle_wait > 0:
            logger.info(f"[LLM_FAST] Throttle: {throttle_wait:.2f}s")
        
        # Step 1: Get Visual Description (Fast - still worth the ~500ms for quality)
        visual_description = _get_visual_description(product_type)
        
        # Step 2: Construct Prompt
        if visual_description:
            clean_desc = visual_description.rstrip('.')
            prompt = (
                f"A professional product catalog shot of {clean_desc}. "
                f"(Context: {product_type} - industrial automation component). "
                "Style: Industrial product photography, flat lighting, soft even illumination, "
                "white background, high resolution. "
                "Composition: Floating object, fully isolated. "
                "Exclude: shadows, cast shadows, drop shadows, floor, surface, table, reflection, "
                "dramatic lighting, high contrast, cinematic, "
                "faces, characters, robots, organic, sci-fi."
            )
        else:
            # Enforce industrial context
            prompt = (
                f"A professional product catalog shot of a {product_type}. "
                f"(Context: {product_type} - industrial automation component). "
                "Style: Industrial product photography, flat lighting, soft even illumination, "
                "white background, high resolution. "
                "Composition: Floating object, fully isolated. "
                "Exclude: shadows, cast shadows, drop shadows, floor, surface, table, reflection, "
                "dramatic lighting, high contrast, cinematic, "
                "faces, characters, robots, organic, sci-fi."
            )
        
        response = _gemini_client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1)
        )
        
        if not response.generated_images:
            logger.warning(f"[LLM_FAST] No images generated for '{product_type}'")
            return None
        
        generated_image = response.generated_images[0]
        image_bytes = generated_image.image.image_bytes
        
        logger.info(f"[LLM_FAST] ✓ Generated image for '{product_type}' ({len(image_bytes)} bytes)")
        
        return {
            'image_bytes': image_bytes,
            'content_type': 'image/png',
            'file_size': len(image_bytes),
            'source': 'gemini_imagen',
            'prompt': prompt
        }
        
    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
            logger.warning(f"[LLM_FAST] Rate limit hit for '{product_type}' - returning None immediately (no retry)")
            return None
        
        logger.error(f"[LLM_FAST] Failed: {e}")
        return None




# --- Main Utilities ---

def get_generic_image_from_azure(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached generic product type image from Azure Blob Storage

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image metadata or None if not found
    """
    try:
        from azure_blob_config import Collections, azure_blob_manager
        from azure.core.exceptions import ResourceNotFoundError
        import json

        # Product type aliases for better matching
        PRODUCT_TYPE_ALIASES = {
            'temperaturetransmitter': ['temptransmitter', 'temperaturesensor', 'rtdtransmitter'],
            'pressuretransmitter': ['pressuretransducer', 'pressuresensor'],
            'flowmeter': ['flowtransmitter', 'flowsensor'],
            'leveltransmitter': ['levelindicator', 'levelmeter', 'levelsensor'],
            'controlvalve': ['controlvalves', 'valve'],
            'variablefrequencydrive': ['vfd', 'frequencyinverter', 'acdrive'],
            'thermocouple': ['thermocouplesensor', 'tcassembly'],
            'junctionbox': ['junctionboxes', 'jboxes', 'enclosure'],
            'mountingbracket': ['bracket', 'mountinghardware'],
        }

        # Normalize product type for Azure path
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # Check if this normalized type has a preferred alias
        for canonical, aliases in PRODUCT_TYPE_ALIASES.items():
            if normalized_type in aliases:
                logger.info(f"[AZURE_CHECK] Alias mapping: '{normalized_type}' -> '{canonical}'")
                normalized_type = canonical
                break

        # Construct metadata path (relative to base path)
        metadata_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.json"

        # Check if azure_blob_manager is available
        if not azure_blob_manager.is_available:
             # logger.warning(f"[AZURE_CHECK] Azure Blob Manager is not initialized. Skipping Azure check for {product_type}")
             # Fail silently to catch block which triggers local fallback
             raise Exception("Azure Blob Manager not initialized or available")

        # Try to get metadata from Azure Blob
        blob_client = azure_blob_manager.get_blob_client(metadata_path)
        metadata_bytes = blob_client.download_blob().readall()

        # Parse metadata
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        # VALIDATION: Verify the actual image file exists, not just metadata
        image_blob_path = metadata.get('image_blob_path')
        if image_blob_path:
            try:
                image_blob_client = azure_blob_manager.get_blob_client(image_blob_path)
                if not image_blob_client.exists():
                    logger.warning(f"[AZURE_CHECK] Metadata exists but image file missing for: {product_type} at {image_blob_path}")
                    raise ResourceNotFoundError(f"Image file not found: {image_blob_path}")
            except ResourceNotFoundError:
                # Metadata exists but image doesn't - treat as cache miss
                logger.warning(f"[AZURE_CHECK] Image validation failed - metadata without image for: {product_type}")
                raise

        logger.info(f"[AZURE_CHECK] ✓ Found cached generic image in Azure Blob for: {product_type}")
        if issue_debug:
            issue_debug.cache_hit("azure_blob", normalized_type)

        return {
            'azure_blob_path': image_blob_path,
            'product_type': metadata.get('product_type'),
            'source': metadata.get('source'),
            'content_type': metadata.get('content_type', 'image/png'),
            'file_size': metadata.get('file_size', 0),
            'generation_method': metadata.get('generation_method', 'llm'),
            'cached': True,
            'storage_location': 'azure_blob'
        }

    except ResourceNotFoundError:
        logger.info(f"[AZURE_CHECK] No cached generic image in Azure Blob for: {product_type} (normalized: {normalized_type})")
        # Try local fallback here if not found in Azure
        return _get_local_generic_image(product_type, normalized_type)
        
    except Exception as e:
        logger.warning(f"[AZURE_CHECK] Failed to retrieve generic image from Azure Blob for '{product_type}': {e}")
        # Try local fallback on error
        return _get_local_generic_image(product_type, normalized_type)


def _get_local_generic_image(product_type: str, normalized_type: str = None) -> Optional[Dict[str, Any]]:
    """Helper to check local disk for generic images."""
    try:
        import json
        if not normalized_type:
             normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
             
        local_dir = os.path.join(os.getcwd(), 'static', 'images', 'generic_images')
        metadata_path = os.path.join(local_dir, f"{normalized_type}.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"[LOCAL_CHECK] ✓ Found cached generic image locally for: {product_type}")
            return {
                'azure_blob_path': f"{normalized_type}.png", # Simplified path for local
                'product_type': metadata.get('product_type'),
                'source': metadata.get('source'),
                'content_type': metadata.get('content_type', 'image/png'),
                'file_size': metadata.get('file_size', 0),
                'generation_method': metadata.get('generation_method', 'llm'),
                'cached': True,
                'storage_location': 'local'
            }
    except Exception as e:
        logger.warning(f"[LOCAL_CHECK] Failed local lookup for {product_type}: {e}")
    
    return None


def cache_generic_image_to_azure(product_type: str, image_data: Dict[str, Any]) -> bool:
    """
    Cache LLM-generated image to BOTH MongoDB and Azure Blob Storage (PHASE 3 Hybrid).

    CACHING STRATEGY:
    1. Upload image file to Azure Blob Storage (actual bytes)
    2. Cache metadata + SAS URL to MongoDB (fast lookups with TTL)
    3. MongoDB cache auto-expires after 90 days (TTL index)

    Args:
        product_type: Product type name
        image_data: Generated image data containing image_bytes, content_type, etc.

    Returns:
        bool: True if successfully cached to at least one storage
    """
    try:
        from azure_blob_config import Collections, azure_blob_manager
        from azure.storage.blob import ContentSettings
        import json

        logger.info(f"[CACHE_HYBRID] Caching generic image to MongoDB + Azure Blob for: {product_type}")

        # FIX: Check if Azure Blob Storage is available before attempting upload
        if azure_blob_manager is None:
            logger.warning(
                f"[CACHE_AZURE] Azure Blob Storage not configured - "
                f"falling back to LOCAL storage for {product_type}"
            )
            return _cache_generic_image_locally(product_type, image_data)

        # Get image bytes
        image_bytes = image_data.get('image_bytes')
        if not image_bytes:
            logger.warning(f"[CACHE_AZURE] No image bytes provided for caching: {product_type}")
            return False

        content_type = image_data.get('content_type', 'image/png')
        file_size = image_data.get('file_size', len(image_bytes))

        # Normalize product type for path
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

        # Determine file extension
        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'

        # Upload image blob
        image_blob_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.{file_extension}"

        # FIX: Additional safety check before calling get_blob_client
        if not hasattr(azure_blob_manager, 'get_blob_client'):
            logger.warning(
                f"[CACHE_AZURE] Azure Blob Manager missing get_blob_client method - "
                f"skipping cache for {product_type}"
            )
            return False

        image_blob_client = azure_blob_manager.get_blob_client(image_blob_path)
        image_blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata={
                'product_type': product_type,
                'normalized_type': normalized_type,
                'generation_method': 'llm'
            }
        )

        logger.info(f"[CACHE_AZURE] ✓ Stored image in Azure Blob: {image_blob_path}")

        # Upload metadata JSON
        metadata_doc = {
            'product_type': product_type,
            'product_type_normalized': normalized_type,
            'image_blob_path': image_blob_path,
            'source': image_data.get('source', 'gemini_imagen'),
            'content_type': content_type,
            'file_size': file_size,
            'generation_method': 'llm',
            'generation_prompt': image_data.get('prompt', ''),
            'created_at': datetime.utcnow().isoformat()
        }

        metadata_blob_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.json"
        metadata_json = json.dumps(metadata_doc, indent=2)

        try:
            # FIX: Wrap metadata upload in try-except to handle Azure client errors
            metadata_blob_client = azure_blob_manager.get_blob_client(metadata_blob_path)
            metadata_blob_client.upload_blob(
                metadata_json,
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json'),
                metadata={
                    'product_type': product_type,
                    'normalized_type': normalized_type,
                    'metadata_file': 'true'
                }
            )
            logger.info(f"[CACHE_AZURE] ✓ Stored metadata in Azure Blob: {metadata_blob_path}")
        except Exception as metadata_error:
            logger.warning(
                f"[CACHE_AZURE] Failed to store metadata for {product_type}: {metadata_error} "
                f"(image is still cached)"
            )
            # Return True anyway since image was cached successfully

        # PHASE 3: Also cache to MongoDB for fast lookups (<50ms vs 1-2s Azure Blob)
        try:
            from services.image_service import image_service
            if image_service:
                # Generate SAS URL for the image (24 hours expiry for display)
                from core.sas_utils import generate_sas_url
                blob_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME', '')}.blob.core.windows.net/{image_blob_path}"
                image_url_with_sas = generate_sas_url(blob_url, expiry_hours=24)

                # Cache to MongoDB with metadata
                mongo_result = image_service.cache_generic_image(
                    product_type=product_type,
                    image_url=image_url_with_sas,
                    source=image_data.get('source', 'gemini_imagen'),
                    metadata={
                        'azure_blob_path': image_blob_path,
                        'content_type': content_type,
                        'file_size': file_size,
                        'generation_method': 'llm',
                        'generation_prompt': image_data.get('prompt', ''),
                        'normalized_type': normalized_type
                    }
                )

                if mongo_result.get('success'):
                    logger.info(f"[CACHE_HYBRID] ✓ Also cached to MongoDB (cache_id: {mongo_result.get('cache_id')})")
                else:
                    logger.warning(f"[CACHE_HYBRID] MongoDB caching failed (Azure cache still OK): {mongo_result.get('error')}")
        except ImportError:
            logger.debug(f"[CACHE_HYBRID] image_service not available, MongoDB caching skipped")
        except Exception as mongo_error:
            logger.warning(f"[CACHE_HYBRID] MongoDB caching error (Azure cache still OK): {mongo_error}")

        return True

    except Exception as e:
        logger.error(f"[CACHE_AZURE] Failed to cache generic image to Azure Blob: {e}")
        logger.exception(e)
        logger.info(f"[CACHE_AZURE] Attempting local fallback due to Azure failure...")
        return _cache_generic_image_locally(product_type, image_data)


def _cache_generic_image_locally(product_type: str, image_data: Dict[str, Any]) -> bool:
    """Cache image to local disk (fallback)."""
    try:
        import json
        
        # Ensure directory exists
        local_dir = os.path.join(os.getcwd(), 'static', 'images', 'generic_images')
        os.makedirs(local_dir, exist_ok=True)
        
        image_bytes = image_data.get('image_bytes')
        if not image_bytes:
            return False

        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # Save Image
        image_path = os.path.join(local_dir, f"{normalized_type}.png")
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
            
        # Save Metadata
        metadata = {
            'product_type': product_type,
            'product_type_normalized': normalized_type,
            'image_filename': f"{normalized_type}.png",
            'source': image_data.get('source', 'gemini_imagen'),
            'content_type': 'image/png',
            'file_size': len(image_bytes),
            'generation_method': 'llm',
            'generation_prompt': image_data.get('prompt', ''),
            'created_at': datetime.utcnow().isoformat()
        }
        
        with open(os.path.join(local_dir, f"{normalized_type}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"[CACHE_LOCAL] ✓ Saved to local disk: {image_path}")
        return True
        
    except Exception as e:
        logger.error(f"[CACHE_LOCAL] Failed to save locally: {e}")
        return False


# =============================================================================
# PARALLEL BATCH IMAGE FETCHING
# =============================================================================
# Optimizes multiple image requests by:
# 1. Checking Azure cache for ALL product types IN PARALLEL
# 2. For cache misses, queuing LLM generation (sequential to respect rate limits)
# 3. Returning results as they become available

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_generic_images_batch(product_types: list, max_parallel_cache_checks: int = 10) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch generic images for multiple product types IN PARALLEL with MongoDB + Azure Blob hybrid caching (PHASE 3).

    PARALLELIZATION STRATEGY:
    - Phase 1: Check MongoDB cache for ALL types simultaneously (very fast <50ms per check, parallel)
    - Phase 2: Check Azure Blob cache for misses (fast, parallel)
    - Phase 3: For remaining misses, generate with LLM (slower, sequential to respect rate limits)

    PERFORMANCE:
    - MongoDB cache hits: <50ms each (parallel) = ~100ms total for 10 items
    - Azure Blob cache hits: 1-2s each (parallel) = ~2s total for 10 items
    - LLM generation: 3-5s each (sequential) = 30-50s for 10 items

    Args:
        product_types: List of product type strings to fetch images for
        max_parallel_cache_checks: Number of parallel cache checks (default 10, increased for MongoDB speed)

    Returns:
        Dict mapping product_type -> image result (or None if failed)
    """
    if not product_types:
        return {}

    results = {}
    cache_hits = []
    cache_misses = []

    logger.info(f"[BATCH_IMAGE] Starting parallel image fetch for {len(product_types)} product types...")
    start_time = time.time()

    # =========================================================================
    # PHASE 1: PARALLEL MONGODB + AZURE BLOB CACHE CHECKS (PHASE 3)
    # =========================================================================
    def check_hybrid_cache(product_type: str) -> tuple:
        """Check MongoDB first, then Azure Blob cache (designed for parallel execution)."""
        try:
            # PHASE 3: Check MongoDB cache FIRST (<50ms)
            try:
                from services.image_service import image_service
                if image_service:
                    mongo_image = image_service.get_generic_image(product_type)
                    if mongo_image:
                        logger.debug(f"[BATCH_IMAGE] MongoDB cache HIT for '{product_type}'")
                        return (product_type, {
                            'url': mongo_image.get('image_url', ''),
                            'product_type': product_type,
                            'source': mongo_image.get('source', 'gemini_imagen'),
                            'cached': True,
                            'cache_source': 'mongodb',
                            'generation_method': mongo_image.get('generation_method', 'llm')
                        })
            except:
                pass  # Fall through to Azure Blob

            # Check Azure Blob cache (fallback, 1-2s)
            azure_image = get_generic_image_from_azure(product_type)
            if azure_image:
                backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
                logger.debug(f"[BATCH_IMAGE] Azure Blob cache HIT for '{product_type}'")
                return (product_type, {
                    'url': backend_url,
                    'product_type': product_type,
                    'source': azure_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'cache_source': 'azure_blob',
                    'generation_method': azure_image.get('generation_method', 'llm')
                })

            # Try fallback types (same logic as fetch_generic_product_image)
            fallback_result = _try_fallback_cache_lookup(product_type)
            if fallback_result:
                return (product_type, fallback_result)

            return (product_type, None)  # Cache miss

        except Exception as e:
            logger.error(f"[BATCH_IMAGE] Error checking cache for '{product_type}': {e}")
            return (product_type, None)
    
    # Execute parallel cache checks (MongoDB + Azure Blob)
    logger.info(f"[BATCH_IMAGE] Phase 1: Checking MongoDB + Azure Blob cache for {len(product_types)} types in parallel...")

    with ThreadPoolExecutor(max_workers=max_parallel_cache_checks) as executor:
        future_to_type = {executor.submit(check_hybrid_cache, pt): pt for pt in product_types}
        
        for future in as_completed(future_to_type):
            product_type = future_to_type[future]
            try:
                pt, result = future.result()
                if result:
                    results[pt] = result
                    cache_hits.append(pt)
                    logger.info(f"[BATCH_IMAGE] ✓ Cache HIT for '{pt}'")
                else:
                    cache_misses.append(pt)
                    logger.info(f"[BATCH_IMAGE] ✗ Cache MISS for '{pt}'")
            except Exception as exc:
                cache_misses.append(product_type)
                logger.error(f"[BATCH_IMAGE] Exception for '{product_type}': {exc}")
    
    phase1_time = time.time() - start_time
    logger.info(f"[BATCH_IMAGE] Phase 1 complete: {len(cache_hits)} hits, {len(cache_misses)} misses in {phase1_time:.2f}s")
    
    # =========================================================================
    # PHASE 2: LLM GENERATION FOR CACHE MISSES (Sequential to respect rate limits)
    # =========================================================================
    if cache_misses:
        logger.info(f"[BATCH_IMAGE] Phase 2: Generating {len(cache_misses)} images with LLM...")
        
        # Process LLM generation sequentially (rate limit aware)
        for i, product_type in enumerate(cache_misses):
            try:
                logger.info(f"[BATCH_IMAGE] Generating image {i+1}/{len(cache_misses)}: '{product_type}'")
                
                # Use the existing fetch function which handles deduplication and rate limits
                result = fetch_generic_product_image(product_type)
                results[product_type] = result
                
                if result:
                    logger.info(f"[BATCH_IMAGE] ✓ Generated image for '{product_type}'")
                else:
                    logger.warning(f"[BATCH_IMAGE] ✗ Failed to generate image for '{product_type}'")
                    
            except Exception as e:
                logger.error(f"[BATCH_IMAGE] Error generating for '{product_type}': {e}")
                results[product_type] = None
    
    total_time = time.time() - start_time
    logger.info(f"[BATCH_IMAGE] Batch complete: {len(results)} images in {total_time:.2f}s")
    
    return results


def _try_fallback_cache_lookup(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Try fallback cache lookups for a product type (helper for parallel batch processing).
    
    This is a lightweight version that only checks cache, no LLM generation.
    """
    import re
    
    # Common prefixes to strip
    COMMON_PREFIXES = [
        # General modifiers (safe to strip - don't change visual appearance)
        "Process ", "Industrial ", "Digital ", "Smart ", "Advanced ", 
        "Precision ", "High Performance ", "Multi ", "Dual ", "Single ",
        "Compact ", "Integrated ", "Electronic ", "Intelligent ", "Programmable ",
        "Wireless ", "Remote ", "Field ", "Panel ", "Portable ", "Handheld ",
        "Ex-rated ", "ATEX ", "Hazardous Area ", "Explosion Proof ",
        # Communication protocol prefixes (don't change physical appearance)
        "HART ", "Foundation Fieldbus ", "Profibus ", "Modbus ", "4-20mA ",
        # Material/application prefixes (minor visual differences)
        "Stainless Steel ", "SS ", "316L ", "Hastelloy ", "Titanium ",
        "Sanitary ", "Hygienic ", "Food Grade ", "Pharmaceutical ",
        # Size/range prefixes
        "High Pressure ", "Low Pressure ", "High Temperature ", "Low Temperature ",
        "Wide Range ", "Narrow Range ", "High Accuracy ",
        # NOTE: DO NOT strip measurement-type prefixes (Coriolis, Radar, Magnetic, Ultrasonic, etc.)
        # or sensor-type prefixes (RTD, Thermocouple, etc.)
        # These distinguish fundamentally different products that look very different physically.
    ]
    
    fallback_types = []
    product_upper = product_type.strip()
    
    for prefix in COMMON_PREFIXES:
        if product_upper.lower().startswith(prefix.lower()):
            fallback_type = product_upper[len(prefix):].strip()
            if fallback_type and fallback_type != product_type:
                fallback_types.append(fallback_type)
    
    # Remove parenthetical content
    base_type = re.sub(r'\s*\([^)]*\)\s*', '', product_type).strip()
    if base_type and base_type != product_type and base_type not in fallback_types:
        fallback_types.append(base_type)
    
    # Extract base instrument type
    words = product_type.strip().split()
    if len(words) >= 2:
        last_word = words[-1]
        if last_word not in fallback_types and last_word != product_type:
            fallback_types.append(last_word)
        last_two = " ".join(words[-2:])
        if last_two not in fallback_types and last_two != product_type:
            fallback_types.append(last_two)
    
    # Try each fallback
    for fallback in fallback_types:
        azure_image = get_generic_image_from_azure(fallback)
        if azure_image:
            logger.info(f"[BATCH_IMAGE] ✓ Fallback cache hit: '{fallback}' for '{product_type}'")
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
            return {
                'url': backend_url,
                'product_type': product_type,
                'source': azure_image.get('source', 'gemini_imagen'),
                'cached': True,
                'generation_method': azure_image.get('generation_method', 'llm'),
                'fallback_used': fallback
            }
    
    # Category-based fallback (last resort)
    CATEGORY_MAPPINGS = {
        # Instruments
        'transmitter': ['Transmitter', 'Pressure Transmitter', 'Temperature Transmitter'],
        'sensor': ['Sensor', 'Temperature Sensor', 'Pressure Sensor'],
        'valve': ['Valve', 'Control Valve', 'Globe Valve', 'Ball Valve'],
        'actuator': ['Actuator', 'Pneumatic Actuator'],
        'meter': ['Meter', 'Flow Meter'],
        'gauge': ['Gauge', 'Pressure Gauge'],
        'cable': ['Cable', 'Instrument Cable', 'Cable Gland'],
        'fitting': ['Fitting', 'Pipe Fitting'],
        'bracket': ['Bracket', 'Mounting Bracket'],
        'junction': ['Junction Box'],
        'switch': ['Switch', 'Pressure Switch'],
        'motor': ['Motor', 'Electric Motor'],
        'drive': ['Drive', 'Variable Frequency Drive'],
        'pump': ['Pump'],
        'controller': ['Controller', 'Digital Valve Controller'],
        'analyzer': ['Analyzer'],
        # Accessories - Temperature
        'thermowell': ['Thermowell', 'Protective Sleeve', 'Temperature Well'],
        'terminal': ['Terminal Head', 'Connection Head', 'Terminal Box'],
        # Accessories - Pressure
        'manifold': ['Manifold', '3-Valve Manifold', '5-Valve Manifold', 'Valve Manifold'],
        'impulse': ['Impulse Line', 'Impulse Tubing'],
        'snubber': ['Snubber', 'Pulsation Dampener'],
        # Accessories - Flow
        'gasket': ['Gasket', 'Flange Gasket', 'Gasket Kit', 'Sealing Gasket'],
        'bolt': ['Bolt', 'Bolts and Nuts', 'Stud Bolt', 'Bolt Kit', 'Fastener'],
        'flange': ['Flange', 'Mounting Flange', 'Process Flange', 'Flange Adapter'],
        'straightener': ['Flow Straightener', 'Flow Conditioner'],
        # Accessories - General
        'gland': ['Cable Gland', 'Gland', 'Connector Gland'],
        'regulator': ['Air Filter Regulator', 'Filter Regulator', 'Pressure Regulator'],
        'sunshield': ['Sunshield', 'Sun Shield', 'Weather Shield'],
        'enclosure': ['Enclosure', 'Junction Box', 'Terminal Enclosure'],
        'positioner': ['Positioner', 'Valve Positioner', 'Digital Positioner'],
    }
    
    product_lower = product_type.lower()
    for category, generic_types in CATEGORY_MAPPINGS.items():
        if category in product_lower:
            for generic_type in generic_types:
                if generic_type.lower() != product_lower and generic_type not in fallback_types:
                    azure_image = get_generic_image_from_azure(generic_type)
                    if azure_image:
                        logger.info(f"[BATCH_IMAGE] ✓ Category fallback: '{generic_type}' for '{product_type}'")
                        backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
                        return {
                            'url': backend_url,
                            'product_type': product_type,
                            'source': azure_image.get('source', 'gemini_imagen'),
                            'cached': True,
                            'generation_method': azure_image.get('generation_method', 'llm'),
                            'fallback_used': generic_type,
                            'fallback_type': 'category'
                        }
    
    return None


def fetch_generic_product_image(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generic product type image with MongoDB + Azure Blob hybrid caching and LLM generation.

    FLOW (PHASE 3 - Hybrid Storage):
    1. Check MongoDB cache first (fastest - <50ms indexed query)
    2. If not in MongoDB, check Azure Blob Storage (slower - 1-2s blob scan)
    3. If not found, try fallback base product types (e.g., "Process Flow Transmitter" -> "Flow Transmitter")
    4. Check if generation is already in-progress for this product type (deduplication)
    5. If not, generate image using Gemini Imagen 4.0 LLM
    6. Cache to BOTH MongoDB (metadata + SAS URL) AND Azure Blob (actual file)

    PERFORMANCE:
    - Cache hit (MongoDB): <50ms
    - Cache hit (Azure Blob): 1-2s
    - Cache miss (LLM generation): 3-5s

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image URL and metadata, or None if generation failed
    """
    logger.info(f"[FETCH] Fetching generic image for product type: {product_type}")

    # PHASE 3: Check MongoDB cache FIRST (fastest path <50ms)
    try:
        from services.image_service import image_service
        if image_service:
            mongo_image = image_service.get_generic_image(product_type)
            if mongo_image:
                logger.info(f"[FETCH] ✓ MongoDB cache HIT for '{product_type}' (<50ms)")
                return {
                    'url': mongo_image.get('image_url', ''),
                    'product_type': product_type,
                    'source': mongo_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'cache_source': 'mongodb',
                    'generation_method': mongo_image.get('generation_method', 'llm'),
                    'cache_id': mongo_image.get('cache_id')
                }
            else:
                logger.info(f"[FETCH] MongoDB cache MISS for '{product_type}'")
    except ImportError:
        logger.debug(f"[FETCH] image_service not available, skipping MongoDB cache")
    except Exception as e:
        logger.warning(f"[FETCH] MongoDB cache error: {e}, falling back to Azure Blob")

    # Normalize product type for deduplication
    normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    # FIX: Check global generation cache first to prevent duplicate generation
    # Using BoundedCache - TTL is handled automatically
    with _generation_lock:
        last_gen = _global_generation_cache.get(normalized_type)
        if last_gen is not None:
                logger.info(f"[FETCH] Global cache: '{product_type}' generated recently, checking Azure...")
                # Already generated recently, try Azure cache
                azure_image = get_generic_image_from_azure(product_type)
                if azure_image:
                    backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
                    return {
                        'url': backend_url,
                        'product_type': product_type,
                        'source': azure_image.get('source', 'gemini_imagen'),
                        'cached': True,
                        'generation_method': azure_image.get('generation_method', 'llm'),
                        'from_global_cache': True
                    }

    # Step 1: Check Azure Blob Storage for cached image (exact match)
    azure_image = get_generic_image_from_azure(product_type)
    if azure_image:
        logger.info(f"[FETCH] ✓ Using cached generic image from {azure_image.get('storage_location', 'Azure')} for '{product_type}'")
        
        # Determine URL based on storage location
        if azure_image.get('storage_location') == 'local':
            # Local URL
            filename = os.path.basename(azure_image.get('azure_blob_path', ''))
            backend_url = f"/api/images/generic_images/{filename}"
        else:
            # Azure URL (proxied via existing logic if needed, but here we stay consistent)
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

        return {
            'url': backend_url,
            'product_type': product_type,
            'source': azure_image.get('source', 'gemini_imagen'),
            'cached': True,
            'generation_method': azure_image.get('generation_method', 'llm')
        }

    # Step 1.5: Extract actual product type from patterns (common naming conventions)
    # Pattern 1: "X for Y" (accessories) - e.g., "Connection Head for Temperature Transmitter" → "Connection Head"
    # Pattern 2: "X with Y" (instruments with modifiers) - e.g., "Temperature Transmitter with RTD Sensor" → "Temperature Transmitter"
    # This MUST run before other fallbacks to prevent wrong category matches

    extracted_base_type = None
    extraction_pattern = None

    if ' for ' in product_type:
        # Pattern 1: "X for Y" - extract the accessory type (X)
        extracted_base_type = product_type.split(' for ')[0].strip()
        extraction_pattern = 'X for Y'
    elif ' with ' in product_type:
        # Pattern 2: "X with Y" - extract the base instrument type (X)
        # e.g., "Temperature Transmitter with RTD Sensor" → "Temperature Transmitter"
        # e.g., "Pressure Gauge with Diaphragm Seal" → "Pressure Gauge"
        extracted_base_type = product_type.split(' with ')[0].strip()
        extraction_pattern = 'X with Y'
    elif ' including ' in product_type:
        # Pattern 3: "X including Y" - extract the base type (X)
        extracted_base_type = product_type.split(' including ')[0].strip()
        extraction_pattern = 'X including Y'
    elif ' featuring ' in product_type:
        # Pattern 4: "X featuring Y" - extract the base type (X)
        extracted_base_type = product_type.split(' featuring ')[0].strip()
        extraction_pattern = 'X featuring Y'

    if extracted_base_type and extracted_base_type.lower() != product_type.lower():
        logger.info(f"[FETCH] Extracted base type from '{extraction_pattern}' pattern: '{extracted_base_type}' (from '{product_type}')")

        # Try exact match for extracted type first
        azure_image = get_generic_image_from_azure(extracted_base_type)
        if azure_image:
            logger.info(f"[FETCH] ✓ Found cached image for extracted base type: '{extracted_base_type}'")
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
            return {
                'url': backend_url,
                'product_type': product_type,
                'source': azure_image.get('source', 'gemini_imagen'),
                'cached': True,
                'generation_method': azure_image.get('generation_method', 'llm'),
                'extracted_from': product_type,
                'extracted_type': extracted_base_type,
                'extraction_pattern': extraction_pattern
            }

        # If not cached, generate image for the EXTRACTED type (not the full name)
        # This ensures we generate "Temperature Transmitter" not "Temperature Transmitter with RTD Sensor"
        logger.info(f"[FETCH] Will generate image for extracted type: '{extracted_base_type}' instead of full name")
        # Update product_type for subsequent generation to use the extracted type
        product_type = extracted_base_type
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    # Step 1.6: Try fallback base product types
    # Remove common prefixes to find a more generic cached image
    COMMON_PREFIXES = [
        # General modifiers
        "Process ", "Industrial ", "Digital ", "Smart ", "Advanced ", 
        "Precision ", "High Performance ", "Multi ", "Dual ", "Single ",
        "Compact ", "Integrated ", "Electronic ", "Intelligent ", "Programmable ",
        "Wireless ", "Remote ", "Field ", "Panel ", "Portable ", "Handheld ",
        "Ex-rated ", "ATEX ", "Hazardous Area ", "Explosion Proof ",
        # Sensor type prefixes (RTD, Thermocouple, etc.)
        "RTD ", "PT100 ", "PT1000 ", "Thermocouple ", "TC ", "K-Type ", "J-Type ",
        "HART ", "Foundation Fieldbus ", "Profibus ", "Modbus ", "4-20mA ",
        # Measurement type prefixes
        "Coriolis ", "Magnetic ", "Ultrasonic ", "Vortex ", "Differential ",
        "Absolute ", "Gauge ", "Sealed ", "Diaphragm ", "Capacitive ",
        # Material/application prefixes
        "Stainless Steel ", "SS ", "316L ", "Hastelloy ", "Titanium ",
        "Sanitary ", "Hygienic ", "Food Grade ", "Pharmaceutical ",
        # Size/range prefixes
        "High Pressure ", "Low Pressure ", "High Temperature ", "Low Temperature ",
        "Wide Range ", "Narrow Range ", "High Accuracy ",
    ]
    
    fallback_types = []
    product_upper = product_type.strip()
    
    for prefix in COMMON_PREFIXES:
        if product_upper.lower().startswith(prefix.lower()):
            fallback_type = product_upper[len(prefix):].strip()
            if fallback_type and fallback_type != product_type:
                fallback_types.append(fallback_type)
    
    # Also try without parenthetical content: "Flow Transmitter (Redundant)" -> "Flow Transmitter"
    import re
    base_type = re.sub(r'\s*\([^)]*\)\s*', '', product_type).strip()
    if base_type and base_type != product_type and base_type not in fallback_types:
        fallback_types.append(base_type)
    
    # Try to extract the base instrument type (last 1-2 significant words)
    # e.g., "RTD Temperature Transmitter" -> "Transmitter", "Temperature Transmitter"
    words = product_type.strip().split()
    if len(words) >= 2:
        # Try last word (e.g., "Transmitter", "Valve", "Meter")
        last_word = words[-1]
        if last_word not in fallback_types and last_word != product_type:
            fallback_types.append(last_word)
        
        # Try last two words (e.g., "Temperature Transmitter", "Flow Meter")
        last_two = " ".join(words[-2:])
        if last_two not in fallback_types and last_two != product_type:
            fallback_types.append(last_two)
    
    # Log fallback attempts
    if fallback_types:
        logger.info(f"[FETCH] Trying {len(fallback_types)} fallback types for '{product_type}': {fallback_types}")
    
    # Try each fallback type
    for fallback in fallback_types:
        logger.info(f"[FETCH] Trying fallback: '{fallback}'")
        azure_image = get_generic_image_from_azure(fallback)
        if azure_image:
            logger.info(f"[FETCH] ✓ Using cached generic image from Azure Blob for '{fallback}' (fallback from '{product_type}')")
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

            return {
                'url': backend_url,
                'product_type': product_type,  # Keep original product type in response
                'source': azure_image.get('source', 'gemini_imagen'),
                'cached': True,
                'generation_method': azure_image.get('generation_method', 'llm'),
                'fallback_used': fallback  # Indicate which fallback was used
            }

    # Step 1.6: Category-based fallback (last resort before LLM)
    # Map product types to generic categories that are likely cached
    CATEGORY_MAPPINGS = {
        # Transmitters
        'transmitter': ['Transmitter', 'Pressure Transmitter', 'Temperature Transmitter', 'Flow Transmitter'],
        'sensor': ['Sensor', 'Temperature Sensor', 'Pressure Sensor'],
        'transducer': ['Transducer', 'Pressure Transducer'],
        # Valves
        'valve': ['Valve', 'Control Valve', 'Ball Valve', 'Gate Valve'],
        'actuator': ['Actuator', 'Pneumatic Actuator', 'Electric Actuator'],
        # Meters
        'meter': ['Meter', 'Flow Meter', 'Level Meter'],
        'gauge': ['Gauge', 'Pressure Gauge', 'Temperature Gauge'],
        # Accessories - Common instrument accessories
        'cable': ['Cable', 'Instrument Cable', 'Power Cable'],
        'fitting': ['Fitting', 'Pipe Fitting', 'Compression Fitting'],
        'bracket': ['Bracket', 'Mounting Bracket', 'Support Bracket'],
        'junction': ['Junction Box', 'Terminal Box'],
        'enclosure': ['Enclosure', 'Instrument Enclosure'],
        # Accessories - Specific types (common in industrial instrumentation)
        'connection head': ['Connection Head', 'Terminal Head', 'Instrument Head'],
        'thermowell': ['Thermowell', 'Protection Tube', 'Sensor Pocket', 'Thermal Well'],
        'cable gland': ['Cable Gland', 'Cable Entry', 'Cord Grip'],
        'process connection': ['Process Connection', 'Flange Connection', 'Threaded Connection'],
        'manifold': ['Manifold', 'Valve Manifold', 'Instrument Manifold'],
        'impulse line': ['Impulse Line', 'Impulse Tubing', 'Sensing Line'],
        'heat trace': ['Heat Trace', 'Heat Tracing', 'Trace Heating'],
        'insulation': ['Insulation', 'Thermal Insulation', 'Pipe Insulation'],
        'sunshade': ['Sunshade', 'Weather Protection', 'Sun Shield'],
        'calibration': ['Calibration Kit', 'Calibration Equipment'],
        'seal': ['Diaphragm Seal', 'Remote Seal', 'Chemical Seal'],
        'orifice': ['Orifice Plate', 'Flow Element', 'Orifice Flange'],
        'thermocouple': ['Thermocouple', 'TC Assembly', 'Temperature Probe'],
        'rtd': ['RTD', 'RTD Assembly', 'Resistance Temperature Detector'],
        # Electrical
        'switch': ['Switch', 'Pressure Switch', 'Level Switch'],
        'relay': ['Relay'],
        'barrier': ['Barrier', 'Intrinsic Safety Barrier'],
        # Motors & Drives
        'motor': ['Motor', 'Electric Motor'],
        'drive': ['Drive', 'Variable Frequency Drive'],
        'pump': ['Pump'],
        # General
        'indicator': ['Indicator', 'Level Indicator'],
        'controller': ['Controller', 'PID Controller'],
        'recorder': ['Recorder', 'Chart Recorder'],
        'analyzer': ['Analyzer', 'Gas Analyzer'],
    }
    
    # Find category match
    product_lower = product_type.lower()
    category_fallbacks = []
    
    for category, generic_types in CATEGORY_MAPPINGS.items():
        if category in product_lower:
            category_fallbacks.extend(generic_types)
    
    # Remove duplicates and already-tried types
    category_fallbacks = [c for c in category_fallbacks if c not in fallback_types and c.lower() != product_lower]
    
    if category_fallbacks:
        logger.info(f"[FETCH] Trying {len(category_fallbacks)} category fallbacks for '{product_type}': {category_fallbacks}")
        
        for category_fallback in category_fallbacks:
            azure_image = get_generic_image_from_azure(category_fallback)
            if azure_image:
                logger.info(f"[FETCH] ✓ Category fallback found: '{category_fallback}' for '{product_type}'")
                backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

                return {
                    'url': backend_url,
                    'product_type': product_type,
                    'source': azure_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'generation_method': azure_image.get('generation_method', 'llm'),
                    'fallback_used': category_fallback,
                    'fallback_type': 'category'
                }


    # Step 2: Check if generation is already in-progress for this product type
    # Using BoundedCache for automatic cleanup of stale pending requests
    existing_event = _pending_requests_cache.get(normalized_type)
    if existing_event is not None:
        # Another request is already generating this image - wait for it
        logger.info(f"[FETCH] Generation already in-progress for '{product_type}', waiting...")
        event = existing_event
    else:
        # We are the first - mark as in-progress
        event = threading.Event()
        _pending_requests_cache.set(normalized_type, event)
        _pending_results_cache.set(normalized_type, None)

    # If we're waiting for another request (event already existed)
    current_event = _pending_requests_cache.get(normalized_type)
    if event.is_set() or (current_event is not None and current_event != event):
        # Wait for the event with timeout
        wait_result = event.wait(timeout=120)  # Wait up to 2 minutes

        if wait_result:
            # Check cache again - the other request should have populated it
            azure_image = get_generic_image_from_azure(product_type)
            if azure_image:
                logger.info(f"[FETCH] ✓ Got result from parallel request for '{product_type}'")
                backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

                return {
                    'url': backend_url,
                    'product_type': product_type,
                    'source': azure_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'generation_method': azure_image.get('generation_method', 'llm')
                }

        logger.warning(f"[FETCH] Parallel request wait timed out for '{product_type}'")
        return None

    # Step 3: We are the primary generator - generate image using LLM (NO rate limiting)
    logger.info(f"[FETCH] Cache miss for '{product_type}', generating image with Gemini Imagen 4.0...")

    try:
        # FIX: Record generation attempt in global cache (BoundedCache handles TTL automatically)
        with _generation_lock:
            _global_generation_cache.set(normalized_type, time.time())

        generated_image_data = _generate_image_with_llm(product_type)

        if generated_image_data:
            # Step 4: Cache to Azure Blob Storage ONLY
            azure_cache_success = cache_generic_image_to_azure(product_type, generated_image_data)

            if azure_cache_success:
                logger.info(f"[FETCH] ✓ Successfully generated and cached LLM image for '{product_type}'")
                
                # FIX 4: Mark success in tracker
                _failure_tracker.mark_success(product_type)

                # Retrieve the cached image to get the URL
                azure_image = get_generic_image_from_azure(product_type)
                if azure_image:
                    backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

                    # Notify waiting requests and cleanup (BoundedCache handles via delete)
                    pending_event = _pending_requests_cache.get(normalized_type)
                    if pending_event is not None:
                        pending_event.set()
                        _pending_requests_cache.delete(normalized_type)
                    _pending_results_cache.delete(normalized_type)

                    return {
                        'url': backend_url,
                        'product_type': product_type,
                        'source': 'gemini_imagen',
                        'cached': False,  # Newly generated
                        'generation_method': 'llm'
                    }

            logger.warning(f"[FETCH] LLM image generated but caching failed for '{product_type}'")
            
            # FIX: Return base64 image directly when Azure caching fails
            # This ensures images are displayed even without Azure Blob Storage
            import base64
            image_bytes = generated_image_data.get('image_bytes')
            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                content_type = generated_image_data.get('content_type', 'image/png')
                logger.info(f"[FETCH] ✓ FIX: Returning base64 image for '{product_type}' ({len(image_bytes)} bytes)")
                
                # Notify waiting requests and cleanup (BoundedCache handles via delete)
                pending_event = _pending_requests_cache.get(normalized_type)
                if pending_event is not None:
                    pending_event.set()
                    _pending_requests_cache.delete(normalized_type)
                _pending_results_cache.delete(normalized_type)

                return {
                    'url': f"data:{content_type};base64,{base64_image}",
                    'product_type': product_type,
                    'source': 'gemini_imagen',
                    'cached': False,
                    'generation_method': 'llm',
                    'storage_fallback': 'base64'
                }
        else:
            logger.error(f"[FETCH] LLM image generation failed for '{product_type}'")
            # FIX 4: Record failure
            _failure_tracker.record_failure(product_type, "LLM generation returned None")

    finally:
        # Clean up pending request tracking (BoundedCache handles via delete)
        pending_event = _pending_requests_cache.get(normalized_type)
        if pending_event is not None:
            pending_event.set()  # Signal completion (even on failure)
            _pending_requests_cache.delete(normalized_type)
        _pending_results_cache.delete(normalized_type)

    # Complete failure - no image available
    logger.error(f"[FETCH] Failed to retrieve/generate image for '{product_type}'")
    return None


def fetch_generic_product_image_fast(product_type: str) -> Dict[str, Any]:
    """
    Fetch generic product image with FAST-FAIL behavior.
    
    Same as fetch_generic_product_image, but returns immediately if:
    - LLM generation is needed AND rate-limited
    
    Returns a special response indicating to use a placeholder.
    
    Returns:
        {
            'success': True/False,
            'url': '/api/images/...' or None,
            'use_placeholder': True if should use placeholder,
            'reason': 'cached' | 'generated' | 'rate_limited' | 'failed'
        }
    """
    logger.info(f"[FETCH_FAST] Fast-fetching image for: {product_type}")
    
    # First, try the normal cache-based lookup (all fallbacks)
    # This part is fast - only checks Azure Cache
    normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    
    # Step 1: Exact match
    azure_image = get_generic_image_from_azure(product_type)
    if azure_image:
        logger.info(f"[FETCH_FAST] ✓ Cache hit for '{product_type}'")
        return {
            'success': True,
            'url': f"images/{azure_image.get('azure_blob_path', '')}",
            'product_type': product_type,
            'source': 'cache',
            'use_placeholder': False,
            'reason': 'cached'
        }
    
    # Step 2: Try fallback cache lookup (fast - no LLM)
    fallback_result = _try_fallback_cache_lookup(product_type)
    if fallback_result:
        logger.info(f"[FETCH_FAST] ✓ Fallback cache hit for '{product_type}'")
        fallback_result['success'] = True
        fallback_result['use_placeholder'] = False
        fallback_result['reason'] = 'cached_fallback'
        return fallback_result
    
    # Step 3: No cache hit - try fast LLM generation (no retry on 429)
    logger.info(f"[FETCH_FAST] Cache miss for '{product_type}', trying fast LLM generation...")
    
    generated_image_data = _generate_image_with_llm_fast(product_type)
    
    if generated_image_data:
        # Cache it
        cache_success = cache_generic_image_to_azure(product_type, generated_image_data)
        
        if cache_success:
            _failure_tracker.mark_success(product_type)  # FIX 4
            azure_image = get_generic_image_from_azure(product_type)
            if azure_image:
                logger.info(f"[FETCH_FAST] ✓ Generated and cached image for '{product_type}'")
                return {
                    'success': True,
                    'url': f"images/{azure_image.get('azure_blob_path', '')}",
                    'product_type': product_type,
                    'source': 'gemini_imagen',
                    'use_placeholder': False,
                    'reason': 'generated'
                }
        else:
            # FIX: Return base64 image when Azure caching fails
            import base64
            image_bytes = generated_image_data.get('image_bytes')
            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                content_type = generated_image_data.get('content_type', 'image/png')
                logger.info(f"[FETCH_FAST] ✓ FIX: Returning base64 image for '{product_type}'")
                return {
                    'success': True,
                    'url': f"data:{content_type};base64,{base64_image}",
                    'product_type': product_type,
                    'source': 'gemini_imagen',
                    'use_placeholder': False,
                    'reason': 'generated_base64'
                }
    else:
        # FIX 4
        _failure_tracker.record_failure(product_type, "Fast LLM generation returned None")
    
    # Step 4: LLM generation failed (likely rate-limited)
    logger.warning(f"[FETCH_FAST] ✗ Cannot generate image for '{product_type}' - use placeholder")
    return {
        'success': False,
        'url': None,
        'product_type': product_type,
        'source': None,
        'use_placeholder': True,
        'reason': 'rate_limited'
    }


# =============================================================================
# REGENERATION WITH RATE LIMITING (For UI-triggered retry)
# =============================================================================
# Prevents abuse while allowing UI to retry failed image generation

# Using BoundedCache for rate limiting with automatic cleanup
_regeneration_cooldowns: BoundedCache = get_or_create_cache(
    name="image_regen_cooldowns",
    max_size=200,
    ttl_seconds=3600  # 1 hour
)

_regeneration_counts: BoundedCache = get_or_create_cache(
    name="image_regen_counts",
    max_size=200,
    ttl_seconds=3600  # 1 hour - counts reset after 1 hour
)

_regeneration_lock = threading.Lock()
_REGEN_COOLDOWN_SECONDS = 30  # Minimum 30 seconds between regeneration attempts
_REGEN_MAX_PER_HOUR = 3  # Maximum 3 regeneration attempts per product type per hour


def _check_regeneration_rate_limit(product_type: str) -> tuple:
    """
    Check if regeneration is allowed for this product type.

    Returns:
        (allowed: bool, reason: str, wait_seconds: int)
    """
    normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    current_time = time.time()

    with _regeneration_lock:
        # Check cooldown (BoundedCache handles TTL expiration automatically)
        last_regen_time = _regeneration_cooldowns.get(normalized_type)

        if last_regen_time is not None:
            time_since_last = current_time - last_regen_time

            if time_since_last < _REGEN_COOLDOWN_SECONDS:
                wait_time = int(_REGEN_COOLDOWN_SECONDS - time_since_last)
                return (False, f"Please wait {wait_time} seconds before retrying", wait_time)

        # Check hourly limit (TTL handles the hourly reset automatically)
        regen_count = _regeneration_counts.get(normalized_type)
        if regen_count is None:
            regen_count = 0

        if regen_count >= _REGEN_MAX_PER_HOUR:
            return (False, f"Maximum {_REGEN_MAX_PER_HOUR} regeneration attempts per hour reached", 0)

        return (True, "allowed", 0)


def _record_regeneration_attempt(product_type: str):
    """Record a regeneration attempt for rate limiting."""
    normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    current_time = time.time()

    with _regeneration_lock:
        _regeneration_cooldowns.set(normalized_type, current_time)
        current_count = _regeneration_counts.get(normalized_type)
        _regeneration_counts.set(normalized_type, (current_count or 0) + 1)


def regenerate_generic_image(product_type: str) -> Dict[str, Any]:
    """
    Force regenerate a generic product image using LLM.
    
    This function is called by the UI when initial image fetch failed
    and the user wants to retry generation.
    
    Includes rate limiting:
    - 30 second cooldown between attempts per product type
    - Maximum 3 attempts per hour per product type
    
    Args:
        product_type: Product type to generate image for
        
    Returns:
        {
            'success': True/False,
            'url': '/api/images/...' or None,
            'reason': 'regenerated' | 'rate_limited' | 'failed',
            'cached': False,
            'wait_seconds': int (if rate limited)
        }
    """
    logger.info(f"[REGENERATE] Regeneration request for: {product_type}")
    
    # Check rate limiting
    allowed, reason, wait_seconds = _check_regeneration_rate_limit(product_type)
    
    if not allowed:
        logger.warning(f"[REGENERATE] Rate limited for '{product_type}': {reason}")
        return {
            'success': False,
            'url': None,
            'reason': 'rate_limited',
            'message': reason,
            'wait_seconds': wait_seconds
        }
    
    # Record this attempt
    _record_regeneration_attempt(product_type)
    
    # Skip cache, directly generate with LLM
    logger.info(f"[REGENERATE] Generating image with LLM for '{product_type}'...")
    generated_image_data = _generate_image_with_llm(product_type)
    
    if generated_image_data:
        # Cache to Azure
        cache_success = cache_generic_image_to_azure(product_type, generated_image_data)
        
        if cache_success:
            azure_image = get_generic_image_from_azure(product_type)
            if azure_image:
                logger.info(f"[REGENERATE] ✓ Successfully regenerated and cached image for '{product_type}'")
                return {
                    'success': True,
                    'url': f"/api/images/{azure_image.get('azure_blob_path', '')}",
                    'product_type': product_type,
                    'source': 'gemini_imagen',
                    'reason': 'regenerated',
                    'cached': False
                }
        
        # Fallback to base64 if Azure cache fails
        import base64
        image_bytes = generated_image_data.get('image_bytes')
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            content_type = generated_image_data.get('content_type', 'image/png')
            logger.info(f"[REGENERATE] ✓ Returning base64 image for '{product_type}' (Azure cache failed)")
            return {
                'success': True,
                'url': f"data:{content_type};base64,{base64_image}",
                'product_type': product_type,
                'source': 'gemini_imagen',
                'reason': 'regenerated',
                'cached': False,
                'storage_fallback': 'base64'
            }
    
    logger.error(f"[REGENERATE] ✗ Failed to regenerate image for '{product_type}'")
    return {
        'success': False,
        'url': None,
        'product_type': product_type,
        'reason': 'failed',
        'message': 'Image generation failed. The LLM may be rate limited or unavailable.'
    }


# =============================================================================
# BACKGROUND RETRY WORKER
# =============================================================================

def retry_generation_worker():
    """
    Background worker that retries failed image generations.
    Runs periodically to process items in the failure tracker.
    Respects strict retry limits (max 2 attempts total).
    """
    import time
    logger.info("[RETRY_WORKER] Started background image generation retry worker")
    
    while True:
        try:
            # Check for pending retries
            pending = _failure_tracker.get_pending_retries()
            
            if pending:
                logger.info(f"[RETRY_WORKER] Processing {len(pending)} failed items...")
                
                for item in pending:
                    product_type = item['product_type']
                    retry_count = item.get('retry_count', 0)
                    
                    # Double check retry limit just in case (Tracker logic should handle this, but be safe)
                    if retry_count >= 1: # 0 = initial, 1 = first retry. If count is 1, this IS the retry. If it fails, count becomes 2.
                        # Do we retry here? Tracker marks 'permanently_failed' AFTER incrementing.
                        # If status is pending_retry, it means count is < max_retries. 
                        # Actually if count is 1, next increment makes it 2 and status 'permanently_failed'.
                        pass
                    
                    logger.info(f"[RETRY_WORKER] Retrying generation for '{product_type}' (Attempt {retry_count + 2}/2)...")
                    
                    # Generate (this handles deduplication via fetch function locks if needed, 
                    # but here we call _generate direct or fetch?)
                    # Best to call fetch_generic_product_image to reuse all logic including caching
                    # BUT fetch calls tracker.record_failure on error, which increments count. Perfect.
                    
                    result = fetch_generic_product_image(product_type)
                    
                    if result:
                        logger.info(f"[RETRY_WORKER] ✓ Retry SUCCESS for '{product_type}'")
                    else:
                        logger.warning(f"[RETRY_WORKER] ✗ Retry FAILED for '{product_type}'")
                        
                    # Sleep to prevent rate limits between retries
                    time.sleep(5) 
            
            # Wait before next check 
            time.sleep(300) # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"[RETRY_WORKER] Worker crashed: {e}")
            time.sleep(60) # Wait before restarting loop

def start_retry_worker():
    """Start the background retry thread."""
    retry_thread = threading.Thread(target=retry_generation_worker, daemon=True, name="ImageRetryWorker")
    retry_thread.start()
    logger.info("[RETRY_WORKER] Background thread launched")
