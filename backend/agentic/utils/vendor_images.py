# agentic/vendor_image_utils.py
# ==============================================================================
# VENDOR PRODUCT IMAGE UTILITIES FOR AGENTIC WORKFLOWS
# ==============================================================================
#
# Internal utilities for fetching real product images from the web during
# vendor analysis in the product search workflow.
#
# ARCHITECTURE:
# 1. Fetch vendor-specific product images from Google Custom Search API
# 2. Search within manufacturer domains for authentic product images
# 3. Fallback to SerpAPI/Serper if Google CSE fails
# 4. Return image URLs for enriching vendor analysis results
#
# USAGE:
#   from agentic.utils.vendor_images import fetch_vendor_product_images
#   images = fetch_vendor_product_images("Emerson", "Flow Meter", "Pressure Transmitter")
#   # Returns: [{"url": "...", "source": "google_cse", ...}, ...]
#
# ==============================================================================

import logging
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import hashlib
from io import BytesIO
import mimetypes

# Try to import Azure Blob config
try:
    from azure_blob_config import Collections, azure_blob_manager
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY")


def get_manufacturer_domains_from_llm(vendor_name: str) -> List[str]:
    """
    Get manufacturer domains for a vendor (static mapping with LLM fallback).

    Args:
        vendor_name: Name of the vendor/manufacturer

    Returns:
        List of domain names to search within
    """
    # Static mapping of common vendors to their domains
    VENDOR_DOMAINS = {
        'emerson': ['emerson.com', 'emersonprocess.com', 'emersonindustrial.com'],
        'honeywell': ['honeywell.com', 'honeywellprocess.com'],
        'siemens': ['siemens.com', 'siemens-industry.com'],
        'yokogawa': ['yokogawa.com', 'ydc.co.jp'],
        'wika': ['wika.com'],
        'abb': ['abb.com', 'abb-group.com'],
        'endress': ['endress.com', 'endress-hauser.com'],
        'rosemount': ['rosemount.com'],
        'danfoss': ['danfoss.com'],
        'bosch': ['bosch.com', 'boschsensortec.com'],
        'hydac': ['hydac.com'],
        'eaton': ['eaton.com'],
        'parker': ['parker.com'],
        'rexroth': ['bosch-rexroth.com'],
        'festo': ['festo.com'],
        'norgren': ['norgren.com'],
        'smc': ['smcworld.com', 'smcusa.com'],
        'aventics': ['aventics.com'],
    }

    vendor_lower = vendor_name.lower().replace(' ', '')

    # Try exact match first
    for key, domains in VENDOR_DOMAINS.items():
        if key in vendor_lower:
            logger.debug(f"[VENDOR_IMAGES] Domain mapping for '{vendor_name}': {domains}")
            return domains

    # Fallback: generate common domains from vendor name
    vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
    fallback_domains = [
        f"{vendor_clean}.com",
        f"{vendor_clean}.de",
        f"{vendor_clean}group.com"
    ]
    logger.debug(f"[VENDOR_IMAGES] Using fallback domains for '{vendor_name}': {fallback_domains}")
    return fallback_domains


def fetch_vendor_product_images_google_cse(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None,
    product_type: Optional[str] = None,
    manufacturer_domains: Optional[List[str]] = None,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch vendor-specific product images using Google Custom Search API.

    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: Optional specific product name/model
        model_family: Optional model family/series
        product_type: Optional type of product
        manufacturer_domains: Optional list of manufacturer domains to search
        timeout: Request timeout in seconds

    Returns:
        List of image dictionaries with URL, source, thumbnail, domain
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logger.warning("[VENDOR_IMAGES] Google CSE credentials not configured")
        return []

    try:
        from googleapiclient.discovery import build

        # Build search query
        query = vendor_name
        if model_family:
            query += f" {model_family}"
        if product_type:
            query += f" {product_type}"
        query += " product image"

        # Get manufacturer domains
        if manufacturer_domains is None:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"

        logger.debug(f"[VENDOR_IMAGES] Google CSE search for '{vendor_name}': {search_query[:100]}...")

        # Execute search
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=5,  # Limit to 5 images to avoid quota exhaustion
            safe="medium",
            imgSize="MEDIUM"
        ).execute()

        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']

        for item in result.get("items", []):
            url = item.get("link")

            # Validate URL
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                continue
            if not url.startswith(('http://', 'https://')):
                continue

            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })

        if images:
            logger.info(f"[VENDOR_IMAGES] Google CSE found {len(images)} images for {vendor_name}")
        return images

    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] Google CSE image search failed for {vendor_name}: {e}")
        return []


def fetch_vendor_logo_google_cse(
    vendor_name: str,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch vendor logo using Google Custom Search API.

    Args:
        vendor_name: Name of the vendor
        timeout: Request timeout

    Returns:
        List of image dictionaries
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logger.warning("[VENDOR_IMAGES] Google CSE credentials not configured")
        return []

    try:
        from googleapiclient.discovery import build

        # Build search query
        query = f"{vendor_name} logo vector transparent"
        
        logger.debug(f"[VENDOR_IMAGES] Google CSE logo search for '{vendor_name}'")

        # Execute search
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=query,
            cx=GOOGLE_CX,
            searchType="image",
            num=3,
            safe="medium",
            imgSize="MEDIUM",  # Prefer medium size for logos
            fileType="png"     # Prefer PNG for transparency
        ).execute()

        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']

        for item in result.get("items", []):
            url = item.get("link")

            # Validate URL
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                continue
            if not url.startswith(('http://', 'https://')):
                continue

            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })

        if images:
            logger.info(f"[VENDOR_IMAGES] Google CSE found {len(images)} logos for {vendor_name}")
        return images

    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] Google CSE logo search failed for {vendor_name}: {e}")
        return []


def fetch_vendor_product_images_serpapi(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None,
    product_type: Optional[str] = None,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch vendor-specific product images using SerpAPI (fallback).

    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: Optional specific product name
        model_family: Optional model family
        product_type: Optional product type
        timeout: Request timeout in seconds

    Returns:
        List of image dictionaries
    """
    if not SERPAPI_KEY:
        logger.warning("[VENDOR_IMAGES] SerpAPI key not configured")
        return []

    try:
        import requests

        # Build query
        query = vendor_name
        if model_family:
            query += f" {model_family}"
        if product_type:
            query += f" {product_type}"
        query += " product"

        logger.debug(f"[VENDOR_IMAGES] SerpAPI search for '{vendor_name}': {query}")

        # Execute search
        params = {
            "q": query,
            "tbm": "isch",  # Image search
            "api_key": SERPAPI_KEY,
            "num": 5
        }

        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=timeout
        )

        if response.status_code != 200:
            logger.warning(f"[VENDOR_IMAGES] SerpAPI returned {response.status_code}")
            return []

        data = response.json()
        images = []

        for item in data.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })

        if images:
            logger.info(f"[VENDOR_IMAGES] SerpAPI found {len(images)} images for {vendor_name}")
        return images

    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] SerpAPI search failed for {vendor_name}: {e}")
        return []


def verify_best_image_with_llm(
    image_urls: List[Dict[str, Any]],
    vendor_name: str,
    model_family: Optional[str] = None,
    product_name: Optional[str] = None,
    product_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to verify and select the best product image URL from search results.
    
    Args:
        image_urls: List of image dictionaries from search engines
        vendor_name: Vendor/manufacturer name
        model_family: Optional model family name
        product_name: Optional product name
        product_type: Optional product type
        
    Returns:
        Best image dict or None if verification fails
    """
    if not image_urls:
        logger.warning(f"[IMAGE_VERIFY] No images to verify for {vendor_name}")
        return None
    
    try:
        # Import LLM client
        import google.generativeai as genai
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1") or os.getenv("GOOGLE_API_KEY")
        
        if not GOOGLE_API_KEY:
            logger.warning("[IMAGE_VERIFY] No Google API key - skipping LLM verification")
            return image_urls[0]  # Return first image as fallback
        
        # Build context for LLM
        context_parts = [vendor_name]
        if model_family:
            context_parts.append(model_family)
        if product_name:
            context_parts.append(product_name)
        if product_type:
            context_parts.append(product_type)
        
        product_context = " ".join(context_parts)
        
        # Prepare image URLs with metadata for LLM
        image_candidates = []
        for idx, img in enumerate(image_urls[:5]):  # Limit to top 5 to avoid token limits
            image_candidates.append({
                "index": idx,
                "url": img.get("url", ""),
                "title": img.get("title", ""),
                "domain": img.get("domain", ""),
                "source": img.get("source", "")
            })
        
        # Create prompt for LLM
        prompt = f"""You are an expert at evaluating product images for industrial equipment and instrumentation.

Product Context:
- Vendor: {vendor_name}
- Model Family: {model_family or 'Not specified'}
- Product Type: {product_type or 'Not specified'}
- Product Name: {product_name or 'Not specified'}

Image Candidates:
{chr(10).join([f"{i+1}. URL: {img['url'][:100]}..." + 
               (f" | Title: {img['title'][:50]}" if img['title'] else "") + 
               (f" | Domain: {img['domain']}" if img['domain'] else "") 
               for i, img in enumerate(image_candidates)])}

Task: Analyze these image URLs and their metadata to determine which image is most likely to be:
1. An actual product photo (not a diagram, screenshot, or unrelated image)
2. From an official manufacturer source or reputable distributor
3. High quality and representative of the product
4. Most relevant to the vendor and product specified

Respond with ONLY a JSON object in this exact format:
{{
    "best_index": 0,
    "confidence": 0.85,
    "reason": "Brief explanation of why this image is best"
}}

Where best_index is the 1-based number (1-{len(image_candidates)}) of the best image.
If uncertain or none are suitable, set confidence below 0.5.
"""

        # Call LLM for verification
        logger.info(f"[IMAGE_VERIFY] Verifying {len(image_candidates)} images for {product_context}")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent selection
                "max_output_tokens": 200
            }
        )
        
        # Parse LLM response
        import json
        import re
        
        response_text = response.text.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
        elif "```" in response_text:
            response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
        
        verification_result = json.loads(response_text)
        
        best_index = verification_result.get("best_index", 1) - 1  # Convert to 0-based
        confidence = verification_result.get("confidence", 0.0)
        reason = verification_result.get("reason", "No reason provided")
        
        logger.info(
            f"[IMAGE_VERIFY] LLM selected image #{best_index + 1} "
            f"with confidence {confidence:.2f}: {reason}"
        )
        
        # Only return if confidence is reasonable
        if 0 <= best_index < len(image_candidates) and confidence >= 0.3:
            selected_image = image_urls[best_index].copy()
            selected_image['llm_verified'] = True
            selected_image['llm_confidence'] = confidence
            selected_image['llm_reason'] = reason
            return selected_image
        else:
            logger.warning(f"[IMAGE_VERIFY] Low confidence ({confidence}) - using first image")
            return image_urls[0]
        
    except Exception as e:
        logger.warning(f"[IMAGE_VERIFY] LLM verification failed: {e}", exc_info=True)
        # Fallback to first image
        return image_urls[0] if image_urls else None


def fetch_vendor_product_images(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None,
    product_type: Optional[str] = None,
    max_retries: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch vendor-specific product images with LLM verification and caching.
    
    ENHANCED FLOW:
    1. Check Azure Blob cache using model_family as key
    2. If not cached, fetch from search engines (Google CSE → SerpAPI fallback)
    3. Use LLM to verify and select the best image from results
    4. Cache the verified image to Azure Blob using model_family
    5. Return cached image URL

    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: Optional product name
        model_family: Optional model family (used as cache key)
        product_type: Optional product type
        max_retries: Number of retry attempts on failure

    Returns:
        List of image dictionaries with URL, source, etc.
    """
    logger.info(f"[VENDOR_IMAGES] Fetching images for vendor: {vendor_name}, model: {model_family}")

    # =========================================================================
    # STEP 1: Check Azure Blob cache first
    # =========================================================================
    cache_key = model_family or vendor_name
    
    try:
        from services.azure.blob_utils import get_cached_image
        
        cached_image = get_cached_image(vendor_name, cache_key)
        if cached_image:
            logger.info(f"[VENDOR_IMAGES] ✓ Cache HIT for {vendor_name} - {cache_key}")
            # Convert to expected format
            return [{
                'url': f"/api/images/{cached_image.get('blob_path', '')}",
                'title': cached_image.get('title', ''),
                'source': cached_image.get('source', 'cache'),
                'domain': cached_image.get('domain', ''),
                'cached': True,
                'content_type': cached_image.get('content_type', 'image/jpeg')
            }]
    except ImportError:
        logger.debug("[VENDOR_IMAGES] Azure utils not available, skipping cache check")
    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] Cache check failed: {e}")
    
    # =========================================================================
    # STEP 2: Cache miss - fetch from search engines
    # =========================================================================
    logger.info(f"[VENDOR_IMAGES] Cache MISS - searching for {vendor_name}")
    
    # Try Google CSE first
    images = fetch_vendor_product_images_google_cse(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )

    # If no results, try SerpAPI
    if not images:
        logger.info(f"[VENDOR_IMAGES] No Google CSE results, trying SerpAPI...")
        images = fetch_vendor_product_images_serpapi(
            vendor_name=vendor_name,
            product_name=product_name,
            model_family=model_family,
            product_type=product_type
        )

    if not images:
        logger.warning(f"[VENDOR_IMAGES] No images found for {vendor_name}")
        return []
    
    logger.info(f"[VENDOR_IMAGES] Retrieved {len(images)} candidate images")
    
    # =========================================================================
    # STEP 3: Use LLM to verify and select the best image
    # =========================================================================
    best_image = verify_best_image_with_llm(
        image_urls=images,
        vendor_name=vendor_name,
        model_family=model_family,
        product_name=product_name,
        product_type=product_type
    )
    
    if not best_image:
        logger.warning(f"[VENDOR_IMAGES] LLM verification failed - no images selected")
        return []
    
    logger.info(
        f"[VENDOR_IMAGES] Selected best image: "
        f"{best_image.get('url', '')[:100]}... "
        f"(LLM verified: {best_image.get('llm_verified', False)})"
    )
    
    # =========================================================================
    # STEP 4: Cache the verified best image to Azure Blob
    # =========================================================================
    try:
        from services.azure.blob_utils import cache_image
        
        cache_success = cache_image(
            vendor_name=vendor_name,
            model_family=cache_key,
            image_data=best_image
        )
        
        if cache_success:
            logger.info(f"[VENDOR_IMAGES] ✓ Cached verified image to Azure Blob for {vendor_name} - {cache_key}")
            
            # Return the cached version with API URL
            from services.azure.blob_utils import get_cached_image
            cached_image = get_cached_image(vendor_name, cache_key)
            if cached_image:
                return [{
                    'url': f"/api/images/{cached_image.get('blob_path', '')}",
                    'title': best_image.get('title', ''),
                    'source': 'cache',
                    'domain': best_image.get('domain', ''),
                    'cached': True,
                    'llm_verified': best_image.get('llm_verified', False),
                    'llm_confidence': best_image.get('llm_confidence', 0.0)
                }]
        else:
            logger.warning(f"[VENDOR_IMAGES] Failed to cache image to Azure")
            
    except ImportError:
        logger.debug("[VENDOR_IMAGES] Azure utils not available, skipping cache")
    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] Caching failed: {e}", exc_info=True)
    
    # =========================================================================
    # STEP 5: Fallback - download and serve directly (temporary caching)
    # =========================================================================
    logger.info(f"[VENDOR_IMAGES] Azure caching failed, using temporary cache")
    
    cached_result = cache_vendor_image(best_image["url"], vendor_name)
    if cached_result:
        best_image["url"] = cached_result["url"]
        best_image["cached"] = True
        best_image["local_path"] = cached_result.get("local_path")
    
    return [best_image]


def cache_vendor_image(image_url: str, vendor_name: str) -> Optional[Dict[str, Any]]:
    """
    Download and cache a vendor image to Azure Blob (or local).
    
    Args:
        image_url: Original URL of the image
        vendor_name: Name of the vendor (for organization)
        
    Returns:
        Dict with 'url' (API URL) and 'local_path' or None if failed
    """
    try:
        # Generate unique filename hash
        img_hash = hashlib.md5(image_url.encode()).hexdigest()
        
        # Download image
        try:
            response = requests.get(image_url, timeout=10, stream=True)
            if response.status_code != 200:
                logger.warning(f"[CACHE_IMG] Failed to download {image_url}: {response.status_code}")
                return None
                
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logger.warning(f"[CACHE_IMG] URL is not an image: {content_type}")
                return None
                
            image_data = response.content
            
            # Determine extension
            ext = mimetypes.guess_extension(content_type) or '.png'
            if ext == '.jpe': ext = '.jpg'
            
            filename = f"{vendor_name.lower().replace(' ', '_')}_{img_hash}{ext}"
            
        except Exception as e:
            logger.warning(f"[CACHE_IMG] Download error: {e}")
            return None

        # 1. Try Azure Blob Storage
        if AZURE_AVAILABLE and azure_blob_manager:
            try:
                from azure.storage.blob import ContentSettings
                
                blob_path = f"vendor_images/{filename}"
                blob_client = azure_blob_manager.get_blob_client(blob_path)
                
                # Check if exists
                if not blob_client.exists():
                    blob_client.upload_blob(
                        image_data,
                        overwrite=True,
                        content_settings=ContentSettings(content_type=content_type)
                    )
                    logger.info(f"[CACHE_IMG] Uploaded to Azure: {blob_path}")
                
                # Return API path - NO leading /api prefix (frontend handles it)
                return {
                    "url": f"/api/images/{blob_path}", 
                    "local_path": blob_path,
                    "storage": "azure"
                }
            except Exception as azure_err:
                logger.warning(f"[CACHE_IMG] Azure upload failed: {azure_err}")
                # Fallthrough to local
        
        # 2. Fallback to Local Storage
        try:
            local_dir = os.path.join(os.getcwd(), 'static', 'images', 'vendor_images')
            os.makedirs(local_dir, exist_ok=True)
            
            local_path = os.path.join(local_dir, filename)
            
            if not os.path.exists(local_path):
                with open(local_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"[CACHE_IMG] Saved locally: {local_path}")
                
            # Return API path
            return {
                "url": f"/api/images/vendor_images/{filename}",
                "local_path": filename,
                "storage": "local"
            }
        except Exception as local_err:
            logger.error(f"[CACHE_IMG] Local save failed: {local_err}")
            return None
            
    except Exception as e:
        logger.error(f"[CACHE_IMG] Critical cache error: {e}")
        return None


def fetch_images_for_vendor_matches(
    vendor_name: str,
    matches: List[Dict[str, Any]],
    max_workers: int = 3
) -> List[Dict[str, Any]]:
    """
    Enrich vendor matches with product images.

    Fetches images for vendor and optionally for specific product models.
    Uses model family and product type for better LLM verification and caching.

    Args:
        vendor_name: Vendor to fetch images for
        matches: List of matched products
        max_workers: Max parallel workers for image fetching

    Returns:
        Enriched matches with image URLs
    """
    try:
        # Extract model family and product type from first match for context
        model_family = None
        product_type = None
        product_name = None
        
        if matches:
            first_match = matches[0]
            # Try various field names for model family
            model_family = (first_match.get('model_family') or 
                          first_match.get('modelFamily') or 
                          first_match.get('model_name') or
                          first_match.get('modelName'))
            
            # Try various field names for product type
            product_type = (first_match.get('product_type') or 
                          first_match.get('productType') or
                          first_match.get('category'))
            
            # Try various field names for product name
            product_name = (first_match.get('product_name') or 
                          first_match.get('productName') or
                          first_match.get('name'))
        
        logger.info(
            f"[VENDOR_IMAGES] Fetching images for {vendor_name} "
            f"(model: {model_family}, type: {product_type})"
        )
        
        # Fetch vendor-level images with enhanced context
        vendor_images = fetch_vendor_product_images(
            vendor_name=vendor_name,
            product_name=product_name,
            model_family=model_family,
            product_type=product_type
        )

        # Add images to matches
        for match in matches:
            # If we have product-specific images, use those
            if vendor_images:
                # Attach vendor-level images to each match
                # Future enhancement: Could do product-specific image search per match
                match['product_images'] = vendor_images[:2]  # Top 2 images (but we now only return 1 verified)
                match['image_source'] = vendor_images[0].get('source', 'unknown') if vendor_images else None
                
                # Add LLM verification metadata if available
                if vendor_images[0].get('llm_verified'):
                    match['llm_verified_image'] = True
                    match['llm_confidence'] = vendor_images[0].get('llm_confidence', 0.0)
            else:
                match['product_images'] = []
                match['image_source'] = None

        logger.info(f"[VENDOR_IMAGES] Enriched {len(matches)} matches with images")
        return matches

    except Exception as e:
        logger.warning(f"[VENDOR_IMAGES] Failed to enrich matches with images: {e}")
        # Return matches without images on error
        for match in matches:
            match['product_images'] = []
            match['image_source'] = None
        return matches


def fetch_vendor_logo(vendor_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch and cache a logo for the specified vendor with MongoDB + Azure Blob hybrid caching (PHASE 3).

    CACHING FLOW:
    1. Check MongoDB cache first (<50ms)
    2. Check Azure Blob cache (1-2s)
    3. Fetch from Google CSE (2-5s)
    4. Cache to BOTH MongoDB + Azure Blob

    Args:
        vendor_name: Name of the vendor

    Returns:
        Dict with logo URL and metadata, or None if failed
    """
    if not vendor_name:
        return None

    normalized_name = vendor_name.strip().lower().replace(' ', '_').replace('.', '').replace(',', '')
    cache_key = f"{normalized_name}_logo"

    # PHASE 3: Check MongoDB cache FIRST (<50ms)
    try:
        from services.image_service import image_service
        if image_service:
            mongo_logo = image_service.get_vendor_logo(vendor_name)
            if mongo_logo:
                logger.debug(f"[VENDOR_LOGO] MongoDB cache HIT for {vendor_name} (<50ms)")
                return {
                    'url': mongo_logo.get('logo_url', ''),
                    'logo': mongo_logo.get('logo_url', ''),  # Alias for frontend
                    'source': 'cache',
                    'cache_source': 'mongodb',
                    'cached': True
                }
            else:
                logger.debug(f"[VENDOR_LOGO] MongoDB cache MISS for {vendor_name}")
    except ImportError:
        logger.debug(f"[VENDOR_LOGO] image_service not available, skipping MongoDB cache")
    except Exception as e:
        logger.warning(f"[VENDOR_LOGO] MongoDB cache check error: {e}, falling back to Azure Blob")

    # Check Azure Blob Cache (fallback)
    try:
        if AZURE_AVAILABLE:
            from services.azure.blob_utils import get_cached_image
            cached = get_cached_image(vendor_name, cache_key)
            if cached:
                logger.debug(f"[VENDOR_LOGO] Azure Blob cache HIT for {vendor_name}")
                return {
                    'url': f"/api/images/{cached.get('blob_path', '')}",
                    'logo': f"/api/images/{cached.get('blob_path', '')}",  # Alias for frontend
                    'source': 'cache',
                    'cache_source': 'azure_blob',
                    'cached': True
                }
    except Exception as e:
        logger.warning(f"[VENDOR_LOGO] Azure Blob cache check failed: {e}")

    # 2. Fetch from Google CSE
    logger.info(f"[VENDOR_LOGO] Cache MISS - fetching logo for {vendor_name}")
    logos = fetch_vendor_logo_google_cse(vendor_name)
    
    # 3. Fallback to general search if specific logo search failed
    if not logos:
        # Try generic search
        logos = fetch_vendor_product_images_google_cse(vendor_name + " logo", product_type="")
    
    if not logos:
        return None
        
    # Use the first logo
    best_logo = logos[0]
    
    # 4. Cache to BOTH Azure Blob + MongoDB (PHASE 3)
    azure_cache_success = False
    azure_blob_url = None

    try:
        if AZURE_AVAILABLE:
            from services.azure.blob_utils import cache_image
            # Cache using the logo specific key
            cache_image(vendor_name, cache_key, best_logo)

            # Re-fetch via cache to get proper blob path
            from services.azure.blob_utils import get_cached_image
            cached = get_cached_image(vendor_name, cache_key)
            if cached:
                azure_cache_success = True
                azure_blob_url = f"/api/images/{cached.get('blob_path', '')}"
                logger.info(f"[VENDOR_LOGO] ✓ Cached to Azure Blob for {vendor_name}")
    except Exception as e:
        logger.warning(f"[VENDOR_LOGO] Azure Blob cache write failed: {e}")

    # PHASE 3: Also cache to MongoDB for fast lookups (<50ms)
    if azure_cache_success:
        try:
            from services.image_service import image_service
            if image_service:
                # Use the Azure Blob URL (already has SAS from get_cached_image)
                logo_url_with_sas = azure_blob_url

                mongo_result = image_service.cache_vendor_logo(
                    vendor_name=vendor_name,
                    logo_url=logo_url_with_sas,
                    source=best_logo.get('source', 'google_cse'),
                    metadata={
                        'normalized_name': normalized_name,
                        'cache_key': cache_key,
                        'original_url': best_logo.get('url', ''),
                        'search_method': 'google_cse'
                    }
                )

                if mongo_result.get('success'):
                    logger.info(f"[VENDOR_LOGO] ✓ Also cached to MongoDB (cache_id: {mongo_result.get('cache_id')})")

                    # Return MongoDB-cached response
                    return {
                        'url': logo_url_with_sas,
                        'logo': logo_url_with_sas,
                        'source': 'cache',
                        'cache_source': 'mongodb',
                        'cached': True
                    }
        except ImportError:
            logger.debug(f"[VENDOR_LOGO] image_service not available, MongoDB caching skipped")
        except Exception as mongo_error:
            logger.warning(f"[VENDOR_LOGO] MongoDB caching error (Azure cache still OK): {mongo_error}")

    # Return Azure Blob cached URL if available
    if azure_cache_success:
        return {
            'url': azure_blob_url,
            'logo': azure_blob_url,
            'source': 'cache',
            'cache_source': 'azure_blob',
            'cached': True
        }

    # Fallback: return original URL if caching failed
    return {
        'url': best_logo['url'],
        'logo': best_logo['url'],
        'source': best_logo['source'],
        'cached': False
    }


def enrich_matches_with_logos(
    matches: List[Dict[str, Any]],
    max_workers: int = 2
) -> List[Dict[str, Any]]:
    """
    Enrich vendor matches with vendor logos.
    
    Args:
        matches: List of matched products
        max_workers: Max parallel workers
        
    Returns:
        Matches enriched with 'vendor_logo' field
    """
    if not matches:
        return matches
        
    # Get unique vendors
    unique_vendors = list(set(m.get('vendor') for m in matches if m.get('vendor')))
    logger.info(f"[VENDOR_LOGO] Fetching logos for {len(unique_vendors)} vendors")
    
    vendor_logos = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_vendor = {
            executor.submit(fetch_vendor_logo, vendor): vendor 
            for vendor in unique_vendors
        }
        
        for future in as_completed(future_to_vendor):
            vendor = future_to_vendor[future]
            try:
                logo_data = future.result()
                if logo_data:
                    vendor_logos[vendor] = logo_data
            except Exception as e:
                logger.error(f"[VENDOR_LOGO] Failed to fetch logo for {vendor}: {e}")
                
    # Attach logos to matches
    populated_count = 0
    for match in matches:
        vendor = match.get('vendor')
        if vendor and vendor in vendor_logos:
            match['vendor_logo'] = vendor_logos[vendor]
            match['vendorLogo'] = vendor_logos[vendor] # CamelCase alias
            populated_count += 1
            
    logger.info(f"[VENDOR_LOGO] Enriched {populated_count} matches with logos")
    return matches


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'fetch_vendor_product_images',
    'fetch_vendor_product_images_google_cse',
    'fetch_vendor_product_images_serpapi',
    'fetch_images_for_vendor_matches',
    'get_manufacturer_domains_from_llm',
    'fetch_vendor_logo',
    'enrich_matches_with_logos'
]
