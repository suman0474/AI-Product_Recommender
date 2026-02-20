# agentic/strategy_csv_filter.py
# =============================================================================
# STRATEGY CSV FILTER - Simple CSV-Based Vendor Filtering
# =============================================================================
#
# PURPOSE: Replace complex Strategy RAG with simple CSV-based vendor filtering.
# Reads vendor strategy data from instrumentation_procurement_strategy.csv
# and filters vendors based on category/subcategory matching.
#
# CSV COLUMNS:
#   vendor ID, vendor name, category, subcategory, strategy, refinery,
#   additional comments, owner name
#
# USAGE:
#   filter = StrategyCSVFilter()
#   result = filter.filter_vendors_for_product(
#       product_type="pressure transmitter",
#       available_vendors=["Emerson", "ABB", "Siemens"]
#   )
#
# =============================================================================

import csv
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the strategy CSV file (for backward compatibility only - NOT USED)
# Primary source is Azure Blob Storage
STRATEGY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "instrumentation_procurement_strategy.csv"
)

# Category mapping: Map product types to CSV categories
PRODUCT_TO_CATEGORY_MAP = {
    # Pressure instruments
    "pressure transmitter": ("Pressure Instruments", None),
    "pressure transmitters": ("Pressure Instruments", None),  # Plural form
    "pressure gauge": ("Pressure Instruments", "Pressure Gauges"),
    "pressure gauges": ("Pressure Instruments", "Pressure Gauges"),  # Plural
    "differential pressure": ("Pressure Instruments", "Differential Pressure Transmitters"),
    "pressure sensor": ("Pressure Instruments", None),
    "pressure sensors": ("Pressure Instruments", None),  # Plural

    # Temperature instruments
    "temperature sensor": ("Temperature Instruments", None),
    "temperature sensors": ("Temperature Instruments", None),  # Plural
    "temperature transmitter": ("Temperature Instruments", None),
    "temperature transmitters": ("Temperature Instruments", None),  # Plural
    "thermocouple": ("Temperature Instruments", "Thermocouples"),
    "thermocouples": ("Temperature Instruments", "Thermocouples"),  # Plural
    "rtd": ("Temperature Instruments", "RTDs"),
    "infrared sensor": ("Temperature Instruments", "Infrared Sensors"),
    "infrared sensors": ("Temperature Instruments", "Infrared Sensors"),  # Plural
    "thermometer": ("Temperature Instruments", None),

    # Flow instruments
    "flow meter": ("Flow Instruments", None),
    "flow meters": ("Flow Instruments", None),  # Plural
    "flowmeter": ("Flow Instruments", None),
    "flowmeters": ("Flow Instruments", None),  # Plural
    "ultrasonic flow": ("Flow Instruments", "Ultrasonic Flow Meters"),
    "coriolis": ("Flow Instruments", None),
    "mass flow": ("Flow Instruments", None),
    "vortex": ("Flow Instruments", None),

    # Level instruments
    "level sensor": ("Level Instruments", None),
    "level sensors": ("Level Instruments", None),  # Plural
    "level transmitter": ("Level Instruments", None),
    "level transmitters": ("Level Instruments", None),  # Plural
    "radar level": ("Level Instruments", "Radar Level Sensors"),
    "ultrasonic level": ("Level Instruments", "Ultrasonic Level Sensors"),
    "capacitance level": ("Level Instruments", "Capacitance Level Sensors"),

    # Control valves
    "control valve": ("Control Valves", None),
    "control valves": ("Control Valves", None),  # Plural
    "ball valve": ("Control Valves", "Ball Valves"),
    "ball valves": ("Control Valves", "Ball Valves"),  # Plural
    "globe valve": ("Control Valves", "Globe Valves"),
    "globe valves": ("Control Valves", "Globe Valves"),  # Plural
    "butterfly valve": ("Control Valves", None),
    "butterfly valves": ("Control Valves", None),  # Plural

    # Analytical instruments
    "analyzer": ("Analytical Instruments", None),
    "analyzers": ("Analytical Instruments", None),  # Plural
    "analytical instruments": ("Analytical Instruments", None),  # Full name
    "ph meter": ("Analytical Instruments", None),
    "ph meters": ("Analytical Instruments", None),  # Plural
    "conductivity meter": ("Analytical Instruments", "Conductivity Meters"),
    "conductivity meters": ("Analytical Instruments", "Conductivity Meters"),  # Plural
    "dissolved oxygen": ("Analytical Instruments", "Dissolved Oxygen Meters"),
    "gas chromatograph": ("Analytical Instruments", None),

    # Safety instruments
    "gas detector": ("Safety Instruments", "Gas Detectors"),
    "gas detectors": ("Safety Instruments", "Gas Detectors"),  # Plural
    "safety valve": ("Safety Instruments", "Safety Valves"),
    "safety valves": ("Safety Instruments", "Safety Valves"),  # Plural
    "flame detector": ("Safety Instruments", None),

    # Vibration instruments
    "vibration sensor": ("Vibration Measurement Instruments", "Vibration Sensors"),
    "vibration sensors": ("Vibration Measurement Instruments", "Vibration Sensors"),  # Plural
    "vibrometer": ("Vibration Measurement Instruments", "Portable Vibrometers"),
    "accelerometer": ("Vibration Measurement Instruments", None),
    "accelerometers": ("Vibration Measurement Instruments", None),  # Plural

    # Signal conditioning
    "transmitter": ("Signal Conditioning", "Transmitters"),
    "transmitters": ("Signal Conditioning", "Transmitters"),  # Plural
    "signal converter": ("Signal Conditioning", "Signal Converters"),
    "signal converters": ("Signal Conditioning", "Signal Converters"),  # Plural
    "isolator": ("Signal Conditioning", None),
    "filter": ("Signal Conditioning", "Filters"),
}


class StrategyCSVFilter:
    """
    Simple CSV-based vendor strategy filter with lazy loading.

    Loads vendor strategy data from CSV and provides vendor filtering
    based on product category matching. Uses lazy loading to minimize memory usage.
    
    UPDATED: Reads from Azure Blob Storage.
    """

    def __init__(self, csv_path: str = None, use_tree_index: bool = True):
        """
        Initialize the strategy filter.

        Args:
            csv_path: Path to the strategy CSV file. Defaults to
                      instrumentation_procurement_strategy.csv in backend directory.
                      (Retained for backward compatibility, but primarily uses Blob)
            use_tree_index: If True, use the tree-based index for faster lookups (default: True)
        """
        self.csv_path = csv_path or STRATEGY_CSV_PATH
        self._category_cache = {}  # Lazy cache: category -> rows
        self._vendor_cache = {}    # Lazy cache: vendor_lower -> rows
        self._max_cache_size = 10  # LRU: Keep at most 10 categories in memory
        self._all_data = None      # Full data (loaded on demand)
        self._csv_content = None   # Cached RAW CSV content from Blob
        self._use_tree_index = use_tree_index
        self._tree_index = None    # Lazy-loaded tree index
        # Cache invalidation (Issue 4)
        self._cache_timestamp = None  # When cache was last loaded
        self._cached_user_id = None   # Which user's data is cached

    def _get_tree_index(self):
        """Get the tree index (lazy load)."""
        if not self._use_tree_index:
            return None
            
        if self._tree_index is None:
            try:
                from .document_index import strategy_index
                if strategy_index.ensure_loaded():
                    self._tree_index = strategy_index
                    logger.info("[StrategyCSVFilter] Using tree-based index for fast lookups")
                else:
                    logger.warning("[StrategyCSVFilter] Tree index unavailable, using CSV fallback")
            except Exception as e:
                logger.warning(f"[StrategyCSVFilter] Could not load tree index: {e}")
        
        return self._tree_index

    def _load_from_mongodb(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Load strategy vendor data from MongoDB with all fixes applied.

        Implements:
        - Issue 1: Status filtering (only "completed" documents)
        - Issue 2: Multiple documents per user (merge all user's documents)
        - Issue 3: User isolation with admin override (user docs + admin docs)
        - Issue 4: Cache invalidation (timestamp-based)
        - Issue 5: Empty results during processing (return status info)

        Args:
            user_id: Current user ID. If None, loads all documents (global mode)

        Returns:
            Dict with:
            - "vendors": List of vendor records
            - "document_count": Number of documents loaded
            - "processing_count": Number of documents still processing
            - "error": Error message if applicable
        """
        try:
            from common.core.mongodb_manager import mongodb_manager
            from datetime import datetime

            strategy_collection = mongodb_manager.get_collection('stratergy')

            if strategy_collection is None:
                logger.warning("[StrategyCSVFilter] MongoDB strategy collection not available")
                return {"vendors": [], "document_count": 0, "processing_count": 0, "error": "Database not available"}

            # Build query filter
            # Issue 3: User isolation with admin override
            if user_id is not None:
                # Load: user's own documents + admin documents
                query_filter = {
                    "$or": [
                        {"user_id": user_id},           # User's own uploads
                        {"is_admin_upload": True}       # Admin uploads (visible to all)
                    ]
                }
                logger.info(f"[StrategyCSVFilter] Loading documents for user {user_id} (including admin docs)")
            else:
                # Global mode - load all (for backward compatibility)
                query_filter = {}
                logger.info("[StrategyCSVFilter] Loading all documents (global mode)")

            # Issue 5: Check for processing documents first
            processing_count = strategy_collection.count_documents({
                **query_filter,
                "status": {"$in": ["pending", "processing"]}
            })

            # Issue 1: Only load "completed" documents
            query_filter["status"] = "completed"

            # Get documents with timestamp for cache invalidation
            documents = list(strategy_collection.find(
                query_filter,
                {
                    "data": 1,
                    "uploaded_at": 1,
                    "file_name": 1,
                    "user_id": 1,
                    "is_admin_upload": 1
                }
            ).sort("uploaded_at", -1))  # Most recent first

            if not documents:
                logger.warning(f"[StrategyCSVFilter] No completed documents found for user {user_id}")

                # Issue 5: Provide helpful message if documents are processing
                if processing_count > 0:
                    error_msg = f"{processing_count} document(s) are still being processed. Results will be available shortly."
                    logger.info(f"[StrategyCSVFilter] {error_msg}")
                    return {
                        "vendors": [],
                        "document_count": 0,
                        "processing_count": processing_count,
                        "error": error_msg
                    }
                else:
                    return {
                        "vendors": [],
                        "document_count": 0,
                        "processing_count": 0,
                        "error": "No strategy documents found. Please upload a strategy document first."
                    }

            # Issue 2: Merge all documents (user can have multiple strategy files)
            all_vendors = []
            admin_doc_count = 0
            user_doc_count = 0

            for doc in documents:
                vendor_data = doc.get('data', [])
                if isinstance(vendor_data, list):
                    all_vendors.extend(vendor_data)

                # Track document types
                if doc.get('is_admin_upload', False):
                    admin_doc_count += 1
                else:
                    user_doc_count += 1

            logger.info(
                f"[StrategyCSVFilter] Loaded {len(all_vendors)} vendor records from "
                f"{len(documents)} documents (admin: {admin_doc_count}, user: {user_doc_count})"
            )

            # Issue 4: Update cache timestamp
            self._cache_timestamp = datetime.utcnow().isoformat()
            self._cached_user_id = user_id

            return {
                "vendors": all_vendors,
                "document_count": len(documents),
                "processing_count": processing_count,
                "error": None
            }

        except Exception as e:
            logger.error(f"[StrategyCSVFilter] Failed to load from MongoDB: {e}")
            return {
                "vendors": [],
                "document_count": 0,
                "processing_count": 0,
                "error": f"Database error: {str(e)}"
            }

    def _get_csv_reader(self, user_id: Optional[int] = None):
        """
        Helper to get a DictReader from MongoDB or Blob content with fallback.

        Args:
            user_id: User ID for MongoDB user-specific loading

        Returns:
            CSV DictReader instance
        """
        import io

        # PRIORITY 1: Try MongoDB first (new primary source, user-specific)
        mongodb_result = self._load_from_mongodb(user_id=user_id)
        mongodb_data = mongodb_result.get('vendors', [])
        if mongodb_data:
            logger.info(f"[StrategyCSVFilter] Using MongoDB data source ({len(mongodb_data)} records)")
            # Convert MongoDB records to CSV format for compatibility
            # MongoDB format: {vendor_name, category, subcategory, strategy}
            # CSV format needs: vendor_id, vendor_name, category, subcategory, strategy, refinery, comments, owner
            csv_lines = ["vendor_id,vendor_name,category,subcategory,strategy,refinery,additional comments,owner name"]
            for idx, record in enumerate(mongodb_data):
                vendor_name = record.get('vendor_name', '').replace(',', '')
                category = record.get('category', '').replace(',', '')
                subcategory = record.get('subcategory', '').replace(',', '')
                strategy = record.get('strategy', '').replace(',', ';')  # Replace commas with semicolons
                csv_lines.append(f"{idx+1},{vendor_name},{category},{subcategory},{strategy},,,")

            self._csv_content = '\n'.join(csv_lines)
            return csv.DictReader(io.StringIO(self._csv_content))

        # PRIORITY 2: Try Azure Blob (legacy fallback)
        if self._csv_content is None:
            try:
                from common.core.azure_blob_file_manager import AzureBlobFileManager
                logger.info(f"[StrategyCSVFilter] Trying Azure Blob fallback...")
                blob_manager = AzureBlobFileManager()
                blob_name = os.path.basename(self.csv_path)
                container_name = blob_manager.CONTAINERS.get('strategy_documents', 'strategy-documents')
                content_bytes = blob_manager.download_file(blob_name, container_name)

                if not content_bytes:
                    logger.warning(f"[StrategyCSVFilter] Blob returned empty content for {blob_name}")
                    return self._get_fallback_reader()

                # Try UTF-8 first, then fall back to latin-1
                try:
                    self._csv_content = content_bytes.decode('utf-8')
                except (UnicodeDecodeError, AttributeError) as e:
                    logger.warning(f"[StrategyCSVFilter] UTF-8 decode failed, trying latin-1: {e}")
                    try:
                        self._csv_content = content_bytes.decode('latin-1')
                    except Exception as e2:
                        logger.warning(f"[StrategyCSVFilter] All decodings failed: {e2}, using fallback")
                        return self._get_fallback_reader()

            except Exception as e:
                logger.warning(f"[StrategyCSVFilter] Failed to load from Blob: {e}, using minimal fallback")
                return self._get_fallback_reader()

        if not self._csv_content:
            return self._get_fallback_reader()

        return csv.DictReader(io.StringIO(self._csv_content))

    def _get_fallback_reader(self):
        """Return a minimal fallback CSV reader when Blob fails."""
        logger.info("[StrategyCSVFilter] Using fallback minimal strategy data")

        # Minimal fallback data - basic vendor strategies for common products
        fallback_csv = """vendor_id,vendor_name,category,subcategory,strategy,refinery,additional comments,owner name
1,Emerson,Control Valves,,Preferred vendor for modulating control valves,Baton Rouge Refinery,High performance,John Smith
2,Yokogawa,Control Valves,,Approved vendor for control valves,Port Arthur Refinery,Good reliability,Jane Doe
3,Honeywell,Safety Instruments,Gas Detectors,Certified gas detectors for hazardous areas,Whiting Refinery,ATEX certified,Bob Johnson
4,Rosemount,Pressure Instruments,Pressure Transmitters,Standard pressure transmitter vendor,Baton Rouge Refinery,High accuracy,Alice Brown
5,ABB,Flow Instruments,Flow Meters,Preferred for coriolis meters,Port Arthur Refinery,High precision,Charlie Davis
"""
        import io
        return csv.DictReader(io.StringIO(fallback_csv))

    def _load_all_data(self, user_id: Optional[int] = None) -> List[Dict]:
        """
        Load all strategy data (MongoDB first, then CSV fallback).

        Implements cache invalidation - clears cache if user changes.

        Args:
            user_id: User ID for user-specific data loading

        Returns:
            List of normalized vendor records
        """
        # Issue 4: Cache invalidation - clear cache if user changed
        if self._all_data is not None and self._cached_user_id == user_id:
            return self._all_data
        elif self._all_data is not None and self._cached_user_id != user_id:
            logger.info(f"[StrategyCSVFilter] Cache invalidation: user changed from {self._cached_user_id} to {user_id}")
            self._all_data = None
            self._category_cache = {}

        self._all_data = []

        # PRIORITY 1: Try MongoDB first (with all fixes)
        mongodb_result = self._load_from_mongodb(user_id=user_id)
        mongodb_records = mongodb_result.get('vendors', [])

        if mongodb_records:
            logger.info(f"[StrategyCSVFilter] Using MongoDB source: {len(mongodb_records)} records")
            # Normalize MongoDB records to expected format
            for idx, record in enumerate(mongodb_records):
                normalized_row = {
                    'vendor_id': str(idx + 1),
                    'vendor_name': record.get('vendor_name', '').strip(),
                    'category': record.get('category', '').strip(),
                    'subcategory': record.get('subcategory', '').strip(),
                    'strategy': record.get('strategy', '').strip(),
                    'refinery': '',  # Not in MongoDB format
                    'comments': '',
                    'owner': ''
                }
                self._all_data.append(normalized_row)
            return self._all_data

        # Check if there's a helpful error message (Issue 5)
        if mongodb_result.get('error'):
            logger.warning(f"[StrategyCSVFilter] MongoDB returned error: {mongodb_result['error']}")
            # Still try fallback methods

        # PRIORITY 2: Fall back to CSV reader (Blob or local)
        try:
            reader = self._get_csv_reader(user_id=user_id)
            if not reader:
                return self._all_data

            for idx, row in enumerate(reader):
                # Normalize the row data
                normalized_row = {
                    'vendor_id': row.get('vendor ID', row.get('vendor_id', '')).strip(),
                    'vendor_name': row.get('vendor name', row.get('vendor_name', '')).strip(),
                    'category': row.get('category', '').strip(),
                    'subcategory': row.get('subcategory', '').strip(),
                    'strategy': row.get('strategy', '').strip(),
                    'refinery': row.get('refinery', '').strip(),
                    'comments': row.get('additional comments', '').strip(),
                    'owner': row.get('owner name', '').strip()
                }
                self._all_data.append(normalized_row)

            logger.info(f"[StrategyCSVFilter] Loaded {len(self._all_data)} strategy records from CSV")

        except Exception as e:
            logger.error(f"[StrategyCSVFilter] Error loading data: {e}")
            self._all_data = []

        return self._all_data

    def _load_category_lazy(self, category: str, subcategory: str = None, user_id: Optional[int] = None) -> List[Dict]:
        """
        Lazy load only rows matching a specific category (30x memory reduction).

        Optimized: Uses tree index for O(1) lookup if available, falls back to CSV scan.

        Args:
            category: Category name to load
            subcategory: Optional subcategory filter
            user_id: User ID for user-specific data loading

        Returns:
            List of matching rows
        """
        # Include user_id in cache key for proper cache isolation
        cache_key = (category, subcategory, user_id)

        # Check if already cached
        if cache_key in self._category_cache:
            logger.debug(f"[StrategyCSVFilter] Using cached category: {category}/{subcategory} for user {user_id}")
            return self._category_cache[cache_key]

        # Try tree index first (O(1) lookup)
        tree_index = self._get_tree_index()
        if tree_index:
            try:
                if subcategory:
                    vendors = tree_index.list_by_subcategory(category, subcategory)
                else:
                    vendors = tree_index.list_by_category(category)
                
                # Convert to expected format
                matching_rows = [{
                    'vendor_id': v['vendor_id'],
                    'vendor_name': v['vendor_name'],
                    'category': v['category'],
                    'subcategory': v['subcategory'],
                    'strategy': v['strategy'],
                    'refinery': v['refinery'],
                    'comments': v.get('additional_comments', ''),
                    'owner': v.get('owner_name', '')
                } for v in vendors]
                
                # Cache and return
                self._category_cache[cache_key] = matching_rows
                logger.debug(f"[StrategyCSVFilter] Tree index found {len(matching_rows)} vendors for {category}/{subcategory}")
                return matching_rows
                
            except Exception as e:
                logger.warning(f"[StrategyCSVFilter] Tree index lookup failed: {e}, falling back to CSV")

        # Fallback: Load only matching rows from CSV (user-specific)
        logger.debug(f"[StrategyCSVFilter] CSV fallback: lazy loading category: {category}/{subcategory} for user {user_id}")
        matching_rows = []

        try:
            reader = self._get_csv_reader(user_id=user_id)
            if not reader:
                logger.warning(f"[StrategyCSVFilter] CSV reader returned None for {category}")
                return matching_rows

            for row in reader:
                if not row:  # Skip empty rows
                    continue

                row_category = row.get('category', '').strip()
                row_subcategory = row.get('subcategory', '').strip()

                # Check if this row matches the requested category
                if row_category == category:
                    if subcategory is None or row_subcategory == subcategory:
                        normalized_row = {
                            'vendor_id': row.get('vendor ID', '').strip(),
                            'vendor_name': row.get('vendor name', '').strip(),
                            'category': row_category,
                            'subcategory': row_subcategory,
                            'strategy': row.get('strategy', '').strip(),
                            'refinery': row.get('refinery', '').strip(),
                            'comments': row.get('additional comments', '').strip(),
                            'owner': row.get('owner name', '').strip()
                        }
                        matching_rows.append(normalized_row)

        except Exception as e:
            logger.warning(f"[StrategyCSVFilter] Error loading category from source: {e}, trying fallback minimal data")
            # Use fallback reader on error
            try:
                fallback_reader = self._get_fallback_reader()
                for row in fallback_reader:
                    if not row:
                        continue
                    row_category = row.get('category', '').strip()
                    row_subcategory = row.get('subcategory', '').strip()

                    if row_category == category:
                        if subcategory is None or row_subcategory == subcategory:
                            vendor_id = row.get('vendor ID', row.get('vendor_id', '')).strip()
                            vendor_name = row.get('vendor name', row.get('vendor_name', '')).strip()
                            additional_comments = row.get('additional comments', row.get('additional_comments', '')).strip()
                            owner_name = row.get('owner name', row.get('owner_name', '')).strip()

                            normalized_row = {
                                'vendor_id': vendor_id,
                                'vendor_name': vendor_name,
                                'category': row_category,
                                'subcategory': row_subcategory,
                                'strategy': row.get('strategy', '').strip(),
                                'refinery': row.get('refinery', '').strip(),
                                'comments': additional_comments,
                                'owner': owner_name
                            }
                            matching_rows.append(normalized_row)
            except Exception as e2:
                logger.error(f"[StrategyCSVFilter] Even fallback failed: {e2}")
                return []

        # Cache with LRU eviction (keep max 10 categories)
        if len(self._category_cache) >= self._max_cache_size:
            # Evict oldest (first) entry
            oldest_key = next(iter(self._category_cache))
            del self._category_cache[oldest_key]
            logger.debug(f"[StrategyCSVFilter] Evicted category cache: {oldest_key}")

        self._category_cache[cache_key] = matching_rows
        logger.debug(f"[StrategyCSVFilter] Cached category: {category}/{subcategory} ({len(matching_rows)} rows)")

        return matching_rows

    def _map_product_to_category(self, product_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Map a product type to CSV category/subcategory.
        
        UPDATED: Uses vector store semantic matching as fallback.

        Args:
            product_type: Product type string (e.g., "pressure transmitter")

        Returns:
            Tuple of (category, subcategory) or (None, None) if no mapping
        """
        product_lower = product_type.lower().strip()

        # 1. Direct mapping (fast path)
        if product_lower in PRODUCT_TO_CATEGORY_MAP:
            return PRODUCT_TO_CATEGORY_MAP[product_lower]

        # 2. Try LLM-based standardization (NLP MAPPING via keyword_standardizer)
        try:
            from common.services.strategy.keyword_standardizer import get_standardizer
            std = get_standardizer()
            # Standardize 'product_type' into a category
            # This uses cached mappings or LLM to find the canonical category
            cat_full, _, conf = std.standardize_keyword(product_type, "category")
            
            if conf > 0.7 and cat_full:
                logger.debug(f"[StrategyCSVFilter] Standardizer mapped '{product_type}' -> '{cat_full}' (conf={conf:.2f})")
                return (cat_full, None)
                
        except Exception as e:
            logger.warning(f"[StrategyCSVFilter] Standardizer mapping failed: {e}")

        # 3. Fuzzy matching - check if any key is contained in product_type
        for key, (category, subcategory) in PRODUCT_TO_CATEGORY_MAP.items():
            if key in product_lower or product_lower in key:
                return (category, subcategory)

        # 3. Check for category keywords
        category_keywords = {
            "pressure": "Pressure Instruments",
            "temperature": "Temperature Instruments",
            "temp": "Temperature Instruments",
            "flow": "Flow Instruments",
            "level": "Level Instruments",
            "valve": "Control Valves",
            "analyz": "Analytical Instruments",
            "gas": "Safety Instruments",
            "safety": "Safety Instruments",
            "vibration": "Vibration Measurement Instruments",
            "signal": "Signal Conditioning"
        }

        for keyword, category in category_keywords.items():
            if keyword in product_lower:
                return (category, None)

        # 4. SEMANTIC FALLBACK: Use vector store for matching
        try:
            from .strategy_vector_store import get_strategy_vector_store
            
            vector_store = get_strategy_vector_store()
            category, subcategory = vector_store.match_product_to_category(product_type)
            
            if category:
                logger.info(f"[StrategyCSVFilter] Semantic match: '{product_type}' -> {category}/{subcategory}")
                return (category, subcategory)
                
        except ImportError:
            logger.debug("[StrategyCSVFilter] Vector store not available for semantic fallback")
        except Exception as e:
            logger.warning(f"[StrategyCSVFilter] Semantic fallback failed: {e}")

        return (None, None)

    def get_vendors_for_category(
        self,
        category: str,
        subcategory: str = None,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all vendors for a specific category/subcategory.

        Uses lazy loading to minimize memory usage - only loads requested categories.

        Args:
            category: CSV category (e.g., "Pressure Instruments")
            subcategory: Optional subcategory (e.g., "Pressure Gauges")
            user_id: User ID for user-specific strategy data

        Returns:
            List of vendor records with strategy information
        """
        # Use lazy loading to load only this category (user-specific)
        if subcategory:
            results = self._load_category_lazy(category, subcategory, user_id=user_id)
        else:
            # Load category with any subcategory
            results = self._load_category_lazy(category, None, user_id=user_id)

        return results

    def get_vendor_strategy(self, vendor_name: str, category: str = None) -> Dict[str, Any]:
        """
        Get strategy information for a specific vendor.

        Args:
            vendor_name: Vendor name to look up
            category: Optional category to filter by

        Returns:
            Dict with vendor strategy info, or empty dict if not found
        """
        # Load all data since we need to search by vendor across all categories
        all_data = self._load_all_data()

        vendor_lower = vendor_name.lower().strip()

        # Check for partial matches (e.g., "Emerson" matches "Emerson Electric Co.")
        matching_records = [
            row for row in all_data
            if vendor_lower in row['vendor_name'].lower() or row['vendor_name'].lower() in vendor_lower
        ]

        if not matching_records:
            return {}

        # If category specified, filter further
        if category:
            for record in matching_records:
                if record['category'] == category:
                    return record

        # Return first match
        return matching_records[0]

    def _get_vendors_with_aliases(
        self,
        category: str,
        subcategory: Optional[str],
        category_aliases: List[str],
        subcategory_aliases: List[str],
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get vendors matching category/subcategory using both original and standardized fields.

        Args:
            category: Original category
            subcategory: Original subcategory
            category_aliases: All category aliases (from query expansion)
            subcategory_aliases: All subcategory aliases (from query expansion)
            user_id: User ID for filtering

        Returns:
            List of matching vendor records
        """
        vendors = self.get_vendors_for_category(category, subcategory, user_id=user_id)

        # If we got vendors with exact match, return them
        if vendors:
            return vendors

        # Try matching with aliases using standardized fields
        all_vendors = self._load_from_mongodb(user_id) if user_id is not None else {}
        vendor_data = all_vendors.get('vendors', [])

        if not vendor_data:
            return []

        matched_vendors = []
        category_aliases_lower = [a.lower() for a in category_aliases if a]
        subcategory_aliases_lower = [a.lower() for a in subcategory_aliases if a]

        for vendor in vendor_data:
            # Check original fields
            vendor_cat = vendor.get('category', '').lower()
            vendor_subcat = vendor.get('subcategory', '').lower()

            # Check standardized fields
            vendor_cat_std = vendor.get('category_std', '').lower()
            vendor_subcat_std = vendor.get('subcategory_std', '').lower()

            # Check abbreviations
            vendor_cat_abbrev = vendor.get('category_abbrev', '').lower()
            vendor_subcat_abbrev = vendor.get('subcategory_abbrev', '').lower()

            # Match if any field matches any alias
            cat_match = (
                vendor_cat in category_aliases_lower or
                vendor_cat_std in category_aliases_lower or
                vendor_cat_abbrev in category_aliases_lower
            )

            subcat_match = True  # Default to true if no subcategory specified
            if subcategory and subcategory_aliases_lower:
                subcat_match = (
                    vendor_subcat in subcategory_aliases_lower or
                    vendor_subcat_std in subcategory_aliases_lower or
                    vendor_subcat_abbrev in subcategory_aliases_lower
                )

            if cat_match and subcat_match:
                matched_vendors.append(vendor)

        logger.info(f"[StrategyCSVFilter] Alias matching found {len(matched_vendors)} vendors")
        return matched_vendors

    def filter_vendors_for_product(
        self,
        product_type: str,
        available_vendors: List[str] = None,
        refinery: str = None,
        user_id: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Filter and prioritize vendors for a product type based on CSV strategy.

        This is the main entry point for vendor filtering.

        Filtering priority:
        1. If strategy provided: Filter by category + subcategory + strategy match
        2. If no strategy: Filter by category + subcategory only

        Args:
            product_type: Product type (e.g., "pressure transmitter")
            available_vendors: List of available vendor names to filter
            refinery: Optional refinery to filter by
            user_id: User ID for user-specific strategy data (includes admin docs)
            strategy: Optional strategy filter (e.g., "preferred", "approved", "high accuracy")
                     If provided, only vendors with matching strategy text are returned

        Returns:
            {
                "filtered_vendors": [
                    {
                        "vendor": "Vendor Name",
                        "strategy": "Strategy statement",
                        "priority_score": 10,
                        "category_match": True,
                        "refinery_match": True,
                        "strategy_match": True
                    }
                ],
                "excluded_vendors": [],
                "category": "Pressure Instruments",
                "total_strategy_entries": 123
            }
        """
        # Map product to category
        category, subcategory = self._map_product_to_category(product_type)

        logger.info(f"[StrategyCSVFilter] Filtering for product: {product_type} -> "
                   f"Category: {category}, Subcategory: {subcategory}, User: {user_id}, Strategy: {strategy}")

        # Expand query terms using keyword standardization for better matching
        try:
            from common.services.strategy.keyword_standardizer import get_standardizer
            standardizer = get_standardizer()

            # Expand category aliases
            category_aliases = standardizer.expand_query_term(category, "category", user_id=user_id) if category else [category]
            logger.debug(f"[StrategyCSVFilter] Category aliases: {category_aliases[:5]}")  # Log first 5

            # Expand subcategory aliases
            subcategory_aliases = standardizer.expand_query_term(subcategory, "subcategory", user_id=user_id) if subcategory else [subcategory]
            logger.debug(f"[StrategyCSVFilter] Subcategory aliases: {subcategory_aliases[:5]}")
        except Exception as e:
            logger.warning(f"[StrategyCSVFilter] Query expansion failed: {e}, using original terms")
            category_aliases = [category] if category else []
            subcategory_aliases = [subcategory] if subcategory else []

        # Get vendors in this category (lazy loaded, user-specific)
        # Try to match using BOTH original and standardized fields
        category_vendors = self._get_vendors_with_aliases(
            category, subcategory, category_aliases, subcategory_aliases, user_id=user_id
        ) if category else []

        # Apply strategy filter if provided (filter by category + subcategory + strategy)
        if strategy and category_vendors:
            strategy_lower = strategy.lower().strip()
            logger.info(f"[StrategyCSVFilter] Applying strategy filter: '{strategy}'")

            # Filter vendors whose strategy field contains the search term
            filtered_by_strategy = []
            for vendor in category_vendors:
                vendor_strategy = vendor.get('strategy', '').lower()
                if strategy_lower in vendor_strategy:
                    filtered_by_strategy.append(vendor)

            if filtered_by_strategy:
                logger.info(f"[StrategyCSVFilter] Strategy filter matched {len(filtered_by_strategy)}/{len(category_vendors)} vendors")
                category_vendors = filtered_by_strategy
            else:
                logger.warning(f"[StrategyCSVFilter] Strategy filter '{strategy}' matched 0 vendors, using all category vendors")
                # Don't filter - return all category vendors if no match

        # Build vendor strategy lookup
        vendor_strategies = {}
        for record in category_vendors:
            vname_lower = record['vendor_name'].lower()
            if vname_lower not in vendor_strategies:
                vendor_strategies[vname_lower] = []
            vendor_strategies[vname_lower].append(record)

        # Filter and score available vendors
        filtered = []
        excluded = []

        # If no available vendors specified, use all vendors from the category
        if not available_vendors and category_vendors:
            available_vendors = list(set(v['vendor_name'] for v in category_vendors))

        vendors_to_check = available_vendors if available_vendors else []

        for vendor in vendors_to_check:
            vendor_lower = vendor.lower().strip()

            # Find matching strategy records
            matching_records = []
            for vname, records in vendor_strategies.items():
                if vendor_lower in vname or vname in vendor_lower:
                    matching_records.extend(records)

            if not matching_records:
                # No strategy entry for this vendor in this category
                # Still include them but with lower priority
                filtered.append({
                    "vendor": vendor,
                    "strategy": "No specific strategy defined",
                    "priority_score": 5,
                    "category_match": False,
                    "refinery_match": False,
                    "comments": ""
                })
                continue

            # Calculate priority score based on strategy
            best_record = matching_records[0]
            priority_score = self._calculate_priority_score(best_record, refinery)

            # Check for refinery match
            refinery_match = False
            if refinery:
                for record in matching_records:
                    if refinery.lower() in record['refinery'].lower():
                        best_record = record
                        refinery_match = True
                        priority_score += 5  # Bonus for refinery match
                        break

            # Check for strategy match (if strategy filter was provided)
            strategy_match = False
            if strategy:
                vendor_strategy = best_record['strategy'].lower()
                if strategy.lower() in vendor_strategy:
                    strategy_match = True
                    priority_score += 10  # Bonus for strategy match

            filtered.append({
                "vendor": vendor,
                "strategy": best_record['strategy'],
                "priority_score": priority_score,
                "category_match": True,
                "refinery_match": refinery_match,
                "strategy_match": strategy_match if strategy else None,
                "comments": best_record['comments'],
                "vendor_id": best_record['vendor_id']
            })

        # Sort by priority score (descending)
        filtered.sort(key=lambda x: x['priority_score'], reverse=True)

        logger.info(f"[StrategyCSVFilter] Filtered {len(filtered)} vendors for {product_type}")

        # Get total strategy entries count (user-specific)
        all_data = self._load_all_data(user_id=user_id)
        total_entries = len(all_data) if all_data else 0

        return {
            "filtered_vendors": filtered,
            "excluded_vendors": excluded,
            "category": category,
            "subcategory": subcategory,
            "total_strategy_entries": total_entries
        }

    def _calculate_priority_score(self, record: Dict[str, Any], refinery: str = None) -> int:
        """
        Calculate priority score based on strategy statement.

        Higher score = higher priority.

        Args:
            record: Strategy record from CSV
            refinery: Optional refinery context

        Returns:
            Priority score (0-100)
        """
        strategy = record.get('strategy', '').lower()
        score = 10  # Base score

        # High priority strategies
        high_priority = [
            "long-term partnership",
            "framework agreement",
            "preferred supplier",
            "strategic partner"
        ]

        # Medium priority strategies
        medium_priority = [
            "dual sourcing",
            "multi-source",
            "standardization",
            "bundling",
            "volume discount",
            "bulk purchasing"
        ]

        # Lower priority strategies
        lower_priority = [
            "spend analysis",
            "cost optimization",
            "evaluation",
            "consignment"
        ]

        for phrase in high_priority:
            if phrase in strategy:
                score += 15
                break

        for phrase in medium_priority:
            if phrase in strategy:
                score += 10
                break

        for phrase in lower_priority:
            if phrase in strategy:
                score += 5
                break

        # Bonus for specific positive indicators
        if "sustainability" in strategy or "green" in strategy:
            score += 3
        if "lifecycle" in strategy or "life-cycle" in strategy:
            score += 3
        if "technology upgrade" in strategy or "digitalization" in strategy:
            score += 3

        return min(score, 100)  # Cap at 100

    def get_all_vendors(self) -> List[str]:
        """Get list of all unique vendor names from CSV."""
        all_data = self._load_all_data()
        return list(set(row['vendor_name'] for row in all_data)) if all_data else []

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories from CSV."""
        all_data = self._load_all_data()
        return list(set(row['category'] for row in all_data)) if all_data else []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
_strategy_filter_instance = None


def get_strategy_filter() -> StrategyCSVFilter:
    """Get or create singleton StrategyCSVFilter instance."""
    global _strategy_filter_instance
    if _strategy_filter_instance is None:
        _strategy_filter_instance = StrategyCSVFilter()
    return _strategy_filter_instance


def filter_vendors_by_strategy(
    product_type: str,
    available_vendors: List[str] = None,
    refinery: str = None,
    user_id: Optional[int] = None,
    strategy: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to filter vendors using CSV strategy.

    Args:
        product_type: Product type (e.g., "pressure transmitter")
        available_vendors: Optional list of vendors to filter
        refinery: Optional refinery context
        user_id: User ID for user-specific strategy data
        strategy: Optional strategy filter (e.g., "preferred", "approved")

    Returns:
        Filter result with filtered_vendors list
    """
    filter_instance = get_strategy_filter()
    return filter_instance.filter_vendors_for_product(
        product_type=product_type,
        available_vendors=available_vendors,
        refinery=refinery,
        user_id=user_id,
        strategy=strategy
    )


def get_vendor_strategy_info(vendor_name: str, product_type: str = None) -> Dict[str, Any]:
    """
    Get strategy information for a specific vendor.

    Args:
        vendor_name: Vendor name
        product_type: Optional product type for category context

    Returns:
        Strategy info dict
    """
    filter_instance = get_strategy_filter()

    category = None
    if product_type:
        category, _ = filter_instance._map_product_to_category(product_type)

    return filter_instance.get_vendor_strategy(vendor_name, category)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING STRATEGY CSV FILTER")
    print("=" * 80)

    strategy_filter = StrategyCSVFilter()

    # Test 1: Get all categories
    print("\n[Test 1] All Categories:")
    categories = strategy_filter.get_all_categories()
    for cat in categories:
        print(f"  - {cat}")

    # Test 2: Get all vendors
    print(f"\n[Test 2] Total Vendors: {len(strategy_filter.get_all_vendors())}")

    # Test 3: Filter for specific product types
    test_products = [
        "pressure transmitter",
        "flow meter",
        "temperature sensor",
        "control valve",
        "gas detector"
    ]

    test_vendors = ["Emerson", "ABB", "Siemens", "Yokogawa", "Honeywell", "WIKA"]

    for product in test_products:
        print(f"\n[Test 3] Filtering for: {product}")
        print("-" * 40)

        result = strategy_filter.filter_vendors_for_product(
            product_type=product,
            available_vendors=test_vendors
        )

        print(f"  Category: {result['category']}")
        print(f"  Filtered vendors:")
        for v in result['filtered_vendors'][:5]:
            print(f"    - {v['vendor']}: Score={v['priority_score']}, "
                  f"Strategy='{v['strategy'][:50]}...'")

    # Test 4: Get specific vendor strategy
    print("\n[Test 4] Emerson Strategy for Pressure Instruments:")
    info = get_vendor_strategy_info("Emerson", "pressure transmitter")
    if info:
        print(f"  Strategy: {info.get('strategy', 'N/A')}")
        print(f"  Comments: {info.get('comments', 'N/A')}")
    else:
        print("  No strategy found")

    print("\n" + "=" * 80)
