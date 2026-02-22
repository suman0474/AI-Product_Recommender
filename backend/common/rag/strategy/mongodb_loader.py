# common/rag/strategy/mongodb_loader.py
# =============================================================================
# STRATEGY MONGODB LOADER
# =============================================================================
#
# PURPOSE: Load vendor strategy data directly from MongoDB.
#          Replaces the old CSV-based StrategyCSVFilter.
#
# Data model (MongoDB collection: 'stratergy'):
#   {
#     user_id: int,
#     is_admin_upload: bool,
#     status: "completed" | "pending" | "processing",
#     uploaded_at: datetime,
#     data: [
#       { vendor_name, category, subcategory, strategy, ... }
#     ]
#   }
#
# Isolation rules:
#   - A user sees their own docs + admin docs (is_admin_upload=True)
#   - Only "completed" documents are used
#   - If user has no docs, admin docs act as the global/default strategy
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product-type → category keyword mapping (used for fast classification)
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS = {
    "pressure":    "Pressure Instruments",
    "temperature": "Temperature Instruments",
    "temp":        "Temperature Instruments",
    "flow":        "Flow Instruments",
    "level":       "Level Instruments",
    "valve":       "Control Valves",
    "analyz":      "Analytical Instruments",
    "gas":         "Safety Instruments",
    "safety":      "Safety Instruments",
    "vibration":   "Vibration Measurement Instruments",
    "signal":      "Signal Conditioning",
}

_PRODUCT_TO_CATEGORY: Dict[str, Tuple[str, Optional[str]]] = {
    "pressure transmitter":             ("Pressure Instruments", None),
    "pressure gauge":                   ("Pressure Instruments", "Pressure Gauges"),
    "differential pressure":            ("Pressure Instruments", "Differential Pressure Transmitters"),
    "pressure sensor":                  ("Pressure Instruments", None),
    "temperature sensor":               ("Temperature Instruments", None),
    "temperature transmitter":          ("Temperature Instruments", None),
    "thermocouple":                     ("Temperature Instruments", "Thermocouples"),
    "rtd":                              ("Temperature Instruments", "RTDs"),
    "infrared sensor":                  ("Temperature Instruments", "Infrared Sensors"),
    "flow meter":                       ("Flow Instruments", None),
    "flowmeter":                        ("Flow Instruments", None),
    "ultrasonic flow":                  ("Flow Instruments", "Ultrasonic Flow Meters"),
    "coriolis":                         ("Flow Instruments", None),
    "mass flow":                        ("Flow Instruments", None),
    "vortex":                           ("Flow Instruments", None),
    "level sensor":                     ("Level Instruments", None),
    "level transmitter":                ("Level Instruments", None),
    "radar level":                      ("Level Instruments", "Radar Level Sensors"),
    "ultrasonic level":                 ("Level Instruments", "Ultrasonic Level Sensors"),
    "control valve":                    ("Control Valves", None),
    "ball valve":                       ("Control Valves", "Ball Valves"),
    "globe valve":                      ("Control Valves", "Globe Valves"),
    "butterfly valve":                  ("Control Valves", None),
    "analyzer":                         ("Analytical Instruments", None),
    "ph meter":                         ("Analytical Instruments", None),
    "conductivity meter":               ("Analytical Instruments", "Conductivity Meters"),
    "dissolved oxygen":                 ("Analytical Instruments", "Dissolved Oxygen Meters"),
    "gas chromatograph":                ("Analytical Instruments", None),
    "gas detector":                     ("Safety Instruments", "Gas Detectors"),
    "safety valve":                     ("Safety Instruments", "Safety Valves"),
    "flame detector":                   ("Safety Instruments", None),
    "vibration sensor":                 ("Vibration Measurement Instruments", "Vibration Sensors"),
    "vibrometer":                       ("Vibration Measurement Instruments", "Portable Vibrometers"),
    "accelerometer":                    ("Vibration Measurement Instruments", None),
    "transmitter":                      ("Signal Conditioning", "Transmitters"),
    "signal converter":                 ("Signal Conditioning", "Signal Converters"),
    "isolator":                         ("Signal Conditioning", None),
}


def _map_product_to_category(product_type: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a free-text product type to a (category, subcategory) pair."""
    product_lower = product_type.lower().strip()

    # 1. Exact map
    if product_lower in _PRODUCT_TO_CATEGORY:
        return _PRODUCT_TO_CATEGORY[product_lower]

    # 2. Plural / partial key containment
    for key, value in _PRODUCT_TO_CATEGORY.items():
        if key in product_lower or product_lower in key:
            return value

    # 3. Keyword heuristics
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in product_lower:
            return (category, None)

    return (None, None)


# ---------------------------------------------------------------------------
# Core MongoDB loader
# ---------------------------------------------------------------------------

def load_strategy_from_mongodb(user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Load vendor strategy records from MongoDB.

    Implements full isolation + admin-override:
      - user_id provided  → user's own docs  +  admin docs
      - user_id=None      → all docs (global / backward-compat mode)

    Only "completed" documents are considered.

    Returns:
        {
          "vendors": [...],           # normalised vendor records
          "document_count": int,
          "processing_count": int,    # docs still being ingested
          "error": str | None
        }
    """
    try:
        from common.core.mongodb_manager import mongodb_manager

        collection = mongodb_manager.get_collection('stratergy')
        if collection is None:
            logger.warning("[StrategyMongoDB] Collection not available")
            return {"vendors": [], "document_count": 0, "processing_count": 0,
                    "error": "Database not available"}

        # Build isolation query
        if user_id is not None:
            base_filter: Dict[str, Any] = {
                "$or": [
                    {"user_id": user_id},
                    {"is_admin_upload": True}
                ]
            }
            logger.info(f"[StrategyMongoDB] Loading for user {user_id} + admin docs")
        else:
            base_filter = {}
            logger.info("[StrategyMongoDB] Loading all documents (global mode)")

        # Count still-processing docs
        processing_count = collection.count_documents({
            **base_filter,
            "status": {"$in": ["pending", "processing"]}
        })

        # Fetch only completed docs
        query = {**base_filter, "status": "completed"}
        documents = list(collection.find(
            query,
            {"data": 1, "uploaded_at": 1, "file_name": 1,
             "user_id": 1, "is_admin_upload": 1}
        ).sort("uploaded_at", -1))

        if not documents:
            msg = (
                f"{processing_count} document(s) are still being processed."
                if processing_count > 0
                else "No strategy documents found. Please upload a strategy document first."
            )
            logger.warning(f"[StrategyMongoDB] No completed docs — {msg}")
            return {"vendors": [], "document_count": 0,
                    "processing_count": processing_count, "error": msg}

        # Merge all docs
        all_vendors: List[Dict] = []
        admin_count = user_count = 0
        for doc in documents:
            rows = doc.get("data", [])
            if isinstance(rows, list):
                all_vendors.extend(rows)
            if doc.get("is_admin_upload"):
                admin_count += 1
            else:
                user_count += 1

        logger.info(
            f"[StrategyMongoDB] Loaded {len(all_vendors)} records from "
            f"{len(documents)} docs (admin={admin_count}, user={user_count})"
        )
        return {
            "vendors": all_vendors,
            "document_count": len(documents),
            "processing_count": processing_count,
            "error": None
        }

    except Exception as exc:
        logger.error(f"[StrategyMongoDB] Load failed: {exc}")
        return {"vendors": [], "document_count": 0,
                "processing_count": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# High-level helpers (drop-in replacements for old csv_filter functions)
# ---------------------------------------------------------------------------

def _calculate_priority_score(record: Dict[str, Any], refinery: Optional[str] = None) -> int:
    """Score a vendor record based on its strategy text."""
    strategy = record.get("strategy", "").lower()
    score = 10  # base

    high = ["long-term partnership", "framework agreement",
            "preferred supplier", "strategic partner"]
    medium = ["dual sourcing", "multi-source", "standardization",
              "bundling", "volume discount", "bulk purchasing"]

    if any(h in strategy for h in high):
        score += 20
    elif any(m in strategy for m in medium):
        score += 10

    if refinery and refinery.lower() in record.get("refinery", "").lower():
        score += 5

    return score


def filter_vendors_by_strategy(
    product_type: str,
    available_vendors: Optional[List[str]] = None,
    refinery: Optional[str] = None,
    user_id: Optional[int] = None,
    strategy: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter and prioritise vendors for a product type using MongoDB strategy data.

    This is the primary entry point — replaces StrategyCSVFilter.filter_vendors_for_product().

    Args:
        product_type:       e.g. "pressure transmitter"
        available_vendors:  vendor names to score (if None, uses all from DB)
        refinery:           optional site/refinery context
        user_id:            user ID for isolation (includes admin docs)
        strategy:           optional strategy keyword filter

    Returns:
        {
          "filtered_vendors": [...],
          "excluded_vendors": [],
          "category": str,
          "subcategory": str,
          "total_strategy_entries": int
        }
    """
    category, subcategory = _map_product_to_category(product_type)

    logger.info(f"[StrategyMongoDB] filter_vendors_by_strategy: "
                f"product={product_type} → cat={category}/{subcategory}, user={user_id}")

    # Load from MongoDB
    result = load_strategy_from_mongodb(user_id=user_id)
    all_records: List[Dict] = result.get("vendors", [])

    # Filter by category
    category_records = []
    for rec in all_records:
        rec_cat = rec.get("category", "").strip()
        rec_sub = rec.get("subcategory", "").strip()
        if category and rec_cat.lower() != category.lower():
            continue
        if subcategory and rec_sub and rec_sub.lower() != subcategory.lower():
            continue
        category_records.append(rec)

    # Optional strategy keyword filter
    if strategy and category_records:
        strat_lower = strategy.lower()
        matched = [r for r in category_records if strat_lower in r.get("strategy", "").lower()]
        if matched:
            category_records = matched
        # else keep all (don't return empty on mismatch)

    # Build vendor lookup
    vendor_map: Dict[str, List[Dict]] = {}
    for rec in category_records:
        key = rec.get("vendor_name", "").strip().lower()
        if key:
            vendor_map.setdefault(key, []).append(rec)

    # Decide which vendors to score
    if not available_vendors and category_records:
        available_vendors = list({r.get("vendor_name", "").strip() for r in category_records})

    scored: List[Dict] = []
    for vendor in (available_vendors or []):
        vendor_lower = vendor.lower().strip()
        matches = []
        for k, recs in vendor_map.items():
            if vendor_lower in k or k in vendor_lower:
                matches.extend(recs)

        if not matches:
            scored.append({
                "vendor": vendor,
                "strategy": "No specific strategy defined",
                "priority_score": 5,
                "category_match": False,
                "refinery_match": False,
                "strategy_match": None,
                "comments": "",
            })
            continue

        best = matches[0]
        priority = _calculate_priority_score(best, refinery)
        refinery_match = refinery and refinery.lower() in best.get("refinery", "").lower()
        strategy_match = None
        if strategy:
            strategy_match = strategy.lower() in best.get("strategy", "").lower()
            if strategy_match:
                priority += 10

        scored.append({
            "vendor": vendor,
            "strategy": best.get("strategy", ""),
            "priority_score": priority,
            "category_match": True,
            "refinery_match": bool(refinery_match),
            "strategy_match": strategy_match,
            "comments": best.get("comments", best.get("additional_comments", "")),
            "vendor_id": best.get("vendor_id", ""),
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "filtered_vendors": scored,
        "excluded_vendors": [],
        "category": category,
        "subcategory": subcategory,
        "total_strategy_entries": len(all_records),
    }


def get_vendor_strategy_info(
    vendor_name: str,
    category: Optional[str] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get strategy information for a specific vendor from MongoDB.

    Args:
        vendor_name:  Vendor to look up
        category:     Optional category filter
        user_id:      User ID for isolation

    Returns:
        Vendor strategy record or empty dict
    """
    result = load_strategy_from_mongodb(user_id=user_id)
    all_records = result.get("vendors", [])

    vendor_lower = vendor_name.lower().strip()
    matches = [
        r for r in all_records
        if vendor_lower in r.get("vendor_name", "").lower()
        or r.get("vendor_name", "").lower() in vendor_lower
    ]

    if not matches:
        return {}

    if category:
        for rec in matches:
            if rec.get("category", "").lower() == category.lower():
                return rec

    return matches[0]


# Singleton instance (replaces get_strategy_filter())
_strategy_filter_instance = None


def get_strategy_filter():
    """
    Return a lightweight proxy object with a .filter_vendors_for_product() method,
    for backward compatibility with callers that used StrategyCSVFilter instances.
    """
    global _strategy_filter_instance
    if _strategy_filter_instance is None:
        _strategy_filter_instance = _StrategyMongoProxy()
    return _strategy_filter_instance


class _StrategyMongoProxy:
    """
    Thin proxy that mirrors the old StrategyCSVFilter interface.
    All data now comes from MongoDB instead of a CSV file.
    """

    def filter_vendors_for_product(
        self,
        product_type: str,
        available_vendors: Optional[List[str]] = None,
        refinery: Optional[str] = None,
        user_id: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        return filter_vendors_by_strategy(
            product_type=product_type,
            available_vendors=available_vendors,
            refinery=refinery,
            user_id=user_id,
            strategy=strategy,
        )

    def get_vendor_strategy(
        self,
        vendor_name: str,
        category: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        return get_vendor_strategy_info(vendor_name, category, user_id)


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "load_strategy_from_mongodb",
    "filter_vendors_by_strategy",
    "get_vendor_strategy_info",
    "get_strategy_filter",
]
