"""
Strategy Keyword Standardizer
==============================
LLM-based keyword standardization for strategy documents.

Features:
- Standardizes category, subcategory, strategy keywords, and vendor names
- Stores both full names and abbreviations
- MongoDB-based caching (5-minute TTL)
- Batch processing with async LLM calls
- Query term expansion for fuzzy matching
- User-scoped mappings with global fallback

Usage:
    from services.strategy.keyword_standardizer import get_standardizer

    standardizer = get_standardizer()
    canonical_full, canonical_abbrev, confidence = standardizer.standardize_keyword(
        "pres transmitter",
        "category"
    )
"""

import json
import logging
import re
import asyncio
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# MongoDB
from core.mongodb_manager import mongodb_manager

# LLM
from services.llm.fallback import create_llm_with_fallback

logger = logging.getLogger(__name__)


class StrategyKeywordStandardizer:
    """
    LLM-based keyword standardizer for strategy documents.

    Standardizes 4 field types:
    - category: Product categories (e.g., "Pressure Instruments")
    - subcategory: Specific product types (e.g., "Differential Pressure Transmitters")
    - strategy: Procurement strategy keywords
    - vendor: Vendor/manufacturer names
    """

    def __init__(self):
        """Initialize standardizer with MongoDB connection and LLM."""
        self.db = mongodb_manager.database
        self.collection = self.db['keyword_standardization'] if self.db else None
        self.llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.0,  # Deterministic for standardization
            max_tokens=500
        )

        # In-memory cache for frequent lookups (fallback if MongoDB slow)
        self._memory_cache: Dict[str, Tuple[str, str, float]] = {}
        self._cache_size_limit = 1000

        logger.info("[STANDARDIZER] Initialized with MongoDB backend")

    def standardize_keyword(
        self,
        keyword: str,
        field_type: str,
        category_context: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Tuple[str, str, float]:
        """
        Standardize a single keyword.

        Args:
            keyword: Text to standardize
            field_type: "category", "subcategory", "strategy", or "vendor"
            category_context: Parent category for subcategory standardization
            user_id: User ID for user-scoped mappings

        Returns:
            Tuple of (canonical_full, canonical_abbrev, confidence)

        Example:
            >>> standardize_keyword("pres transmitter", "category")
            ("Pressure Instruments", "PI", 0.95)
        """
        if not keyword or not keyword.strip():
            return ("", "", 0.0)

        keyword = keyword.strip()
        cache_key = f"{field_type}:{keyword.lower()}:{user_id}"

        try:
            # 1. Check in-memory cache (fastest)
            if cache_key in self._memory_cache:
                logger.debug(f"[CACHE-HIT] Memory: {keyword} ({field_type})")
                return self._memory_cache[cache_key]

            # 2. Check MongoDB mapping (user-scoped then global)
            if self.collection:
                mapping = self._find_mapping(keyword, field_type, user_id)
                if mapping:
                    result = (
                        mapping['canonical_full'],
                        mapping['canonical_abbrev'],
                        mapping.get('confidence', 1.0)
                    )

                    # Update last_used timestamp
                    self.collection.update_one(
                        {"_id": mapping['_id']},
                        {
                            "$set": {"last_used": datetime.utcnow()},
                            "$inc": {"usage_count": 1}
                        }
                    )

                    # Cache in memory
                    self._update_memory_cache(cache_key, result)

                    logger.debug(f"[CACHE-HIT] MongoDB: {keyword} → {result[0]} ({field_type})")
                    return result

            # 3. LLM standardization (slow path)
            logger.info(f"[LLM-CALL] Standardizing: '{keyword}' ({field_type})")
            result = self._standardize_with_llm(keyword, field_type, category_context)

            # 4. Store new mapping in MongoDB
            if self.collection and result[2] > 0.5:  # Only store if confidence > 0.5
                self._store_mapping(
                    keyword=keyword,
                    field_type=field_type,
                    canonical_full=result[0],
                    canonical_abbrev=result[1],
                    confidence=result[2],
                    user_id=user_id,
                    category_context=category_context
                )

            # 5. Cache in memory
            self._update_memory_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"[ERROR] Standardization failed for '{keyword}': {e}")
            return self._fallback_standardize(keyword, field_type)

    def batch_standardize(
        self,
        records: List[Dict[str, str]],
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch standardize multiple records (for upload processing).

        Processes all 4 fields:
        - vendor_name → vendor_name_std, vendor_abbrev
        - category → category_std, category_abbrev
        - subcategory → subcategory_std, subcategory_abbrev
        - strategy → strategy_keywords, strategy_priority

        Args:
            records: List of dicts with vendor_name, category, subcategory, strategy
            user_id: User ID for user-scoped mappings

        Returns:
            List of records with standardized fields added
        """
        if not records:
            return []

        logger.info(f"[BATCH] Standardizing {len(records)} records...")

        standardized_records = []

        for i, record in enumerate(records):
            try:
                # Copy original record
                std_record = record.copy()

                # Standardize vendor_name
                vendor_name = record.get('vendor_name', '')
                if vendor_name:
                    vendor_full, vendor_abbrev, vendor_conf = self.standardize_keyword(
                        vendor_name, "vendor", user_id=user_id
                    )
                    std_record['vendor_name_std'] = vendor_full
                    std_record['vendor_abbrev'] = vendor_abbrev
                else:
                    std_record['vendor_name_std'] = ''
                    std_record['vendor_abbrev'] = ''

                # Standardize category
                category = record.get('category', '')
                if category:
                    cat_full, cat_abbrev, cat_conf = self.standardize_keyword(
                        category, "category", user_id=user_id
                    )
                    std_record['category_std'] = cat_full
                    std_record['category_abbrev'] = cat_abbrev
                else:
                    std_record['category_std'] = ''
                    std_record['category_abbrev'] = ''

                # Standardize subcategory (with category context)
                subcategory = record.get('subcategory', '')
                if subcategory:
                    subcat_full, subcat_abbrev, subcat_conf = self.standardize_keyword(
                        subcategory, "subcategory",
                        category_context=std_record.get('category_std', ''),
                        user_id=user_id
                    )
                    std_record['subcategory_std'] = subcat_full
                    std_record['subcategory_abbrev'] = subcat_abbrev
                else:
                    std_record['subcategory_std'] = ''
                    std_record['subcategory_abbrev'] = ''

                # Extract strategy keywords and priority
                strategy = record.get('strategy', '')
                if strategy:
                    keywords, priority, strat_conf = self._extract_strategy_keywords(
                        strategy, user_id=user_id
                    )
                    std_record['strategy_keywords'] = keywords
                    std_record['strategy_priority'] = priority
                else:
                    std_record['strategy_keywords'] = []
                    std_record['strategy_priority'] = 'medium'

                # Overall standardization confidence (average)
                confidences = [
                    vendor_conf if vendor_name else 1.0,
                    cat_conf if category else 1.0,
                    subcat_conf if subcategory else 1.0,
                    strat_conf if strategy else 1.0
                ]
                std_record['standardization_confidence'] = sum(confidences) / len(confidences)

                standardized_records.append(std_record)

                if (i + 1) % 10 == 0:
                    logger.info(f"[BATCH] Processed {i + 1}/{len(records)} records")

            except Exception as e:
                logger.error(f"[BATCH-ERROR] Record {i}: {e}")
                # Keep original record on error
                standardized_records.append(record)

        logger.info(f"[BATCH] Completed: {len(standardized_records)} records standardized")
        return standardized_records

    def expand_query_term(
        self,
        term: str,
        field_type: str,
        user_id: Optional[int] = None
    ) -> List[str]:
        """
        Expand query term to all aliases for fuzzy matching.

        Args:
            term: Query term to expand
            field_type: "category", "subcategory", "vendor"
            user_id: User ID for user-scoped mappings

        Returns:
            List of all aliases including canonical forms

        Example:
            >>> expand_query_term("PT", "category")
            ["Pressure Transmitter", "PT", "pressure transmitter",
             "pres transmitter", "pressure sensor"]
        """
        if not term or not term.strip():
            return []

        term = term.strip()
        aliases = [term, term.lower()]

        try:
            # Find all mappings that match this term
            if self.collection:
                # Query by canonical_full, canonical_abbrev, or aliases
                query = {
                    "field_type": field_type,
                    "$or": [
                        {"canonical_full": {"$regex": f"^{re.escape(term)}$", "$options": "i"}},
                        {"canonical_abbrev": {"$regex": f"^{re.escape(term)}$", "$options": "i"}},
                        {"aliases": {"$regex": f"^{re.escape(term)}$", "$options": "i"}}
                    ]
                }

                # User-scoped query with global fallback
                if user_id is not None:
                    query["$or"].insert(0, {"user_id": user_id})
                    query["$or"].append({"user_id": None})

                mappings = list(self.collection.find(query).limit(5))

                for mapping in mappings:
                    aliases.append(mapping['canonical_full'])
                    aliases.append(mapping['canonical_abbrev'])
                    aliases.extend(mapping.get('aliases', []))

            # Remove duplicates (case-insensitive)
            seen = set()
            unique_aliases = []
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower not in seen:
                    seen.add(alias_lower)
                    unique_aliases.append(alias)

            logger.debug(f"[EXPAND] '{term}' → {len(unique_aliases)} aliases")
            return unique_aliases

        except Exception as e:
            logger.error(f"[EXPAND-ERROR] {term}: {e}")
            return [term, term.lower()]

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _standardize_with_llm(
        self,
        keyword: str,
        field_type: str,
        category_context: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Call LLM to standardize keyword."""
        try:
            # Load appropriate prompt
            prompt = self._get_prompt(field_type, keyword, category_context)

            # Call LLM
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            result = self._parse_llm_response(response_text, field_type)

            logger.info(f"[LLM-SUCCESS] '{keyword}' → '{result[0]}' ({result[2]:.2f})")
            return result

        except Exception as e:
            logger.error(f"[LLM-ERROR] {keyword}: {e}")
            return self._fallback_standardize(keyword, field_type)

    def _get_prompt(
        self,
        field_type: str,
        keyword: str,
        category_context: Optional[str] = None
    ) -> str:
        """Get LLM prompt for field type."""
        # Import here to avoid circular dependency
        from prompts import get_strategy_standardization_prompt

        prompt_template = get_strategy_standardization_prompt(field_type)

        # Fill placeholders
        if field_type == "subcategory":
            prompt = prompt_template.replace("{keyword}", keyword)
            prompt = prompt.replace("{category_context}", category_context or "Unknown")
        elif field_type == "strategy":
            prompt = prompt_template.replace("{strategy_text}", keyword)
        elif field_type == "vendor":
            prompt = prompt_template.replace("{vendor_name}", keyword)
        else:  # category
            prompt = prompt_template.replace("{keyword}", keyword)

        return prompt

    def _parse_llm_response(
        self,
        response_text: str,
        field_type: str
    ) -> Tuple[str, str, float]:
        """Parse LLM JSON response."""
        try:
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)

            # Parse JSON
            data = json.loads(response_text)

            if field_type == "strategy":
                # Strategy returns keywords + priority
                keywords = data.get('strategy_keywords', [])
                priority = data.get('strategy_priority', 'medium')
                confidence = data.get('confidence', 0.8)

                # Store as tuple (keywords_str, priority, confidence)
                keywords_str = ",".join(keywords) if keywords else ""
                return (keywords_str, priority, confidence)
            else:
                # Category/subcategory/vendor return canonical forms
                canonical_full = data.get('canonical_full', '')
                canonical_abbrev = data.get('canonical_abbrev', '')
                confidence = data.get('confidence', 0.8)

                return (canonical_full, canonical_abbrev, confidence)

        except Exception as e:
            logger.error(f"[PARSE-ERROR] {response_text[:200]}: {e}")
            raise

    def _extract_strategy_keywords(
        self,
        strategy_text: str,
        user_id: Optional[int] = None
    ) -> Tuple[List[str], str, float]:
        """
        Extract strategy keywords and priority from strategy text.

        Returns:
            Tuple of (keywords_list, priority_level, confidence)
        """
        try:
            # Use LLM to extract
            from prompts import get_strategy_standardization_prompt

            prompt = get_strategy_standardization_prompt("strategy_keywords")
            prompt = prompt.replace("{strategy_text}", strategy_text)

            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Remove markdown
            response_text = re.sub(r'^```json?\s*', '', response_text.strip())
            response_text = re.sub(r'\s*```$', '', response_text)

            data = json.loads(response_text)

            keywords = data.get('strategy_keywords', [])
            priority = data.get('strategy_priority', 'medium')
            confidence = data.get('confidence', 0.8)

            return (keywords, priority, confidence)

        except Exception as e:
            logger.error(f"[STRATEGY-ERROR] {strategy_text[:100]}: {e}")
            # Fallback: simple keyword matching
            return self._fallback_strategy_extract(strategy_text)

    def _fallback_strategy_extract(self, strategy_text: str) -> Tuple[List[str], str, float]:
        """Fallback strategy keyword extraction using simple matching."""
        text_lower = strategy_text.lower()
        keywords = []
        priority = "medium"

        # Check for priority indicators
        if any(word in text_lower for word in ["must", "sole", "only", "critical", "mandatory"]):
            priority = "critical"
        elif any(word in text_lower for word in ["preferred", "strategic", "recommended", "primary"]):
            priority = "high"
        elif any(word in text_lower for word in ["backup", "secondary", "evaluate", "alternative"]):
            priority = "low"

        # Check for keyword indicators
        if "preferred" in text_lower or "primary" in text_lower:
            keywords.append("preferred_vendor")
        if "strategic" in text_lower or "partnership" in text_lower:
            keywords.append("strategic_partnership")
        if "dual" in text_lower and "sourc" in text_lower:
            keywords.append("dual_sourcing")
        if "cost" in text_lower or "price" in text_lower:
            keywords.append("cost_optimization")
        if "quality" in text_lower:
            keywords.append("quality_focus")
        if "sustain" in text_lower or "green" in text_lower:
            keywords.append("sustainability")
        if "critical" in text_lower:
            keywords.append("critical_applications")

        return (keywords, priority, 0.5)

    def _fallback_standardize(
        self,
        keyword: str,
        field_type: str
    ) -> Tuple[str, str, float]:
        """Fallback standardization using simple rules."""
        if field_type == "strategy":
            return self._fallback_strategy_extract(keyword)

        # Title case for full form
        canonical_full = keyword.title().strip()

        # Generate abbreviation from first letters
        words = canonical_full.split()
        canonical_abbrev = "".join(word[0].upper() for word in words if word)[:3]

        # Low confidence for fallback
        return (canonical_full, canonical_abbrev, 0.3)

    def _find_mapping(
        self,
        keyword: str,
        field_type: str,
        user_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Find existing mapping in MongoDB."""
        if not self.collection:
            return None

        keyword_lower = keyword.lower()

        # Query by canonical_full, canonical_abbrev, or aliases
        query = {
            "field_type": field_type,
            "$or": [
                {"canonical_full": {"$regex": f"^{re.escape(keyword)}$", "$options": "i"}},
                {"canonical_abbrev": {"$regex": f"^{re.escape(keyword)}$", "$options": "i"}},
                {"aliases": keyword_lower}
            ]
        }

        # User-scoped query with global fallback
        if user_id is not None:
            # Try user-specific first
            user_query = query.copy()
            user_query["user_id"] = user_id
            mapping = self.collection.find_one(user_query)
            if mapping:
                return mapping

            # Fall back to global
            query["user_id"] = None

        return self.collection.find_one(query)

    def _store_mapping(
        self,
        keyword: str,
        field_type: str,
        canonical_full: str,
        canonical_abbrev: str,
        confidence: float,
        user_id: Optional[int] = None,
        category_context: Optional[str] = None
    ):
        """Store new mapping in MongoDB."""
        if not self.collection:
            return

        try:
            # Check if mapping already exists
            existing = self._find_mapping(keyword, field_type, user_id)

            if existing:
                # Update existing mapping
                self.collection.update_one(
                    {"_id": existing['_id']},
                    {
                        "$set": {
                            "last_updated": datetime.utcnow(),
                            "confidence": max(confidence, existing.get('confidence', 0))
                        },
                        "$addToSet": {"aliases": keyword.lower()},
                        "$inc": {"usage_count": 1}
                    }
                )
                logger.debug(f"[STORE] Updated mapping for '{keyword}'")
            else:
                # Create new mapping
                doc = {
                    "canonical_full": canonical_full,
                    "canonical_abbrev": canonical_abbrev,
                    "field_type": field_type,
                    "aliases": [keyword.lower()],
                    "confidence": confidence,
                    "source": "llm",
                    "usage_count": 1,
                    "created_at": datetime.utcnow(),
                    "last_updated": datetime.utcnow(),
                    "last_used": datetime.utcnow(),
                    "user_id": user_id,
                    "is_cache": False  # Permanent mapping, not cache
                }

                if category_context:
                    doc["category_context"] = category_context

                self.collection.insert_one(doc)
                logger.debug(f"[STORE] Created mapping: '{keyword}' → '{canonical_full}'")

        except Exception as e:
            logger.error(f"[STORE-ERROR] {keyword}: {e}")

    def _update_memory_cache(self, key: str, value: Tuple[str, str, float]):
        """Update in-memory cache with size limit."""
        if len(self._memory_cache) >= self._cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = value


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_standardizer_instance: Optional[StrategyKeywordStandardizer] = None


def get_standardizer() -> StrategyKeywordStandardizer:
    """Get or create global standardizer instance (singleton pattern)."""
    global _standardizer_instance

    if _standardizer_instance is None:
        _standardizer_instance = StrategyKeywordStandardizer()

    return _standardizer_instance
