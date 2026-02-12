"""
Strategy Document Tree Index
==============================
Hierarchical tree-based index for fast retrieval of 2000+ vendor 
strategy documents from Azure Blob Storage.

Tree Structure:
    Root
    ├── Category 1
    │   ├── Subcategory A
    │   │   ├── V00001
    │   │   └── V00002
    │   └── Subcategory B
    │       └── V00003
    └── Category 2
        └── Subcategory C
            └── V00004

Performance:
    - O(1) vendor lookup by ID
    - O(1) category/subcategory listing
    - O(n) full-text search within category (n = vendors in category)

Usage:
    from core.strategy_index import strategy_index

    # Get a vendor's blob path
    path = strategy_index.get_blob_path("V00001")

    # Get all vendors in a category
    vendors = strategy_index.list_by_category("Pressure Instruments")

    # Search by keyword
    results = strategy_index.search("Emerson", category="Flow Instruments")
"""

import os
import csv
import logging
import threading
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger(__name__)


class TreeNode:
    """A node in the strategy document tree."""

    def __init__(self, name: str, level: str = "root"):
        self.name = name
        self.level = level  # 'root', 'category', 'subcategory', 'vendor'
        self.children: Dict[str, 'TreeNode'] = {}
        self.metadata: Dict[str, Any] = {}

    def add_child(self, key: str, node: 'TreeNode') -> 'TreeNode':
        """Add a child node. Returns the child (existing or new)."""
        if key not in self.children:
            self.children[key] = node
        return self.children[key]

    def get_child(self, key: str) -> Optional['TreeNode']:
        """Get a child node by key."""
        return self.children.get(key)

    def list_children(self) -> List[str]:
        """List all child keys."""
        return list(self.children.keys())

    def count_descendants(self) -> int:
        """Count all descendants (recursive)."""
        count = len(self.children)
        for child in self.children.values():
            count += child.count_descendants()
        return count

    def __repr__(self):
        return f"TreeNode({self.name}, level={self.level}, children={len(self.children)})"


class StrategyDocumentIndex:
    """
    Tree-based index for fast vendor strategy document retrieval.

    The index is built from the strategy CSV and provides:
    - O(1) lookup by vendor ID
    - O(1) category/subcategory listing
    - Hierarchical blob path resolution
    """

    _instance: Optional['StrategyDocumentIndex'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._tree = TreeNode("root", level="root")
            self._vendor_index: Dict[str, Dict[str, str]] = {}  # vendor_id -> {category, subcategory, blob_path, ...}
            self._category_vendors: Dict[str, Set[str]] = {}  # category -> set of vendor_ids
            self._subcategory_vendors: Dict[str, Dict[str, Set[str]]] = {}  # category -> subcategory -> set of vendor_ids
            self._loaded = False
            self._csv_path = None
            self._build_lock = threading.Lock()

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in blob paths."""
        return "".join([c if c.isalnum() else "_" for c in name])

    def build_from_csv(self, csv_path: str) -> bool:
        """
        Build the tree index from the strategy CSV file.

        Args:
            csv_path: Path to the instrumentation_procurement_strategy.csv

        Returns:
            True if index was built successfully
        """
        with self._build_lock:
            if self._loaded and self._csv_path == csv_path:
                return True

            try:
                logger.info(f"[StrategyIndex] Building tree index from: {csv_path}")

                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if not rows:
                    logger.warning("[StrategyIndex] CSV is empty.")
                    return False

                # Reset
                self._tree = TreeNode("root", level="root")
                self._vendor_index.clear()
                self._category_vendors.clear()
                self._subcategory_vendors.clear()

                for row in rows:
                    vendor_id = row.get('vendor ID', '').strip()
                    vendor_name = row.get('vendor name', '').strip()
                    category = row.get('category', '').strip()
                    subcategory = row.get('subcategory', '').strip()
                    strategy = row.get('strategy', '').strip()
                    refinery = row.get('refinery', '').strip()
                    comments = row.get('additional comments', '').strip()
                    owner = row.get('owner name', '').strip()

                    if not vendor_id:
                        continue

                    # Build tree path
                    safe_category = self._sanitize_name(category)
                    safe_subcategory = self._sanitize_name(subcategory)
                    safe_vendor = self._sanitize_name(vendor_id)

                    # Add to tree
                    cat_node = self._tree.add_child(
                        category,
                        TreeNode(category, level="category")
                    )
                    sub_node = cat_node.add_child(
                        subcategory,
                        TreeNode(subcategory, level="subcategory")
                    )
                    vendor_node = sub_node.add_child(
                        vendor_id,
                        TreeNode(vendor_id, level="vendor")
                    )

                    # Store metadata on vendor node
                    vendor_node.metadata = {
                        'vendor_id': vendor_id,
                        'vendor_name': vendor_name,
                        'category': category,
                        'subcategory': subcategory,
                        'strategy': strategy,
                        'refinery': refinery,
                        'additional_comments': comments,
                        'owner_name': owner,
                    }

                    # Build blob path (hierarchical)
                    blob_path = f"{safe_category}/{safe_subcategory}/{safe_vendor}.txt"

                    # Index: vendor_id -> full info
                    self._vendor_index[vendor_id] = {
                        'vendor_id': vendor_id,
                        'vendor_name': vendor_name,
                        'category': category,
                        'subcategory': subcategory,
                        'strategy': strategy,
                        'refinery': refinery,
                        'additional_comments': comments,
                        'owner_name': owner,
                        'blob_path': blob_path,
                    }

                    # Index: category -> vendors
                    self._category_vendors.setdefault(category, set()).add(vendor_id)

                    # Index: category -> subcategory -> vendors
                    self._subcategory_vendors.setdefault(category, {}).setdefault(subcategory, set()).add(vendor_id)

                self._loaded = True
                self._csv_path = csv_path

                cat_count = len(self._tree.children)
                sub_count = sum(len(c.children) for c in self._tree.children.values())
                vendor_count = len(self._vendor_index)

                logger.info(
                    f"[StrategyIndex] Tree built: {cat_count} categories, "
                    f"{sub_count} subcategories, {vendor_count} vendors"
                )
                return True

            except Exception as e:
                logger.error(f"[StrategyIndex] Failed to build index: {e}")
                return False

    def build_from_blob(self, container_key: str = 'strategy_documents') -> bool:
        """
        Build the tree index by downloading the CSV from Azure Blob Storage.

        Args:
            container_key: The blob container key

        Returns:
            True if index was built successfully
        """
        try:
            from core.azure_blob_file_manager import AzureBlobFileManager
            import io
            import tempfile

            blob_manager = AzureBlobFileManager()
            blob_name = "instrumentation_procurement_strategy.csv"
            container_name = blob_manager.CONTAINERS.get(container_key, container_key)

            logger.info(f"[StrategyIndex] Downloading CSV from blob: {blob_name}")
            content_bytes = blob_manager.download_file(blob_name, container_name)

            if not content_bytes:
                logger.error(f"[StrategyIndex] CSV not found in blob storage.")
                return False

            # Write to a temp file so build_from_csv can read it
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            result = self.build_from_csv(tmp_path)

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            return result

        except Exception as e:
            logger.error(f"[StrategyIndex] Failed to build from blob: {e}")
            return False

    def ensure_loaded(self) -> bool:
        """Ensure the index is loaded from Azure Blob Storage."""
        if self._loaded:
            return True

        # Load from Azure Blob Storage (PRIMARY SOURCE)
        logger.info("[StrategyIndex] Loading from Azure Blob Storage...")
        return self.build_from_blob()

    # ============================================================
    # Lookup Methods
    # ============================================================

    def get_blob_path(self, vendor_id: str) -> Optional[str]:
        """
        O(1) lookup: Get the blob path for a vendor ID.

        Args:
            vendor_id: e.g., "V00001"

        Returns:
            Blob path like "Pressure_Instruments/Pressure_Gauges/V00001.txt"
        """
        self.ensure_loaded()
        info = self._vendor_index.get(vendor_id)
        return info['blob_path'] if info else None

    def get_vendor_info(self, vendor_id: str) -> Optional[Dict[str, str]]:
        """
        O(1) lookup: Get full metadata for a vendor.

        Args:
            vendor_id: e.g., "V00001"

        Returns:
            Dict with vendor_id, vendor_name, category, subcategory, strategy, etc.
        """
        self.ensure_loaded()
        return self._vendor_index.get(vendor_id)

    def list_categories(self) -> List[str]:
        """List all categories."""
        self.ensure_loaded()
        return sorted(self._category_vendors.keys())

    def list_subcategories(self, category: str) -> List[str]:
        """List subcategories for a given category."""
        self.ensure_loaded()
        subs = self._subcategory_vendors.get(category, {})
        return sorted(subs.keys())

    def list_by_category(self, category: str) -> List[Dict[str, str]]:
        """
        Get all vendors in a category.

        Args:
            category: e.g., "Pressure Instruments"

        Returns:
            List of vendor info dicts
        """
        self.ensure_loaded()
        vendor_ids = self._category_vendors.get(category, set())
        return [self._vendor_index[vid] for vid in sorted(vendor_ids) if vid in self._vendor_index]

    def list_by_subcategory(self, category: str, subcategory: str) -> List[Dict[str, str]]:
        """
        Get all vendors in a specific subcategory.

        Args:
            category: e.g., "Pressure Instruments"
            subcategory: e.g., "Pressure Gauges"

        Returns:
            List of vendor info dicts
        """
        self.ensure_loaded()
        vendor_ids = self._subcategory_vendors.get(category, {}).get(subcategory, set())
        return [self._vendor_index[vid] for vid in sorted(vendor_ids) if vid in self._vendor_index]

    def search(self, keyword: str, category: str = None, subcategory: str = None) -> List[Dict[str, str]]:
        """
        Search vendors by keyword, optionally filtered by category/subcategory.

        Searches across: vendor_name, strategy, additional_comments, refinery

        Args:
            keyword: Search term (case-insensitive)
            category: Optional category filter
            subcategory: Optional subcategory filter

        Returns:
            List of matching vendor info dicts
        """
        self.ensure_loaded()
        keyword_lower = keyword.lower()

        # Determine the search space
        if category and subcategory:
            candidates = self.list_by_subcategory(category, subcategory)
        elif category:
            candidates = self.list_by_category(category)
        else:
            candidates = list(self._vendor_index.values())

        results = []
        for vendor in candidates:
            searchable = ' '.join([
                vendor.get('vendor_name', ''),
                vendor.get('strategy', ''),
                vendor.get('additional_comments', ''),
                vendor.get('refinery', ''),
                vendor.get('vendor_id', ''),
            ]).lower()

            if keyword_lower in searchable:
                results.append(vendor)

        return results

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        self.ensure_loaded()
        return {
            'total_vendors': len(self._vendor_index),
            'total_categories': len(self._category_vendors),
            'total_subcategories': sum(
                len(subs) for subs in self._subcategory_vendors.values()
            ),
            'categories': {
                cat: {
                    'vendor_count': len(vendors),
                    'subcategories': {
                        sub: len(sub_vendors)
                        for sub, sub_vendors in self._subcategory_vendors.get(cat, {}).items()
                    }
                }
                for cat, vendors in self._category_vendors.items()
            },
            'loaded': self._loaded,
        }

    def get_tree_display(self, max_vendors_per_sub: int = 3) -> str:
        """
        Get a human-readable tree display.

        Args:
            max_vendors_per_sub: Max vendors to show per subcategory

        Returns:
            Formatted tree string
        """
        self.ensure_loaded()
        lines = ["Strategy Document Tree", "=" * 40]

        for cat_name in sorted(self._tree.children.keys()):
            cat_node = self._tree.children[cat_name]
            cat_count = len(self._category_vendors.get(cat_name, set()))
            lines.append(f"├── {cat_name} ({cat_count} vendors)")

            sub_names = sorted(cat_node.children.keys())
            for i, sub_name in enumerate(sub_names):
                sub_node = cat_node.children[sub_name]
                sub_count = len(sub_node.children)
                is_last_sub = (i == len(sub_names) - 1)
                prefix = "│   └── " if is_last_sub else "│   ├── "
                lines.append(f"{prefix}{sub_name} ({sub_count} vendors)")

                vendor_ids = sorted(sub_node.children.keys())[:max_vendors_per_sub]
                for j, vid in enumerate(vendor_ids):
                    v_node = sub_node.children[vid]
                    v_name = v_node.metadata.get('vendor_name', '')
                    inner_prefix = "│       " if not is_last_sub else "        "
                    lines.append(f"{inner_prefix}├── {vid}: {v_name}")

                if sub_count > max_vendors_per_sub:
                    inner_prefix = "│       " if not is_last_sub else "        "
                    lines.append(f"{inner_prefix}└── ... and {sub_count - max_vendors_per_sub} more")

        return "\n".join(lines)


# Module-level singleton instance
strategy_index = StrategyDocumentIndex()
