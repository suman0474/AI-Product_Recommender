"""
Database Indexes Setup
======================
Creates optimal indexes for all MongoDB collections.

Features:
- Performance indexes for fast queries
- TTL indexes for automatic cache expiration
- Compound indexes for common query patterns
- Unique indexes where needed

Usage:
    python -m core.db_indexes

Or from code:
    from common.core.db_indexes import ensure_indexes
    ensure_indexes()
"""
import logging
import os
import sys
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.core.mongodb_manager import mongodb_manager

logger = logging.getLogger(__name__)


def _safe_create_index(collection, keys, **kwargs):
    """
    Create a MongoDB index, silently skipping conflicts where the same key
    pattern already exists under a different name (error code 85 —
    IndexOptionsConflict).  Any other error is re-raised.
    """
    try:
        collection.create_index(keys, **kwargs)
    except Exception as exc:
        # pymongo.errors.OperationFailure code 85 = IndexOptionsConflict
        code = getattr(exc, 'code', None)
        if code == 85:
            name = kwargs.get('name', str(keys))
            logger.warning(
                "[db_indexes] Skipping index '%s' — already exists with a different name "
                "(IndexOptionsConflict). This is harmless.", name
            )
        else:
            raise


def ensure_indexes():
    """
    Create all required indexes for optimal performance.

    This function is idempotent - safe to run multiple times.
    MongoDB ignores index creation if index already exists.
    """
    db = mongodb_manager.database

    if db is None:
        print("❌ MongoDB not connected - cannot create indexes")
        return False

    print("\n" + "="*60)
    print("   CREATING DATABASE INDEXES")
    print("="*60)

    try:
        # ============================================================
        # COLLECTION 1: specs (Product Schemas)
        # ============================================================
        logger.info("specs collection:")
        specs = db['specs']

        _safe_create_index(specs, [("product_type", 1)], name="idx_product_type")
        _safe_create_index(specs, [("metadata.normalized_product_type", 1)], name="idx_normalized_product_type")
        _safe_create_index(specs, [("product_type", "text"), ("metadata.product_type", "text")], name="idx_product_type_text")

        # ============================================================
        # COLLECTION 2: vendors (Product Catalogs)
        # ============================================================
        logger.info("vendors collection:")
        vendors = db['vendors']

        _safe_create_index(vendors, [("product_type", 1), ("vendor_name", 1)], name="idx_product_vendor")
        _safe_create_index(vendors, [("vendor_name", 1)], name="idx_vendor_name")
        _safe_create_index(vendors, [("model_family", 1)], name="idx_model_family")
        _safe_create_index(vendors, [("vendor_name_normalized", 1)], name="idx_vendor_name_normalized")

        # ============================================================
        # COLLECTION 3: images (Product Image Cache)
        # ============================================================
        logger.info("images collection:")
        images = db['images']

        _safe_create_index(images, [("vendor_name_normalized", 1), ("model_family_normalized", 1)], name="idx_image_lookup", unique=True, sparse=True)
        _safe_create_index(images, [("product_type_normalized", 1)], name="idx_image_product_type")
        _safe_create_index(images, "createdAt", name="idx_image_ttl", expireAfterSeconds=60*60*24*90)

        # ============================================================
        # COLLECTION 4: generic_images (AI-Generated Images)
        # ============================================================
        logger.info("generic_images collection:")
        generic_images = db['generic_images']

        _safe_create_index(generic_images, "product_type_normalized", name="idx_generic_product_type", unique=True)
        _safe_create_index(generic_images, "createdAt", name="idx_generic_ttl", expireAfterSeconds=60*60*24*90)

        # ============================================================
        # COLLECTION 5: vendor_logos (Vendor Logo Cache)
        # ============================================================
        logger.info("vendor_logos collection:")
        vendor_logos = db['vendor_logos']

        _safe_create_index(vendor_logos, "vendor_name_normalized", name="idx_logo_vendor", unique=True)

        # ============================================================
        # COLLECTION 6: advanced_parameters (Parameter Cache)
        # ============================================================
        logger.info("advanced_parameters collection:")
        advanced_parameters = db['advanced_parameters']

        _safe_create_index(advanced_parameters, [("normalized_product_type", 1)], name="idx_param_product_type", unique=True)
        _safe_create_index(advanced_parameters, "created_at", name="idx_param_ttl", expireAfterSeconds=60*60*24*30)

        # ============================================================
        # COLLECTION 7: user_projects (Project Metadata)
        # ============================================================
        logger.info("user_projects collection:")
        user_projects = db['user_projects']

        _safe_create_index(user_projects, [("user_id", 1), ("project_status", 1), ("updated_at", -1)], name="idx_user_projects_list")
        _safe_create_index(user_projects, [("user_id", 1), ("project_name", 1)], name="idx_user_project_name")
        _safe_create_index(user_projects, [("user_id", 1), ("product_type", 1)], name="idx_user_product_type")

        # ============================================================
        # COLLECTION 8: strategy (Strategy Documents)
        # ============================================================
        logger.info("strategy collection:")
        strategy = db['stratergy']

        _safe_create_index(strategy, [("user_id", 1), ("uploaded_at", -1)], name="idx_strategy_user_date")
        _safe_create_index(strategy, [("user_id", 1), ("filename", 1)], name="idx_strategy_filename")

        # ============================================================
        # COLLECTION 9: keyword_standardization (Strategy Keywords)
        # ============================================================
        logger.info("keyword_standardization collection:")
        keyword_standardization = db['keyword_standardization']

        _safe_create_index(keyword_standardization, [("canonical_full", 1), ("field_type", 1)], name="idx_keyword_canonical_full")
        _safe_create_index(keyword_standardization, [("canonical_abbrev", 1), ("field_type", 1)], name="idx_keyword_canonical_abbrev")
        _safe_create_index(keyword_standardization, [("aliases", 1), ("field_type", 1)], name="idx_keyword_aliases")
        _safe_create_index(keyword_standardization, [("user_id", 1), ("field_type", 1)], name="idx_keyword_user_field")
        _safe_create_index(keyword_standardization, [("last_used", 1)], name="idx_keyword_cache_ttl", expireAfterSeconds=300, partialFilterExpression={"is_cache": True})

        # ============================================================
        # COLLECTION 10: standards (Engineering Standards)
        # ============================================================
        logger.info("standards collection:")
        standards = db['standards']

        _safe_create_index(standards, [("user_id", 1), ("uploaded_at", -1)], name="idx_standards_user_date")
        _safe_create_index(standards, [("user_id", 1), ("filename", 1)], name="idx_standards_filename")

        # ============================================================
        # COLLECTION 11: documents (General Documents)
        # ============================================================
        logger.info("documents collection:")
        documents = db['documents']

        _safe_create_index(documents, [("product_type", 1), ("vendor_name", 1)], name="idx_doc_product_vendor")
        _safe_create_index(documents, [("upload_date", -1)], name="idx_doc_upload_date")

        logger.info("[db_indexes] All indexes created/verified successfully")
        return True

    except Exception as e:
        logger.error("Error creating indexes: %s", e, exc_info=True)
        return False


def list_indexes():
    """List all indexes in all collections"""
    db = mongodb_manager.database

    if db is None:
        print("❌ MongoDB not connected")
        return

    print("\n" + "="*60)
    print("   EXISTING INDEXES")
    print("="*60)

    collections = [
        'specs', 'vendors', 'images', 'generic_images', 'vendor_logos',
        'advanced_parameters', 'user_projects', 'stratergy', 'standards', 'documents'
    ]

    for collection_name in collections:
        collection = db[collection_name]
        indexes = list(collection.list_indexes())

        print(f"\n📦 {collection_name}:")
        for idx in indexes:
            name = idx.get('name', 'unknown')
            keys = idx.get('key', {})
            unique = idx.get('unique', False)
            ttl = idx.get('expireAfterSeconds')

            desc = f"   - {name}: {dict(keys)}"
            if unique:
                desc += " (unique)"
            if ttl:
                days = ttl / (60*60*24)
                desc += f" (TTL: {days:.0f} days)"
            print(desc)


def drop_all_indexes():
    """
    Drop all custom indexes (keeps _id index).

    Use with caution! This will slow down queries until indexes are recreated.
    """
    db = mongodb_manager.database

    if db is None:
        print("❌ MongoDB not connected")
        return False

    print("\n⚠️  DROPPING ALL CUSTOM INDEXES")
    print("This will NOT affect data, only query performance.\n")

    collections = [
        'specs', 'vendors', 'images', 'generic_images', 'vendor_logos',
        'advanced_parameters', 'user_projects', 'stratergy', 'standards', 'documents'
    ]

    for collection_name in collections:
        try:
            collection = db[collection_name]
            collection.drop_indexes()
            print(f"   ✅ Dropped indexes for: {collection_name}")
        except Exception as e:
            print(f"   ❌ Error dropping indexes for {collection_name}: {e}")

    print("\n✅ Done. Run ensure_indexes() to recreate.")
    return True


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about indexes"""
    db = mongodb_manager.database

    if db is None:
        return {'error': 'MongoDB not connected'}

    stats = {
        'total_indexes': 0,
        'ttl_indexes': 0,
        'unique_indexes': 0,
        'collections': {}
    }

    collections = [
        'specs', 'vendors', 'images', 'generic_images', 'vendor_logos',
        'advanced_parameters', 'user_projects', 'stratergy', 'standards', 'documents'
    ]

    for collection_name in collections:
        collection = db[collection_name]
        indexes = list(collection.list_indexes())

        collection_stats = {
            'count': len(indexes) - 1,  # Exclude _id
            'indexes': []
        }

        for idx in indexes:
            if idx['name'] == '_id_':
                continue

            idx_info = {
                'name': idx['name'],
                'keys': dict(idx['key']),
                'unique': idx.get('unique', False),
                'ttl': idx.get('expireAfterSeconds')
            }
            collection_stats['indexes'].append(idx_info)
            stats['total_indexes'] += 1

            if idx.get('unique'):
                stats['unique_indexes'] += 1
            if idx.get('expireAfterSeconds'):
                stats['ttl_indexes'] += 1

        stats['collections'][collection_name] = collection_stats

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MongoDB Index Management')
    parser.add_argument('--list', action='store_true', help='List all indexes')
    parser.add_argument('--drop', action='store_true', help='Drop all custom indexes')
    parser.add_argument('--stats', action='store_true', help='Show index statistics')

    args = parser.parse_args()

    if args.list:
        list_indexes()
    elif args.drop:
        response = input("⚠️ This will drop all custom indexes. Continue? (yes/no): ")
        if response.lower() == 'yes':
            drop_all_indexes()
    elif args.stats:
        import json
        stats = get_index_stats()
        print(json.dumps(stats, indent=2))
    else:
        # Default: create indexes
        ensure_indexes()
