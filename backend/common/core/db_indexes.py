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
import os
import sys
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.core.mongodb_manager import mongodb_manager


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
        print("\n📦 specs collection:")
        specs = db['specs']

        # Index for exact product type match
        specs.create_index([("product_type", 1)], name="idx_product_type")
        print("   ✅ Created: idx_product_type")

        # Index for normalized product type search
        specs.create_index(
            [("metadata.normalized_product_type", 1)],
            name="idx_normalized_product_type"
        )
        print("   ✅ Created: idx_normalized_product_type")

        # Text index for fuzzy search
        specs.create_index(
            [("product_type", "text"), ("metadata.product_type", "text")],
            name="idx_product_type_text"
        )
        print("   ✅ Created: idx_product_type_text")

        # ============================================================
        # COLLECTION 2: vendors (Product Catalogs)
        # ============================================================
        print("\n📦 vendors collection:")
        vendors = db['vendors']

        # Compound index for product + vendor search
        vendors.create_index(
            [("product_type", 1), ("vendor_name", 1)],
            name="idx_product_vendor"
        )
        print("   ✅ Created: idx_product_vendor")

        # Index for vendor-only search
        vendors.create_index([("vendor_name", 1)], name="idx_vendor_name")
        print("   ✅ Created: idx_vendor_name")

        # Index for model family search
        vendors.create_index([("model_family", 1)], name="idx_model_family")
        print("   ✅ Created: idx_model_family")

        # Index for normalized names
        vendors.create_index(
            [("vendor_name_normalized", 1)],
            name="idx_vendor_name_normalized"
        )
        print("   ✅ Created: idx_vendor_name_normalized")

        # ============================================================
        # COLLECTION 3: images (Product Image Cache)
        # ============================================================
        print("\n📦 images collection:")
        images = db['images']

        # Compound unique index for cache lookup
        images.create_index(
            [("vendor_name_normalized", 1), ("model_family_normalized", 1)],
            name="idx_image_lookup",
            unique=True,
            sparse=True
        )
        print("   ✅ Created: idx_image_lookup (unique)")

        # Index for product type filter
        images.create_index(
            [("product_type_normalized", 1)],
            name="idx_image_product_type"
        )
        print("   ✅ Created: idx_image_product_type")

        # TTL index: Auto-delete after 90 days
        images.create_index(
            "createdAt",
            name="idx_image_ttl",
            expireAfterSeconds=60*60*24*90  # 90 days
        )
        print("   ✅ Created: idx_image_ttl (TTL: 90 days)")

        # ============================================================
        # COLLECTION 4: generic_images (AI-Generated Images)
        # ============================================================
        print("\n📦 generic_images collection:")
        generic_images = db['generic_images']

        # Unique index for product type
        generic_images.create_index(
            "product_type_normalized",
            name="idx_generic_product_type",
            unique=True
        )
        print("   ✅ Created: idx_generic_product_type (unique)")

        # TTL index: Auto-delete after 90 days
        generic_images.create_index(
            "createdAt",
            name="idx_generic_ttl",
            expireAfterSeconds=60*60*24*90  # 90 days
        )
        print("   ✅ Created: idx_generic_ttl (TTL: 90 days)")

        # ============================================================
        # COLLECTION 5: vendor_logos (Vendor Logo Cache)
        # ============================================================
        print("\n📦 vendor_logos collection:")
        vendor_logos = db['vendor_logos']

        # Unique index for vendor name
        vendor_logos.create_index(
            "vendor_name_normalized",
            name="idx_logo_vendor",
            unique=True
        )
        print("   ✅ Created: idx_logo_vendor (unique)")

        # ============================================================
        # COLLECTION 6: advanced_parameters (Parameter Cache)
        # ============================================================
        print("\n📦 advanced_parameters collection:")
        advanced_parameters = db['advanced_parameters']

        # Unique index for product type
        advanced_parameters.create_index(
            [("normalized_product_type", 1)],
            name="idx_param_product_type",
            unique=True
        )
        print("   ✅ Created: idx_param_product_type (unique)")

        # TTL index: Auto-delete after 30 days
        advanced_parameters.create_index(
            "created_at",
            name="idx_param_ttl",
            expireAfterSeconds=60*60*24*30  # 30 days
        )
        print("   ✅ Created: idx_param_ttl (TTL: 30 days)")

        # ============================================================
        # COLLECTION 7: user_projects (Project Metadata)
        # ============================================================
        print("\n📦 user_projects collection:")
        user_projects = db['user_projects']

        # Compound index for user's project list
        user_projects.create_index(
            [("user_id", 1), ("project_status", 1), ("updated_at", -1)],
            name="idx_user_projects_list"
        )
        print("   ✅ Created: idx_user_projects_list")

        # Index for user + project name
        user_projects.create_index(
            [("user_id", 1), ("project_name", 1)],
            name="idx_user_project_name"
        )
        print("   ✅ Created: idx_user_project_name")

        # Index for user + product type filter
        user_projects.create_index(
            [("user_id", 1), ("product_type", 1)],
            name="idx_user_product_type"
        )
        print("   ✅ Created: idx_user_product_type")

        # ============================================================
        # COLLECTION 8: strategy (Strategy Documents)
        # ============================================================
        print("\n📦 strategy collection:")
        strategy = db['stratergy']

        # Index for user's documents
        strategy.create_index(
            [("user_id", 1), ("uploaded_at", -1)],
            name="idx_strategy_user_date"
        )
        print("   ✅ Created: idx_strategy_user_date")

        # Index for filename search
        strategy.create_index(
            [("user_id", 1), ("filename", 1)],
            name="idx_strategy_filename"
        )
        print("   ✅ Created: idx_strategy_filename")

        # ============================================================
        # COLLECTION 9: keyword_standardization (Strategy Keywords)
        # ============================================================
        print("\n📦 keyword_standardization collection:")
        keyword_standardization = db['keyword_standardization']

        # Index for canonical_full + field_type lookup
        keyword_standardization.create_index(
            [("canonical_full", 1), ("field_type", 1)],
            name="idx_keyword_canonical_full"
        )
        print("   ✅ Created: idx_keyword_canonical_full")

        # Index for canonical_abbrev + field_type lookup
        keyword_standardization.create_index(
            [("canonical_abbrev", 1), ("field_type", 1)],
            name="idx_keyword_canonical_abbrev"
        )
        print("   ✅ Created: idx_keyword_canonical_abbrev")

        # Index for aliases array + field_type (for query expansion)
        keyword_standardization.create_index(
            [("aliases", 1), ("field_type", 1)],
            name="idx_keyword_aliases"
        )
        print("   ✅ Created: idx_keyword_aliases")

        # Index for user_id + field_type (user-scoped mappings)
        keyword_standardization.create_index(
            [("user_id", 1), ("field_type", 1)],
            name="idx_keyword_user_field"
        )
        print("   ✅ Created: idx_keyword_user_field")

        # TTL index for cache expiration (5 minutes = 300 seconds)
        # Note: Only affects documents with is_cache=true
        keyword_standardization.create_index(
            [("last_used", 1)],
            expireAfterSeconds=300,
            name="idx_keyword_cache_ttl",
            partialFilterExpression={"is_cache": True}
        )
        print("   ✅ Created: idx_keyword_cache_ttl (5-min TTL for cache)")

        # ============================================================
        # COLLECTION 10: standards (Engineering Standards)
        # ============================================================
        print("\n📦 standards collection:")
        standards = db['standards']

        # Index for user's documents
        standards.create_index(
            [("user_id", 1), ("uploaded_at", -1)],
            name="idx_standards_user_date"
        )
        print("   ✅ Created: idx_standards_user_date")

        # Index for filename search
        standards.create_index(
            [("user_id", 1), ("filename", 1)],
            name="idx_standards_filename"
        )
        print("   ✅ Created: idx_standards_filename")

        # ============================================================
        # COLLECTION 11: documents (General Documents)
        # ============================================================
        print("\n📦 documents collection:")
        documents = db['documents']

        # Index for product + vendor filter
        documents.create_index(
            [("product_type", 1), ("vendor_name", 1)],
            name="idx_doc_product_vendor"
        )
        print("   ✅ Created: idx_doc_product_vendor")

        # Index for upload date
        documents.create_index(
            [("upload_date", -1)],
            name="idx_doc_upload_date"
        )
        print("   ✅ Created: idx_doc_upload_date")

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "="*60)
        print("   ✅ ALL INDEXES CREATED SUCCESSFULLY")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Error creating indexes: {str(e)}")
        import traceback
        traceback.print_exc()
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
