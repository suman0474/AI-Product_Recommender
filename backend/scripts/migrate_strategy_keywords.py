"""
Strategy Keywords Migration Script
===================================
Migrate existing strategy documents to add standardized keywords.

This script adds standardized fields to all completed strategy documents in MongoDB:
- vendor_name_std, vendor_abbrev
- category_std, category_abbrev
- subcategory_std, subcategory_abbrev
- strategy_keywords, strategy_priority
- standardization_confidence

Usage:
    python migrate_strategy_keywords.py --dry-run  # Test migration
    python migrate_strategy_keywords.py            # Execute migration
    python migrate_strategy_keywords.py --limit 10 # Migrate only 10 documents

Author: Strategy Keyword Standardization System
Date: 2026-02-11
"""

import argparse
import sys
import os
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mongodb_manager import mongodb_manager
from services.strategy.keyword_standardizer import get_standardizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_strategy_documents(dry_run=True, limit=None):
    """
    Migrate all existing strategy documents to add standardized keywords.

    Args:
        dry_run: If True, only simulate migration without updating database
        limit: Maximum number of documents to migrate (None = all)

    Returns:
        Dict with migration statistics
    """
    logger.info("="*70)
    logger.info("  STRATEGY KEYWORDS MIGRATION")
    logger.info("="*70)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    if limit:
        logger.info(f"Limit: {limit} documents")
    logger.info("")

    # Get MongoDB connection
    db = mongodb_manager.database
    if not db:
        logger.error("❌ MongoDB not connected")
        return {"success": False, "error": "MongoDB not connected"}

    collection = db['stratergy']
    standardizer = get_standardizer()

    # Query only completed documents
    query = {"status": "completed"}

    # Count total documents
    total_docs = collection.count_documents(query)
    logger.info(f"📊 Found {total_docs} completed strategy documents")

    if total_docs == 0:
        logger.warning("⚠️  No completed documents to migrate")
        return {"success": True, "migrated": 0, "errors": 0}

    # Apply limit if specified
    if limit:
        documents = list(collection.find(query).limit(limit))
        logger.info(f"🎯 Processing {len(documents)} documents (limited)")
    else:
        documents = list(collection.find(query))
        logger.info(f"🎯 Processing all {len(documents)} documents")

    logger.info("")

    # Migration stats
    migrated = 0
    errors = 0
    skipped = 0
    error_details = []

    # Migrate each document
    for i, doc in enumerate(documents):
        doc_id = doc['_id']
        filename = doc.get('file_name', 'Unknown')
        user_id = doc.get('user_id')
        data = doc.get('data', [])

        try:
            logger.info(f"\n[{i+1}/{len(documents)}] Document: {filename}")
            logger.info(f"  ID: {doc_id}")
            logger.info(f"  User: {user_id}")
            logger.info(f"  Records: {len(data)}")

            if not data:
                logger.warning(f"  ⚠️  Skipping - no vendor records")
                skipped += 1
                continue

            # Check if already migrated
            if data and 'vendor_name_std' in data[0]:
                logger.warning(f"  ⚠️  Skipping - already migrated")
                skipped += 1
                continue

            # Batch standardize all records
            logger.info(f"  🔄 Standardizing {len(data)} vendor records...")
            standardized_data = standardizer.batch_standardize(data, user_id=user_id)

            # Count successful standardizations
            std_count = sum(1 for rec in standardized_data if rec.get('vendor_name_std'))
            avg_confidence = sum(rec.get('standardization_confidence', 0) for rec in standardized_data) / len(standardized_data) if standardized_data else 0

            logger.info(f"  ✓ Standardized: {std_count}/{len(data)} records")
            logger.info(f"  ✓ Avg confidence: {avg_confidence:.2f}")

            # Update database if not dry run
            if not dry_run:
                result = collection.update_one(
                    {"_id": doc_id},
                    {
                        "$set": {
                            "data": standardized_data,
                            "migration_date": datetime.utcnow(),
                            "migration_version": "1.0"
                        }
                    }
                )

                if result.modified_count > 0:
                    logger.info(f"  ✓ Updated in MongoDB")
                    migrated += 1
                else:
                    logger.warning(f"  ⚠️  MongoDB update returned 0 modified")
            else:
                logger.info(f"  ✓ Would update in MongoDB (dry run)")
                migrated += 1

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            errors += 1
            error_details.append({
                "document_id": str(doc_id),
                "filename": filename,
                "error": str(e)
            })

    # Print summary
    logger.info("")
    logger.info("="*70)
    logger.info("  MIGRATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total documents found:     {total_docs}")
    logger.info(f"Documents processed:       {len(documents)}")
    logger.info(f"Successfully migrated:     {migrated}")
    logger.info(f"Skipped (already done):    {skipped}")
    logger.info(f"Errors:                    {errors}")
    logger.info("")

    if error_details:
        logger.error("ERROR DETAILS:")
        for err in error_details:
            logger.error(f"  - {err['filename']} ({err['document_id']}): {err['error']}")
        logger.info("")

    if dry_run:
        logger.info("🔸 DRY RUN - No changes were made to the database")
    else:
        logger.info("✅ MIGRATION COMPLETE - Database updated")

    logger.info("="*70)

    return {
        "success": True,
        "total_found": total_docs,
        "processed": len(documents),
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate strategy documents to add standardized keywords'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test migration without updating database'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to migrate'
    )

    args = parser.parse_args()

    # Run migration
    result = migrate_strategy_documents(
        dry_run=args.dry_run,
        limit=args.limit
    )

    # Exit with appropriate code
    if result.get('success') and result.get('errors', 0) == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
