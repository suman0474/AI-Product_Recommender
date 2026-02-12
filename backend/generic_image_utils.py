"""
COMPATIBILITY SHIM FOR GENERIC IMAGE UTILS

This module provides backward compatibility for code still importing from
'generic_image_utils' at the root level.

The actual implementation has been moved to:
    services/azure/image_utils.py

ALL NEW CODE SHOULD USE:
    from services.azure.image_utils import fetch_generic_product_image

This file will be removed in a future version once all imports are updated.
"""

import warnings

# Show deprecation warning when this module is imported
warnings.warn(
    "The 'generic_image_utils' module at root is deprecated. "
    "Please use 'from services.azure.image_utils import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functions from the new location
from services.azure.image_utils import (
    fetch_generic_product_image,
    fetch_generic_product_image_fast,
    fetch_generic_images_batch,
    regenerate_generic_image,
)

__all__ = [
    'fetch_generic_product_image',
    'fetch_generic_product_image_fast',
    'fetch_generic_images_batch',
    'regenerate_generic_image',
]
