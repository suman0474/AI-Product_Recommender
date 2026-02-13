"""
Validators Module

Centralized validation logic for query validation, input sanitization, etc.

Includes:
- query_validator: Main query domain validation (industrial vs out-of-domain)
- validation_patterns: Shared patterns and keywords
"""

from .query_validator import (
    ValidationResult,
    validate_query_domain,
    create_rejection_response,
    validate_query,
)

# Import validation_patterns module (available as validators.validation_patterns)
from . import validation_patterns

__all__ = [
    'ValidationResult',
    'validate_query_domain',
    'create_rejection_response',
    'validate_query',
    'validation_patterns',
]
