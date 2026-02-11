# agentic/auth_decorators.py
# =============================================================================
# CONSOLIDATED AUTHENTICATION DECORATORS
# =============================================================================
#
# This module consolidates all authentication-related decorators that were
# previously duplicated across multiple API modules.
#
# Previously duplicated in:
#   - agentic/api.py
#   - agentic/session_api.py
#   - agentic/product_info_api.py
#   - agentic/engenie_chat/engenie_chat_api.py
#   - agentic/strategy_rag/strategy_admin_api.py
#   - tools/api.py
#
# Usage:
#   from agentic.infrastructure.utils.auth_decorators import login_required
#
#   @app.route('/protected')
#   @login_required
#   def protected_endpoint():
#       ...
#
# =============================================================================

import logging
from functools import wraps
from flask import session, jsonify

logger = logging.getLogger(__name__)


def login_required(func):
    """
    Decorator to require user authentication for endpoints.

    Checks if 'user_id' exists in the Flask session.
    Returns 401 Unauthorized if not authenticated.

    Args:
        func: The view function to wrap

    Returns:
        Wrapped function that checks authentication before executing

    Example:
        @app.route('/api/protected')
        @login_required
        def protected_endpoint():
            return jsonify({'message': 'You are authenticated'})
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.warning(f"Unauthorized access attempt to {func.__name__}")
            return jsonify({
                "success": False,
                "error": "Unauthorized: Please log in"
            }), 401
        return func(*args, **kwargs)
    return decorated_function


def admin_required(func):
    """
    Decorator to require admin privileges for endpoints.

    Checks if user is authenticated AND has admin role.
    Returns 401 if not authenticated, 403 if not admin.

    Args:
        func: The view function to wrap

    Returns:
        Wrapped function that checks admin status before executing

    Example:
        @app.route('/api/admin/users')
        @admin_required
        def admin_users():
            return jsonify({'users': [...]})
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.warning(f"Unauthorized access attempt to admin endpoint {func.__name__}")
            return jsonify({
                "success": False,
                "error": "Unauthorized: Please log in"
            }), 401

        # Check for admin role (stored in session during login)
        if not session.get('is_admin', False):
            logger.warning(
                f"Non-admin user {session.get('user_id')} attempted to access {func.__name__}"
            )
            return jsonify({
                "success": False,
                "error": "Forbidden: Admin access required"
            }), 403

        return func(*args, **kwargs)
    return decorated_function


def optional_auth(func):
    """
    Decorator for endpoints that work with or without authentication.

    Does not block unauthenticated requests, but makes user_id
    available in the request context if authenticated.

    Args:
        func: The view function to wrap

    Returns:
        Wrapped function that adds auth context without requiring it

    Example:
        @app.route('/api/public')
        @optional_auth
        def public_endpoint():
            user_id = session.get('user_id')  # May be None
            return jsonify({'authenticated': user_id is not None})
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        # Just log if user is authenticated, but don't block
        if 'user_id' in session:
            logger.debug(f"Authenticated request to {func.__name__} by user {session['user_id']}")
        return func(*args, **kwargs)
    return decorated_function


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'login_required',
    'admin_required',
    'optional_auth'
]
