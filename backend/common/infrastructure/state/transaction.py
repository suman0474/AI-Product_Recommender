"""
State Transaction Management

Provides transaction semantics for state modifications with rollback support.
"""

import logging
import copy
from typing import Any, Dict, Callable
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)

class StateTransaction:
    """
    Provides transaction semantics for state modifications with rollback support.
    """

    def __init__(self, state: Dict[str, Any], auto_commit: bool = False):
        """
        Initialize transaction.

        Args:
            state: Original state dictionary
            auto_commit: If True, automatically commit on success
        """
        self.original_state = copy.deepcopy(state)
        self.working_state = state  # Reference to original
        self.auto_commit = auto_commit
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        """Commit changes (mark as successful)"""
        if self.rolled_back:
            raise RuntimeError("Cannot commit after rollback")

        self.committed = True
        logger.debug("Transaction committed")

    def rollback(self) -> None:
        """Rollback changes to original state"""
        if self.committed:
            logger.warning("Attempting rollback on committed transaction")
            return

        # Restore original state
        self.working_state.clear()
        self.working_state.update(self.original_state)
        self.rolled_back = True
        logger.info("Transaction rolled back")

    def __enter__(self):
        """Enter transaction context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context with automatic rollback on error"""
        if exc_type is not None:
            # Exception occurred - rollback
            logger.error(f"Transaction failed with {exc_type.__name__}: {exc_val}")
            self.rollback()
            return False  # Re-raise exception

        if self.auto_commit and not self.committed:
            self.commit()

        return True


@contextmanager
def state_transaction(state: Dict[str, Any], auto_commit: bool = True):
    """
    Context manager for transactional state modifications.

    Args:
        state: State dictionary to protect
        auto_commit: Auto-commit on success, rollback on error

    Yields:
        StateTransaction object

    Usage:
        with state_transaction(state) as txn:
            state["field"] = "new value"
            # Automatically commits on success
            # Automatically rolls back on exception
    """
    txn = StateTransaction(state, auto_commit=auto_commit)
    try:
        yield txn
    except Exception as e:
        txn.rollback()
        raise


def with_state_transaction(auto_commit: bool = True):
    """
    Decorator to add transaction semantics to workflow nodes.

    Args:
        auto_commit: Auto-commit on success

    Usage:
        @with_state_transaction()
        def my_node(state: dict) -> dict:
            state["field"] = "value"
            return state
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First arg should be state dict
            if not args or not isinstance(args[0], dict):
                logger.warning(
                    f"{func.__name__} called without state dict, "
                    "skipping transaction"
                )
                return func(*args, **kwargs)

            state = args[0]

            with state_transaction(state, auto_commit=auto_commit):
                return func(*args, **kwargs)

        return wrapper
    return decorator

__all__ = ['StateTransaction', 'state_transaction', 'with_state_transaction']
