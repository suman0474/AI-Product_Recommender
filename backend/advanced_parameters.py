# advanced_parameters.py
#
# Backward-compatibility shim.
# The implementation has been merged into search/advanced_specification_agent.py
# as a LangGraph Deep Agent. This file simply re-exports the public API
# so that any existing callers using:
#
#     from advanced_parameters import discover_advanced_parameters
#
# continue to work without modification.

from search.advanced_specification_agent import (
    AdvancedSpecificationAgent,
    AdvancedSpecificationAgent,       # alias
    discover_advanced_parameters,
)

__all__ = [
    "AdvancedSpecificationAgent",
    "AdvancedSpecificationAgent",
    "discover_advanced_parameters",
]
