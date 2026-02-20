"""
Solution Deep Agent Module.
Contains the core logic for industrial solution analysis, instrument identification,
and automated specification enrichment.
"""

from .solution_deep_agent import (
    SolutionDeepAgentState,
    create_solution_deep_agent_state,
)
from .workflow import (
    run_solution_deep_agent,
    run_solution_deep_agent_stream,
    create_solution_deep_agent_workflow,
)
from .intent_analyzer import (
    SolutionIntentClassifier,
    IntentResult,
)
from .context_manager import (
    SolutionContextManager,
    ConversationMessage,
    ExtractedEntities,
)
from .flash_personality import (
    FlashPersonality,
    FlashPlanner,
    FlashResponseComposer,
    ExecutionStrategy,
    ResponseTone,
    ExecutionPlan,
)
from .identification_agents import (
    InstrumentIdentificationAgent,
    AccessoryIdentificationAgent,
    identify_instruments_and_accessories_parallel,
)
from .orchestration import (
    OrchestrationContext,
    get_current_context,
    get_orchestration_logger,
    SolutionOrchestrator,
    get_solution_orchestrator,
)

__all__ = [
    # Core Agent
    "SolutionDeepAgentState",
    "create_solution_deep_agent_state",
    
    # Workflow
    "run_solution_deep_agent",
    "run_solution_deep_agent_stream",
    "create_solution_deep_agent_workflow",
    
    # Intent
    "SolutionIntentClassifier",
    "IntentResult",
    
    # Context
    "SolutionContextManager",
    "ConversationMessage",
    "ExtractedEntities",
    
    # Flash Personality
    "FlashPersonality",
    "FlashPlanner",
    "FlashResponseComposer",
    "ExecutionStrategy",
    "ResponseTone",
    "ExecutionPlan",
    
    # Identification
    "InstrumentIdentificationAgent",
    "AccessoryIdentificationAgent",
    "identify_instruments_and_accessories_parallel",


    # Orchestration
    "OrchestrationContext",
    "get_current_context",
    "get_orchestration_logger",
    "SolutionOrchestrator",
    "get_solution_orchestrator",
]
