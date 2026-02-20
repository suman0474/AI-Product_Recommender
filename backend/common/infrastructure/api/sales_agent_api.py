
# common/infrastructure/api/sales_agent_api.py
import logging
from typing import Any, Dict, List
from flask import Blueprint, request, jsonify
from common.agentic.deep_agent.workflows.deep_agentic_workflow import DeepAgenticWorkflowOrchestrator
from common.infrastructure.api.utils import api_response, handle_errors

logger = logging.getLogger(__name__)

sales_agent_bp = Blueprint('sales_agent', __name__)

# Initialize orchestrator lazily or globally
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DeepAgenticWorkflowOrchestrator()
    return _orchestrator

@sales_agent_bp.route('/', methods=['POST'])
@handle_errors
def process_sales_agent_request():
    """
    Process requests for the Sales Agent workflow using the Deep Agentic Orchestrator.
    
    Expected JSON payload:
    {
        "user_message": str,       # User's input text
        "session_id": str,         # Session identifier
        "step": str,               # Current frontend step (informational)
        "data_context": dict,      # Context data
        "intent": str,             # "workflow" or "knowledgeQuestion"
        "thread_id": str,          # Optional thread ID
        "main_thread_id": str,     # Optional main thread ID
        "zone": str                # Optional geographic zone
    }
    """
    data = request.get_json() or {}
    
    user_message = data.get('user_message') or data.get('userMessage', '')
    session_id = data.get('session_id') or data.get('sessionId')
    step = data.get('step')
    data_context = data.get('data_context') or data.get('dataContext', {})
    intent = data.get('intent')
    thread_id = data.get('thread_id')
    main_thread_id = data.get('main_thread_id')
    zone = data.get('zone')
    
    logger.info(f"[SALES_AGENT_API] Processing request for session={session_id}, step={step}")
    
    orchestrator = get_orchestrator()
    
    # Map frontend context to orchestrator parameters
    # If user provided fields in data_context, extract them
    user_provided_fields = {}
    if step == 'awaitMissingInfo' or step == 'awaitAdditionalAndLatestSpecs':
        # Check if user provided filled fields in context
        if data_context.get('filledFields'):
             user_provided_fields = data_context.get('filledFields')
        elif data_context.get('requirements'):
             user_provided_fields = data_context.get('requirements')

    # Determine user decision from message or context
    user_decision = None
    if user_message:
        # Simple heuristic: let orchestrator parse the message for decision
        user_decision = user_message
    
    # Process request
    result = orchestrator.process_request(
        user_input=user_message,
        session_id=session_id,
        thread_id=thread_id,
        main_thread_id=main_thread_id,
        zone=zone,
        user_decision=user_decision,
        user_provided_fields=user_provided_fields,
        product_type_hint=data_context.get('productType')
    )
    
    # Map result back to frontend expected format
    response_data = {
        "content": result.get('sales_agent_response', ''),
        "nextStep": _map_phase_to_step(result.get('current_phase')),
        "maintainWorkflow": not result.get('completed', False),
        "dataContext": {
            "productType": result.get('product_type'),
            "schema": result.get('schema'),
            "missingFields": result.get('missing_fields'),
            "providedRequirements": result.get('provided_requirements', {}),
            "discoveredSpecs": result.get('advanced_parameters_result', {}).get('discovered_specifications', []),
            "generatedResponse": result.get('sales_agent_response')
        },
        "discoveredParameters": result.get('advanced_parameters_result', {}).get('discovered_specifications', [])
    }
    
    return api_response(True, data=response_data)


def _map_phase_to_step(phase):
    """Map backend workflow phase to frontend step."""
    if not phase:
        return 'initialInput'
        
    mapping = {
        'initial': 'initialInput',
        'validation': 'initialInput',
        'await_missing_fields': 'awaitMissingInfo',
        'collect_fields': 'awaitMissingInfo',
        'await_advanced_params': 'awaitAdvancedSpecs',
        'advanced_discovery': 'awaitAdvancedSpecs', 
        'await_advanced_selection': 'awaitAdvancedSpecs',
        'vendor_analysis': 'finalAnalysis',
        'ranking': 'finalAnalysis',
        'complete': 'showSummary',
        'error': 'initialInput' 
    }
    return mapping.get(phase, 'initialInput')


# =============================================================================
# FLASK API FUNCTION: Generate Clarification Request
# =============================================================================

@sales_agent_bp.route('/clarify', methods=['POST'])
@handle_errors
def generate_clarification_request():
    """
    Generate a clarification request message for missing requirements.
    
    Expected JSON payload:
    {
        "product_type": str,
        "missing_fields": List[str],
        "schema": dict,
        "provided_requirements": dict
    }
    """
    data = request.get_json() or {}
    product_type = data.get('product_type', '')
    missing_fields = data.get('missing_fields', [])
    schema = data.get('schema', {})
    provided_requirements = data.get('provided_requirements', {})

    if not missing_fields:
        return api_response(True, data={"message": "", "missing_fields": []})

    result = _build_clarification_request(
        product_type=product_type,
        missing_fields=missing_fields,
        schema=schema,
        provided_requirements=provided_requirements
    )
    
    return api_response(True, data=result)


def _build_clarification_request(
    product_type: str,
    missing_fields: List[str],
    schema: Dict[str, Any],
    provided_requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured clarification payload for the user.
    (Ported from sales_agent_node.py)

    Returns a dict with:
        - message: Human-readable prompt listing the missing fields
        - missing_fields: List of field names
        - field_descriptions: Description for each missing field (from schema)
        - provided_so_far: Summary of what the user already provided
    """
    # Extract human-readable descriptions from schema
    field_descriptions: Dict[str, str] = {}
    mandatory_section = (
        schema.get("mandatory_requirements")
        or schema.get("mandatory_fields")
        or schema.get("mandatory")
        or {}
    )

    if isinstance(mandatory_section, dict):
        for field in missing_fields:
            field_descriptions[field] = mandatory_section.get(field, field)
    elif isinstance(mandatory_section, list):
        for field in missing_fields:
            field_descriptions[field] = field

    # Build human-readable message
    field_lines = []
    for i, field in enumerate(missing_fields[:10], 1):
        desc = field_descriptions.get(field, field)
        # Format field name nicely
        display_name = field.replace("_", " ").replace("camel", " ").title()
        if desc and desc != field:
            field_lines.append(f"  {i}. **{display_name}** — {desc}")
        else:
            field_lines.append(f"  {i}. **{display_name}**")

    fields_text = "\n".join(field_lines)
    provided_count = len(provided_requirements)

    message = (
        f"I'm working on finding the best **{product_type}** options for you. "
        f"I've captured {provided_count} specification(s) so far, but I need a few more details "
        f"to ensure accurate results:\n\n"
        f"{fields_text}\n\n"
        f"Please provide values for the fields above, or type **'skip'** to proceed "
        f"with the information I already have."
    )

    return {
        "message": message,
        "missing_fields": missing_fields,
        "field_descriptions": field_descriptions,
        "provided_so_far": {k: str(v) for k, v in provided_requirements.items()},
        "can_skip": True,
    }

