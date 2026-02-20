
from enum import Enum, auto

class ResponseTags(Enum):
    """
    Tags for classifying response types.
    """
    GREETING = auto()
    CLARIFICATION = auto()
    DIRECT_ANSWER = auto()
    PRODUCT_RECOMMENDATION = auto()
    ERROR = auto()
    UNKNOWN = auto()

def classify_response(
    response_text: str = "",
    user_input: str = "",
    response_data: dict = None,
    workflow_type: str = "",
) -> ResponseTags:
    """
    Rough classification of response text.
    Accepts either a plain response_text string or the structured kwargs
    (user_input, response_data, workflow_type) used by API endpoints.
    """
    if not response_text and response_data:
        response_text = (
            response_data.get("response")
            or response_data.get("message")
            or response_data.get("answer")
            or ""
        )
    text = (response_text or "").lower()
    if "hello" in text or "hi " in text:
        return ResponseTags.GREETING
    if "sorry" in text or "error" in text:
        return ResponseTags.ERROR
    return ResponseTags.UNKNOWN
