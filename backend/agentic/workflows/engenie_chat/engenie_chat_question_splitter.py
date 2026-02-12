# agentic/workflows/engenie_chat/engenie_chat_question_splitter.py
"""
Multi-Question Splitter for EnGenie Chat

Parses multi-question inputs and extracts individual questions with context
(refinery names, product types, vendor names) for routing to appropriate RAGs.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParsedQuestion:
    """Represents a single extracted question with context."""
    original_text: str
    cleaned_text: str
    question_number: int
    extracted_context: Dict[str, Any] = field(default_factory=dict)
    # Context includes: refinery, product_type, vendor, standards_mentioned

    def __post_init__(self):
        if not self.extracted_context:
            self.extracted_context = {}


@dataclass
class MultiQuestionInput:
    """Result of parsing multi-question input."""
    questions: List[ParsedQuestion]
    is_multi_question: bool
    original_input: str
    question_count: int


# =============================================================================
# KNOWN ENTITIES (From Strategy CSV & Standards)
# =============================================================================

# Refineries from instrumentation_procurement_strategy.csv
KNOWN_REFINERIES = [
    "Baton Rouge Refinery",
    "Baytown Refinery",
    "Martinez Refinery",
    "Port Arthur Refinery",
    "Whiting Refinery",
    # Add more as needed from your data
]

# Product type keywords for extraction
PRODUCT_TYPE_PATTERNS = {
    # Control Valves
    r"control\s+valves?": "Control Valves",
    r"ball\s+valves?": "Ball Valves",
    r"globe\s+valves?": "Globe Valves",
    r"butterfly\s+valves?": "Butterfly Valves",
    r"needle\s+valves?": "Needle Valves",

    # Flow Instruments
    r"flow\s+meters?": "Flow Meters",
    r"flowmeters?": "Flow Meters",
    r"mass\s+flow": "Mass Flow Meters",
    r"ultrasonic\s+flow": "Ultrasonic Flow Meters",
    r"magnetic\s+flow": "Magnetic Flow Meters",
    r"coriolis": "Coriolis Flow Meters",

    # Pressure Instruments
    r"pressure\s+transmitters?": "Pressure Transmitters",
    r"pressure\s+gauges?": "Pressure Gauges",
    r"differential\s+pressure": "Differential Pressure Transmitters",

    # Temperature Instruments
    r"temperature\s+sensors?": "Temperature Sensors",
    r"temperature\s+transmitters?": "Temperature Transmitters",
    r"thermocouples?": "Thermocouples",
    r"rtds?": "RTDs",

    # Level Instruments
    r"level\s+instruments?": "Level Instruments",
    r"level\s+sensors?": "Level Sensors",
    r"level\s+transmitters?": "Level Transmitters",
    r"radar\s+level": "Radar Level Sensors",

    # Analytical Instruments
    r"analytical\s+instruments?": "Analytical Instruments",
    r"analyzers?": "Analyzers",
    r"ph\s+meters?": "pH Meters",
    r"conductivity": "Conductivity Meters",

    # Safety Instruments
    r"safety\s+instruments?": "Safety Instruments",
    r"gas\s+detectors?": "Gas Detectors",
    r"safety\s+valves?": "Safety Valves",

    # Vibration
    r"vibration\s+sensors?": "Vibration Sensors",
    r"accelerometers?": "Accelerometers",
}

# Standards keywords for detecting Standards RAG questions
STANDARDS_KEYWORDS = [
    r"sil[-\s]?[1-4]", r"iec\s*\d+", r"iso\s*\d+", r"api\s*\d+",
    r"atex", r"iecex", r"certification", r"compliance", r"standard",
    r"hazardous\s+area", r"zone\s+[0-2]", r"functional\s+safety"
]


# =============================================================================
# QUESTION SPLITTING
# =============================================================================

def split_questions(input_text: str) -> MultiQuestionInput:
    """
    Split a multi-question input into individual questions.

    Handles:
    - Numbered questions: "1) ...", "1. ...", "(1) ...", "1- ..."
    - Newline-separated questions ending with "?"
    - Single questions (returns list of 1)

    Args:
        input_text: Raw user input potentially containing multiple questions

    Returns:
        MultiQuestionInput with parsed questions and metadata
    """
    if not input_text or not input_text.strip():
        return MultiQuestionInput(
            questions=[],
            is_multi_question=False,
            original_input=input_text or "",
            question_count=0
        )

    input_text = input_text.strip()
    questions = []

    # Pattern 1: Numbered questions (1), 1., 1-, 1:
    numbered_pattern = r'(?:^|\n)\s*(?:\d+[\)\.\-:\]]\s*|\(\d+\)\s*)'

    # Check if input has numbered format
    if re.search(numbered_pattern, input_text):
        # Split by numbered patterns
        parts = re.split(numbered_pattern, input_text)
        parts = [p.strip() for p in parts if p.strip()]

        for i, part in enumerate(parts, 1):
            if part:
                questions.append(_create_parsed_question(part, i))

    # Pattern 2: Question mark separated (if no numbers found)
    elif input_text.count('?') > 1:
        # Split by question marks, keeping the question mark
        parts = re.split(r'(\?)', input_text)
        current_question = ""
        question_num = 1

        for part in parts:
            current_question += part
            if part == '?' and current_question.strip():
                questions.append(_create_parsed_question(current_question.strip(), question_num))
                question_num += 1
                current_question = ""

        # Handle any remaining text
        if current_question.strip():
            questions.append(_create_parsed_question(current_question.strip(), question_num))

    # Pattern 3: Newline separated
    elif '\n' in input_text:
        lines = [l.strip() for l in input_text.split('\n') if l.strip()]
        for i, line in enumerate(lines, 1):
            questions.append(_create_parsed_question(line, i))

    # Single question
    else:
        questions.append(_create_parsed_question(input_text, 1))

    is_multi = len(questions) > 1

    logger.info(f"[QUESTION_SPLITTER] Split into {len(questions)} question(s), multi={is_multi}")

    return MultiQuestionInput(
        questions=questions,
        is_multi_question=is_multi,
        original_input=input_text,
        question_count=len(questions)
    )


def _create_parsed_question(text: str, question_number: int) -> ParsedQuestion:
    """Create a ParsedQuestion with extracted context."""
    cleaned = _clean_question_text(text)
    context = extract_question_context(cleaned)

    return ParsedQuestion(
        original_text=text,
        cleaned_text=cleaned,
        question_number=question_number,
        extracted_context=context
    )


def _clean_question_text(text: str) -> str:
    """Clean and normalize question text."""
    # Remove leading numbers/bullets
    text = re.sub(r'^[\d\)\.\-:\]\s\(]+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


# =============================================================================
# CONTEXT EXTRACTION
# =============================================================================

def extract_question_context(question: str) -> Dict[str, Any]:
    """
    Extract context from a single question.

    Extracts:
    - refinery: Refinery name if mentioned
    - product_type: Product type/category
    - vendor: Any vendor names mentioned
    - standards_mentioned: Any standards (IEC, ISO, SIL, etc.)
    - is_strategy_question: Hint for Strategy RAG routing
    - is_standards_question: Hint for Standards RAG routing

    Args:
        question: Question text to analyze

    Returns:
        Dict with extracted context
    """
    context = {
        "refinery": None,
        "product_type": None,
        "vendor": None,
        "standards_mentioned": [],
        "is_strategy_question": False,
        "is_standards_question": False,
    }

    question_lower = question.lower()

    # Extract refinery
    context["refinery"] = extract_refinery(question)

    # Extract product type
    context["product_type"] = extract_product_type(question)

    # Extract standards mentions
    context["standards_mentioned"] = extract_standards(question)

    # Determine question type hints
    strategy_indicators = [
        "preferred vendor", "approved vendor", "procurement", "strategy",
        "vendor for", "supplier for", "who supplies", "which vendor",
        "buy for", "purchase for", "should i buy", "should we use"
    ]

    for indicator in strategy_indicators:
        if indicator in question_lower:
            context["is_strategy_question"] = True
            break

    # If refinery mentioned, likely strategy question
    if context["refinery"]:
        context["is_strategy_question"] = True

    # Standards question detection
    if context["standards_mentioned"]:
        context["is_standards_question"] = True

    standards_question_patterns = [
        "requirement", "certification", "compliance", "standard",
        "what is sil", "explain sil", "atex", "iecex"
    ]
    for pattern in standards_question_patterns:
        if pattern in question_lower:
            context["is_standards_question"] = True
            break

    return context


def extract_refinery(text: str) -> Optional[str]:
    """
    Extract refinery name from question text.

    Args:
        text: Question text

    Returns:
        Refinery name if found, None otherwise
    """
    text_lower = text.lower()

    # Check against known refineries (case-insensitive)
    for refinery in KNOWN_REFINERIES:
        if refinery.lower() in text_lower:
            return refinery

    # Try pattern matching for unknown refineries
    # Pattern: "at/for/regarding [Name] Refinery"
    patterns = [
        r"(?:at|for|regarding|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Refinery",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Refinery(?:\s+regarding)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            refinery_name = match.group(1).strip()
            return f"{refinery_name} Refinery"

    return None


def extract_product_type(text: str) -> Optional[str]:
    """
    Extract product type from question text.

    Args:
        text: Question text

    Returns:
        Product type if found, None otherwise
    """
    text_lower = text.lower()

    for pattern, product_type in PRODUCT_TYPE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return product_type

    return None


def extract_standards(text: str) -> List[str]:
    """
    Extract standards mentions from question text.

    Args:
        text: Question text

    Returns:
        List of standards mentioned
    """
    standards = []
    text_lower = text.lower()

    # SIL levels
    sil_match = re.findall(r'sil[-\s]?([1-4])', text_lower)
    for level in sil_match:
        standards.append(f"SIL-{level}")

    # IEC standards
    iec_match = re.findall(r'iec\s*(\d+)', text_lower)
    for num in iec_match:
        standards.append(f"IEC {num}")

    # ISO standards
    iso_match = re.findall(r'iso\s*(\d+)', text_lower)
    for num in iso_match:
        standards.append(f"ISO {num}")

    # API standards
    api_match = re.findall(r'api\s*(\d+)', text_lower)
    for num in api_match:
        standards.append(f"API {num}")

    # ATEX/IECEx
    if 'atex' in text_lower:
        standards.append("ATEX")
    if 'iecex' in text_lower:
        standards.append("IECEx")

    return standards


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_multi_question_input(text: str) -> bool:
    """
    Quick check if input likely contains multiple questions.

    Args:
        text: Input text to check

    Returns:
        True if likely multi-question, False otherwise
    """
    if not text:
        return False

    # Check for numbered pattern
    if re.search(r'(?:^|\n)\s*\d+[\)\.\-:]', text):
        return True

    # Check for multiple question marks
    if text.count('?') > 1:
        return True

    # Check for multiple newline-separated sentences
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 1:
        return True

    return False


def get_questions_summary(parsed: MultiQuestionInput) -> str:
    """
    Get a summary of parsed questions for logging.

    Args:
        parsed: MultiQuestionInput result

    Returns:
        Summary string
    """
    lines = [f"Questions: {parsed.question_count}, Multi: {parsed.is_multi_question}"]

    for q in parsed.questions:
        ctx = q.extracted_context
        refinery = ctx.get("refinery", "N/A")
        product = ctx.get("product_type", "N/A")
        is_strategy = ctx.get("is_strategy_question", False)
        is_standards = ctx.get("is_standards_question", False)

        rag_hint = "Strategy" if is_strategy else ("Standards" if is_standards else "Index")
        lines.append(f"  Q{q.question_number}: {q.cleaned_text[:50]}... | {rag_hint} | {refinery} | {product}")

    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ParsedQuestion',
    'MultiQuestionInput',
    'split_questions',
    'extract_question_context',
    'extract_refinery',
    'extract_product_type',
    'extract_standards',
    'is_multi_question_input',
    'get_questions_summary',
    'KNOWN_REFINERIES',
    'PRODUCT_TYPE_PATTERNS',
]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_input = """
    1) What is the preferred vendor for Control Valves at Baton Rouge Refinery?
    2) Which Flow Meters should I buy for Port Arthur Refinery?
    3) What is the procurement strategy for Analytical Instruments at Whiting Refinery?
    4) What are SIL-2 certification requirements for pressure transmitters?
    5) What is the spec for Rosemount 3051S?
    """

    print("=" * 80)
    print("TESTING QUESTION SPLITTER")
    print("=" * 80)

    result = split_questions(test_input)
    print(get_questions_summary(result))

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for q in result.questions:
        print(f"\nQuestion {q.question_number}:")
        print(f"  Text: {q.cleaned_text}")
        print(f"  Context: {q.extracted_context}")
