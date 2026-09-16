from typing import TypedDict

from .regex import (
    normalize_guardrail_text,
    INPUT_GUARDRAIL_REGEX,
)


# =========================================================
# CONFIGURATION
# =========================================================

MAX_INPUT_LENGTH = 5000


# =========================================================
# RESULT
# =========================================================

class InputGuardrailResult(TypedDict):
    allowed: bool
    message: str
    reason: str


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_input(
    text: str,
) -> InputGuardrailResult:

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    message = normalize_guardrail_text(text)

    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    if not message:

        return {
            "allowed": False,
            "message": (
                "Please provide a hiking-related question."
            ),
            "reason": "empty_input",
        }

    # -----------------------------------------------------
    # Length
    # -----------------------------------------------------

    if len(message) > MAX_INPUT_LENGTH:

        return {
            "allowed": False,
            "message": (
                "Your message is too long. "
                "Please shorten it."
            ),
            "reason": "input_too_long",
        }

    # -----------------------------------------------------
    # Regex security check
    # -----------------------------------------------------

    match = INPUT_GUARDRAIL_REGEX.search(message)

    if match:

        return {
            "allowed": False,
            "message": (
                "I can only help with hiking, "
                "trekking, trails, equipment, "
                "planning, and safety."
            ),
            "reason": "input_security_pattern",
        }

    # -----------------------------------------------------
    # Allowed
    # -----------------------------------------------------

    return {
        "allowed": True,
        "message": "",
        "reason": "",
    }
