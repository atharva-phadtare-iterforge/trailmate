from typing import TypedDict

from .regex import (
    normalize_guardrail_text,
    OUTPUT_GUARDRAIL_REGEX,
)


# =========================================================
# CONFIGURATION
# =========================================================

MAX_OUTPUT_LENGTH = 20000


# =========================================================
# RESULT
# =========================================================

class OutputGuardrailResult(TypedDict):
    allowed: bool
    message: str
    reason: str


# =========================================================
# OUTPUT VALIDATION
# =========================================================

def validate_output(
    text: str,
) -> OutputGuardrailResult:

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    response = normalize_guardrail_text(text)

    # -----------------------------------------------------
    # Empty output
    # -----------------------------------------------------

    if not response:

        return {
            "allowed": False,
            "message": (
                "The assistant could not generate "
                "a valid response."
            ),
            "reason": "empty_output",
        }

    # -----------------------------------------------------
    # Length
    # -----------------------------------------------------

    if len(response) > MAX_OUTPUT_LENGTH:

        return {
            "allowed": False,
            "message": (
                "The generated response was too long."
            ),
            "reason": "output_too_long",
        }

    # -----------------------------------------------------
    # Regex security check
    # -----------------------------------------------------

    match = OUTPUT_GUARDRAIL_REGEX.search(response)

    if match:

        return {
            "allowed": False,
            "message": (
                "The generated response failed "
                "the output safety check."
            ),
            "reason": "output_security_pattern",
        }

    # -----------------------------------------------------
    # Allowed
    # -----------------------------------------------------

    return {
        "allowed": True,
        "message": "",
        "reason": "",
    }
