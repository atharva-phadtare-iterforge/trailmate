import re
import unicodedata


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_guardrail_text(text: str) -> str:
    """
    Normalize text before applying guardrail checks.

    - Unicode NFKC normalization
    - Removes zero-width/invisible characters
    - Normalizes whitespace
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove zero-width / invisible characters
    text = re.sub(
        r"[\u200B-\u200D\u2060\uFEFF]",
        "",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# INPUT GUARDRAIL REGEX
# =========================================================

INPUT_GUARDRAIL_REGEX = re.compile(
    r"(?i)\b(?:"
    r"ignore previous instructions|"
    r"ignore all previous instructions|"
    r"ignore prior instructions|"
    r"ignore all prior instructions|"
    r"disregard previous instructions|"
    r"disregard all previous instructions|"
    r"disregard prior instructions|"
    r"forget previous instructions|"
    r"forget all previous instructions|"
    r"override previous instructions|"
    r"override system instructions|"
    r"replace previous instructions|"
    r"bypass safety|"
    r"bypass security|"
    r"bypass guardrails|"
    r"disable safety|"
    r"disable security|"
    r"disable guardrails|"
    r"remove restrictions|"
    r"remove guardrails|"
    r"jailbreak|"
    r"jail break|"
    r"do anything now|"
    r"DAN|"
    r"developer mode|"
    r"debug mode|"
    r"admin mode|"
    r"unrestricted mode|"
    r"uncensored mode|"
    r"unfiltered mode|"
    r"god mode|"
    r"reveal your system prompt|"
    r"reveal your prompt|"
    r"show me your system prompt|"
    r"show me your prompt|"
    r"show your instructions|"
    r"reveal your instructions|"
    r"show system instructions|"
    r"reveal system instructions|"
    r"show developer instructions|"
    r"reveal developer instructions|"
    r"api key|"
    r"api-key|"
    r"api_key|"
    r"access token|"
    r"auth token|"
    r"authentication token|"
    r"private key|"
    r"secret key|"
    r"password|"
    r"credentials|"
    r"environment variables|"
    r"env variables|"
    r"dump secrets|"
    r"extract secrets|"
    r"retrieve secrets|"
    r"export secrets|"
    r"leak secrets|"
    r"show configuration|"
    r"reveal configuration|"
    r"show internal data|"
    r"reveal internal data|"
    r"show private data|"
    r"reveal private data|"
    r"execute shell|"
    r"run shell|"
    r"execute command|"
    r"run command|"
    r"execute bash|"
    r"run bash|"
    r"execute powershell|"
    r"run powershell|"
    r"execute code|"
    r"run code|"
    r"execute script|"
    r"run script|"
    r"database dump|"
    r"dump database|"
    r"extract database|"
    r"filesystem|"
    r"file system"
    r")\b|"
    r"sk-[A-Za-z0-9_-]{8,}|"
    r"Bearer [A-Za-z0-9._~+/=-]{20,}"
)


# =========================================================
# OUTPUT GUARDRAIL REGEX
# =========================================================

OUTPUT_GUARDRAIL_REGEX = re.compile(
    r"(?i)\b(?:"
    r"system prompt|"
    r"developer prompt|"
    r"hidden prompt|"
    r"internal prompt|"
    r"system instructions|"
    r"developer instructions|"
    r"hidden instructions|"
    r"internal instructions|"
    r"system message|"
    r"developer message|"
    r"internal message|"
    r"internal configuration|"
    r"private configuration|"
    r"hidden configuration|"
    r"system configuration|"
    r"developer configuration|"
    r"internal policy|"
    r"private policy|"
    r"api key|"
    r"api-key|"
    r"api_key|"
    r"access token|"
    r"auth token|"
    r"authentication token|"
    r"private key|"
    r"secret key|"
    r"password|"
    r"credentials"
    r")\b|"
    r"sk-[A-Za-z0-9_-]{8,}|"
    r"Bearer [A-Za-z0-9._~+/=-]{20,}|"
    r"\b(?:"
    r"ignore previous instructions|"
    r"ignore all previous instructions|"
    r"disregard previous instructions|"
    r"disregard all previous instructions|"
    r"override previous instructions|"
    r"bypass safety|"
    r"bypass security|"
    r"disable safety|"
    r"disable security"
    r")\b"
)
