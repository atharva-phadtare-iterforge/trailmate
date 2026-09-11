from typing import (
    TypedDict,
    Literal,
    Annotated,
)

from uuid import uuid4
import tiktoken
import asyncio
import json
import time
from ..init.service_client import search_trails

from fastapi import (
    APIRouter,
    Request,
    Response,
    HTTPException,
)

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from langgraph.runtime import Runtime
from litellm import acompletion

from ..guardrails import (
    validate_input,
    validate_output,
)


# =========================================================
# FASTAPI ROUTER
# =========================================================

router = APIRouter(
    prefix="/hiking",
    tags=["hiking"],
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL = "openai/hiking-small"
API_KEY = "sk-1234"
API_BASE = "http://localhost:4000/v1"

CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.50

GUARDRAIL_CHECK_SIZE = 200


# =========================================================
# TYPES
# =========================================================

Category = Literal[
    "trail",
    "general",
    "reject",
]

TrailType = Literal[
    "trail_info",
    "difficulty",
    "equipment",
    "safety",
    "planning",
    "general_hiking",
]


class ClassificationResult(TypedDict):
    category: Category
    trail_type: TrailType
    confidence: float


class State(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    category: Category
    trail_type: TrailType
    classification_confidence: float
    response: str
    input_allowed: bool
    input_guardrail_message: str
    output_allowed: bool
    output_guardrail_message: str
    retrieved_trails: list[dict]

# =========================================================
# REQUEST MODEL
# =========================================================

class UserMessage(BaseModel):
    message: str

# =========================================================
# STREAM CONTEXT
# =========================================================

class StreamContext:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

# =========================================================
# SSE HELPER
# =========================================================

async def send_sse_event(
    stream_context: StreamContext,
    event_type: str,
    **data,
):
    await stream_context.queue.put(
        {
            "type": event_type,
            **data,
        }
    )

# =========================================================
# JAVA TRAIL RETRIEVAL
# =========================================================

async def retrieve_trails_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    query = get_latest_message(state)

    print(
        "[HIKING RETRIEVAL]"
        f" Searching Java service for: {query}"
    )

    try:

        results = await asyncio.to_thread(
            search_trails,
            query,
            3,
        )

        # -------------------------------------------------
        # Keep only information that the LLM needs
        # -------------------------------------------------

        retrieved_trails = []

        for result in results[:3]:

            retrieved_trails.append(
                {
                    "trailId": result.get("trailId"),
                    "trailName": result.get("trailName"),
                    "sectionName": result.get("sectionName"),
                    "chunkText": result.get("chunkText"),
                }
            )

        print(
            "[HIKING RETRIEVAL]"
            f" Retrieved {len(retrieved_trails)} trails"
        )

        for index, trail in enumerate(
            retrieved_trails,
            start=1,
        ):

            print(
                "[HIKING RETRIEVAL]"
                f" {index}. "
                f"{trail['trailName']} "
                f"- {trail['sectionName']}"
            )

        return {
            "retrieved_trails": retrieved_trails,
        }

    except Exception as exc:

        print(
            "[HIKING RETRIEVAL]"
            f" Error: {exc}"
        )

        return {
            "retrieved_trails": [],
        }

    
# =========================================================
# MESSAGE HELPERS
# =========================================================

def get_latest_message(state: State) -> str:

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):

            content = message.content

            if isinstance(content, str):
                return content

            return str(content)

    return ""


def message_to_dict(message: BaseMessage) -> dict:

    if isinstance(message, HumanMessage):

        return {
            "role": "user",
            "content": message.content,
        }

    if isinstance(message, AIMessage):

        return {
            "role": "assistant",
            "content": message.content,
        }

    return {
        "role": "user",
        "content": str(message.content),
    }


def conversation_prompt(
    state: State,
    system_prompt: str,
) -> list[dict]:

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # -----------------------------------------------------
    # Conversation history
    # -----------------------------------------------------

    for message in state["messages"]:

        messages.append(
            message_to_dict(message)
        )

    # -----------------------------------------------------
    # Retrieved trail information
    # -----------------------------------------------------

    retrieved_trails = state.get(
        "retrieved_trails",
        [],
    )

    if retrieved_trails:

        retrieval_context = (
            "\n\n"
            "RETRIEVED TRAIL INFORMATION:\n"
            "The following information was retrieved "
            "from the TrailMate search system.\n"
            "Use it as the source of truth for "
            "trail-specific facts.\n"
            "Do not invent facts that are not present "
            "in the retrieved information.\n"
        )

        for index, trail in enumerate(
            retrieved_trails,
            start=1,
        ):

            retrieval_context += (
                f"\nTrail {index}:\n"
                f"Trail ID: "
                f"{trail.get('trailId')}\n"
                f"Trail Name: "
                f"{trail.get('trailName')}\n"
                f"Section: "
                f"{trail.get('sectionName')}\n"
                f"Information: "
                f"{trail.get('chunkText')}\n"
            )

        messages[0]["content"] += retrieval_context

    return messages


# =========================================================
# INPUT GUARDRAIL
# =========================================================

def input_guardrail_node(state: State):

    message = get_latest_message(state)

    result = validate_input(message)

    print(
        "[HIKING INPUT GUARDRAIL]"
        f" allowed={result['allowed']}"
    )

    return {
        "input_allowed": result["allowed"],
        "input_guardrail_message": result["message"],
    }


# =========================================================
# INPUT GUARDRAIL ROUTER
# =========================================================

def route_input_guardrail(state: State):

    if state["input_allowed"]:
        return "continue"

    return "block"


# =========================================================
# INPUT GUARDRAIL BLOCK
# =========================================================

async def input_guardrail_blocked(
    state: State,
    runtime: Runtime[StreamContext],
):

    response = state["input_guardrail_message"]

    await send_sse_event(
        runtime.context,
        "guardrail_block",
        guardrail="input",
        content=response,
    )

    await send_sse_event(
        runtime.context,
        "stream_done",
    )

    return {
        "response": response,
        "messages": [
            AIMessage(
                content=response
            )
        ],
    }


# =========================================================
# LLM CLASSIFIER
# =========================================================

async def classify_node(
    state: State,
    runtime: Runtime[StreamContext],
) -> dict:

    message = get_latest_message(state)

    classifier_prompt = """
You are the STRICT classification engine for TrailMate.

Your ONLY job is to classify the user's message.

DO NOT answer the user's question.

DO NOT provide advice.

DO NOT explain your decision.

Return ONLY valid JSON.

=========================================================
ALLOWED CATEGORY VALUES
=========================================================

category MUST be exactly one of:

- trail
- general
- reject

=========================================================
ALLOWED TRAIL TYPES
=========================================================

If category = "trail", trail_type MUST be exactly one of:

- trail_info
- difficulty
- equipment
- safety
- planning
- general_hiking

If category = "general" or "reject",
trail_type MUST be:

general_hiking

=========================================================
CLASSIFICATION RULES
=========================================================

TRAIL:

Use category "trail" when the user is asking about:

- hiking
- trekking
- trails
- mountains
- outdoor hiking
- hiking preparation
- hiking equipment
- hiking safety
- hiking difficulty
- hiking planning
- trail information

GENERAL:

Use category "general" ONLY for simple conversational messages
such as:

- hello
- hi
- hey
- thanks
- thank you
- good morning
- good afternoon
- good evening

REJECT:

Use category "reject" for questions unrelated to:

- hiking
- trekking
- trails
- mountains
- hiking preparation
- outdoor hiking

=========================================================
TRAIL TYPE RULES
=========================================================

trail_info:

- trail location
- route
- distance
- duration
- elevation
- altitude
- starting point
- destination
- trail information
- opening hours
- entry fee
- best season
- trail details

difficulty:

- difficulty
- difficult
- easy
- hard
- beginner suitability
- fitness requirements
- steepness
- challenging terrain
- physical difficulty

equipment:

- equipment
- gear
- shoes
- backpack
- clothing
- water bottle
- trekking poles
- flashlight
- first aid
- what to carry
- what to bring

safety:

- hiking safety
- dangerous conditions
- weather safety
- rain
- monsoon
- storms
- wild animals
- snakes
- injuries
- emergencies
- rescue

planning:

- planning a hike
- planning a trek
- itinerary
- hiking schedule
- trip planning
- preparing for a hike
- preparing for a trek
- organizing a hiking trip

general_hiking:

Hiking-related questions that do not clearly fit
trail_info, difficulty, equipment, safety, or planning.

=========================================================
IMPORTANT
=========================================================

Classify based on USER INTENT.

Examples:

"What equipment should I carry for a difficult hike?"

=> equipment

"Is this difficult for a beginner?"

=> difficulty

"Is this trail safe during monsoon?"

=> safety

"How far is the trail and where does it start?"

=> trail_info

"Create a two-day plan for this trek."

=> planning

=========================================================
CONFIDENCE
=========================================================

Return confidence between 0 and 1.

0.90 - 1.00 = very clear
0.75 - 0.89 = reasonably clear
0.50 - 0.74 = ambiguous
0.00 - 0.49 = highly uncertain

=========================================================
OUTPUT
=========================================================

Return EXACTLY:

{
  "category": "trail",
  "trail_type": "planning",
  "confidence": 0.95
}

No markdown.
No explanation.
No additional fields.
"""

    messages = [
        {
            "role": "system",
            "content": classifier_prompt,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    try:

        response = await acompletion(
            model=MODEL,
            messages=messages,
            api_key=API_KEY,
            api_base=API_BASE,
            temperature=0,
        )

        content = response.choices[0].message.content

        print(
            f"[HIKING CLASSIFIER] Raw response: {content}"
        )

        if not content:
            raise ValueError(
                "Classifier returned empty response"
            )

        # -------------------------------------------------
        # Remove accidental markdown fences
        # -------------------------------------------------

        content = content.strip()

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        result = json.loads(content)

        category = result.get("category")
        trail_type = result.get("trail_type")
        confidence = result.get("confidence")

        # -------------------------------------------------
        # STRICT CATEGORY VALIDATION
        # -------------------------------------------------

        valid_categories = {
            "trail",
            "general",
            "reject",
        }

        if category not in valid_categories:

            raise ValueError(
                f"Invalid category: {category}"
            )

        # -------------------------------------------------
        # STRICT TRAIL TYPE VALIDATION
        # -------------------------------------------------

        valid_trail_types = {
            "trail_info",
            "difficulty",
            "equipment",
            "safety",
            "planning",
            "general_hiking",
        }

        if trail_type not in valid_trail_types:

            raise ValueError(
                f"Invalid trail_type: {trail_type}"
            )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = float(confidence)

        if not 0 <= confidence <= 1:

            raise ValueError(
                f"Invalid confidence: {confidence}"
            )

        # -------------------------------------------------
        # Non-trail categories
        # -------------------------------------------------

        if category != "trail":

            trail_type = "general_hiking"

        # -------------------------------------------------
        # LOW CONFIDENCE
        # -------------------------------------------------

        if confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:

            print(
                "[HIKING CLASSIFIER]"
                f" Low confidence={confidence:.2f}"
                " -> reject"
            )

            category = "reject"
            trail_type = "general_hiking"

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        print(
            "[HIKING CLASSIFIER]"
            f" category={category}"
            f" trail_type={trail_type}"
            f" confidence={confidence:.2f}"
        )

        # -------------------------------------------------
        # Send classification to frontend
        # -------------------------------------------------

        await send_sse_event(
            runtime.context,
            "classification",
            category=category,
            trail_type=trail_type,
            confidence=confidence,
        )

        return {
            "category": category,
            "trail_type": trail_type,
            "classification_confidence": confidence,
        }

    except Exception as exc:

        print(
            f"[HIKING CLASSIFIER] Error: {exc}"
        )

        # -------------------------------------------------
        # Strict failure
        # -------------------------------------------------

        await send_sse_event(
            runtime.context,
            "classification",
            category="reject",
            trail_type="general_hiking",
            confidence=0.0,
        )

        return {
            "category": "reject",
            "trail_type": "general_hiking",
            "classification_confidence": 0.0,
        }


# =========================================================
# ROUTING
# =========================================================

def route_message(state: State):

    category = state["category"]
    trail_type = state["trail_type"]

    print(
        "[HIKING ROUTER]"
        f" category={category}"
        f" trail_type={trail_type}"
    )

    # -----------------------------------------------------
    # TRAIL
    # -----------------------------------------------------

    if category == "trail":

        # IMPORTANT:
        # Route using trail_type.
        return trail_type

    # -----------------------------------------------------
    # GENERAL
    # -----------------------------------------------------

    if category == "general":

        return "general"

    # -----------------------------------------------------
    # REJECT
    # -----------------------------------------------------

    return "reject"


# =========================================================
# STREAM LLM RESPONSE
# =========================================================

async def stream_llm_response(
    state: State,
    system_prompt: str,
    stream_context: StreamContext,
):

    messages = conversation_prompt(
        state,
        system_prompt,
    )

    print(
        "[HIKING LLM]"
        f" Starting {MODEL}"
    )

    full_response = ""

    guardrail_buffer = ""

    try:

        response = await acompletion(
            model=MODEL,
            messages=messages,
            api_key=API_KEY,
            api_base=API_BASE,
            temperature=0,
            stream=True,
        )

        # =================================================
        # STREAM
        # =================================================

        async for chunk in response:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            text = delta.content or ""

            if not text:
                continue

            full_response += text

            guardrail_buffer += text

            # ---------------------------------------------
            # Incremental output guardrail
            # ---------------------------------------------

            if len(guardrail_buffer) >= GUARDRAIL_CHECK_SIZE:

                result = validate_output(
                    guardrail_buffer
                )

                if not result["allowed"]:

                    guardrail_message = result["message"]

                    print(
                        "[HIKING OUTPUT GUARDRAIL]"
                        " BLOCKED"
                    )

                    await send_sse_event(
                        stream_context,
                        "guardrail_block",
                        guardrail="output",
                        content=guardrail_message,
                    )

                    return {
                        "response": guardrail_message,
                        "output_allowed": False,
                        "output_guardrail_message":
                            guardrail_message,
                        "messages": [
                            AIMessage(
                                content=guardrail_message
                            )
                        ],
                    }

                guardrail_buffer = ""

            # ---------------------------------------------
            # Send chunk
            # ---------------------------------------------

            await send_sse_event(
                stream_context,
                "chunk",
                source="llm",
                content=text,
            )

            await asyncio.sleep(0)

        # =================================================
        # FINAL OUTPUT GUARDRAIL
        # =================================================

        final_result = validate_output(
            full_response
        )

        if not final_result["allowed"]:

            guardrail_message = final_result["message"]

            print(
                "[HIKING OUTPUT GUARDRAIL]"
                " FINAL BLOCK"
            )

            await send_sse_event(
                stream_context,
                "guardrail_block",
                guardrail="output",
                content=guardrail_message,
            )

            return {
                "response": guardrail_message,
                "output_allowed": False,
                "output_guardrail_message":
                    guardrail_message,
                "messages": [
                    AIMessage(
                        content=guardrail_message
                    )
                ],
            }

        # =================================================
        # COMPLETE
        # =================================================

        print(
            "[HIKING LLM]"
            f" Completed response length={len(full_response)}"
        )

        return {
            "response": full_response,
            "output_allowed": True,
            "output_guardrail_message": "",
            "messages": [
                AIMessage(
                    content=full_response
                )
            ],
        }

    except Exception as exc:

        print(
            f"[HIKING LLM] Error: {exc}"
        )

        await send_sse_event(
            stream_context,
            "error",
            message=str(exc),
        )

        raise


# =========================================================
# TRAIL RESPONSE
# =========================================================

async def trail_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] trail")

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking trail assistant.

Use the entire conversation history.

Answer the user's hiking-related question.

Focus on:

- trail information
- route
- distance
- estimated duration
- location
- elevation
- best season
- general trail details

If the user says:

- it
- that trail
- there
- this place
- this trek

use previous conversation context.

Be concise and helpful.

IMPORTANT:

- Do not answer unrelated questions.
- If you don't know a specific fact, say you are not certain.
- Do not invent exact trail information.
- Do not pretend to have live weather or trail conditions.
- Do not make up opening hours, prices, distances, or elevations.
""",
        runtime.context,
    )


# =========================================================
# DIFFICULTY RESPONSE
# =========================================================

async def difficulty_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] difficulty")

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking difficulty assistant.

Use the entire conversation history.

Answer the user's question about hiking difficulty.

Explain when relevant:

- difficulty level
- fitness required
- terrain
- elevation
- steep sections
- approximate challenge
- whether beginners can attempt it

Be concise and practical.

IMPORTANT:

- Do not answer unrelated questions.
- Do not invent exact measurements or facts.
- If the trail is unknown, clearly say difficulty
  depends on the specific trail.
- Do not claim exact trail conditions without reliable data.
""",
        runtime.context,
    )


# =========================================================
# EQUIPMENT RESPONSE
# =========================================================

async def equipment_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] equipment")

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking equipment assistant.

Use the entire conversation history.

Answer the user's question about hiking equipment.

Consider:

- trail
- duration
- difficulty
- weather
- terrain

Recommend practical items such as:

- hiking shoes
- water
- backpack
- first-aid kit
- navigation
- rain protection
- appropriate clothing
- flashlight/headlamp
- trekking poles when useful

Be concise and practical.

IMPORTANT:

- Do not answer unrelated questions.
- Do not recommend unnecessary equipment.
- Do not invent specific trail conditions.
""",
        runtime.context,
    )


# =========================================================
# SAFETY RESPONSE
# =========================================================

async def safety_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] safety")

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking safety assistant.

Use the entire conversation history.

Answer the user's hiking safety question.

Focus on:

- weather awareness
- staying on marked trails
- sufficient water
- informing someone about the route
- navigation
- emergency preparation
- appropriate clothing and footwear
- avoiding unnecessary risks

If conditions are dangerous, prioritize safety.

Be concise and practical.

IMPORTANT:

- Do not answer unrelated questions.
- Do not encourage dangerous behavior.
- Do not invent specific weather or trail conditions.
- If there is immediate danger, recommend contacting
  local emergency services or appropriate authorities.
""",
        runtime.context,
    )



# =========================================================
# PLANNING RESPONSE
# =========================================================

async def planning_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] planning")

    # -----------------------------------------------------
    # Planning system prompt
    # -----------------------------------------------------

    system_prompt = """
You are TrailMate, a hiking planning assistant.

Use the entire conversation history.

The user wants a hiking or trekking plan.

Create the plan DIRECTLY.

DO NOT ask follow-up questions.

Use information already provided by the user
and previous conversation.

If information is missing:

- make reasonable general assumptions
- mention important assumptions
- continue creating the plan

Do not invent exact trail facts.

Include:

1. Overview
2. Preparation
3. Suggested schedule
4. Route approach
5. What to carry
6. Food and water
7. Clothing
8. Safety
9. Backup plan

Keep the plan concise and practical.
"""

    # -----------------------------------------------------
    # Build the EXACT messages that will be sent to LLM
    # -----------------------------------------------------

    messages = conversation_prompt(
        state,
        system_prompt,
    )

    # -----------------------------------------------------
    # TIKTOKEN PRE-FLIGHT CHECK
    # -----------------------------------------------------

    try:

        encoder = tiktoken.get_encoding(
            "cl100k_base"
        )

        total_text = ""

        for message in messages:

            total_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        token_count = len(
            encoder.encode(total_text)
        )

        character_count = len(total_text)

        print(
            "[HIKING TOKENS]"
            f" planning prompt"
            f" characters={character_count}"
            f" estimated_tokens={token_count}"
        )

    except Exception as exc:

        print(
            "[HIKING TOKENS]"
            f" Token counting failed: {exc}"
        )

    # -----------------------------------------------------
    # Call the existing streaming function
    # -----------------------------------------------------

    return await stream_llm_response(
        state,
        system_prompt,
        runtime.context,
    )


# =========================================================
# GENERAL HIKING RESPONSE
# =========================================================

async def general_hiking_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] general_hiking")

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking assistant.

Use the entire conversation history.

Answer the user's hiking-related question.

The question does not clearly fall into:

- trail information
- difficulty
- equipment
- safety
- planning

Give a useful general hiking answer.

Be concise and friendly.

IMPORTANT:

Only discuss:

- hiking
- trekking
- trails
- mountains
- outdoor preparation
- closely related topics
""",
        runtime.context,
    )


# =========================================================
# GENERAL RESPONSE
# =========================================================

async def general_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] general")

    response = (
        "Hello! I'm TrailMate. "
        "I can help you with hiking trails, "
        "trekking, hiking plans, equipment, "
        "difficulty, and safety."
    )

    await send_sse_event(
        runtime.context,
        "chunk",
        source="system",
        content=response,
    )

    return {
        "response": response,
        "output_allowed": True,
        "output_guardrail_message": "",
        "messages": [
            AIMessage(
                content=response
            )
        ],
    }


# =========================================================
# REJECT RESPONSE
# =========================================================

async def reject_response(
    state: State,
    runtime: Runtime[StreamContext],
):

    print("[HIKING NODE] reject")

    response = (
        "I'm sorry, but I can only help with "
        "hiking, trekking, trails, hiking plans, "
        "equipment, difficulty, and safety."
    )

    await send_sse_event(
        runtime.context,
        "chunk",
        source="guardrail",
        content=response,
    )

    return {
        "response": response,
        "output_allowed": True,
        "output_guardrail_message": "",
        "messages": [
            AIMessage(
                content=response
            )
        ],
    }


# =========================================================
# OUTPUT GUARDRAIL
# =========================================================

def output_guardrail_node(
    state: State,
):

    response = state.get(
        "response",
        "",
    )

    result = validate_output(response)

    print(
        "[HIKING OUTPUT GUARDRAIL]"
        f" allowed={result['allowed']}"
    )

    return {
        "output_allowed": result["allowed"],
        "output_guardrail_message": result["message"],
    }


# =========================================================
# OUTPUT GUARDRAIL ROUTER
# =========================================================

def route_output_guardrail(
    state: State,
):

    if state["output_allowed"]:
        return "allow"

    return "block"


# =========================================================
# OUTPUT GUARDRAIL BLOCK
# =========================================================

async def output_guardrail_blocked(
    state: State,
    runtime: Runtime[StreamContext],
):

    response = state["output_guardrail_message"]

    await send_sse_event(
        runtime.context,
        "guardrail_block",
        guardrail="output",
        content=response,
    )

    return {
        "response": response,
        "messages": [
            AIMessage(
                content=response
            )
        ],
    }


# =========================================================
# BUILD GRAPH
# =========================================================

workflow = StateGraph(State)


# =========================================================
# INPUT
# =========================================================

workflow.add_node(
    "input_guardrail",
    input_guardrail_node,
)

workflow.add_node(
    "input_guardrail_blocked",
    input_guardrail_blocked,
)


# =========================================================
# CLASSIFIER
# =========================================================

workflow.add_node(
    "classify",
    classify_node,
)

workflow.add_node(
    "retrieve_trails",
    retrieve_trails_node,
)

# =========================================================
# RESPONSE NODES
# =========================================================

workflow.add_node(
    "trail_info",
    trail_response,
)

workflow.add_node(
    "difficulty",
    difficulty_response,
)

workflow.add_node(
    "equipment",
    equipment_response,
)

workflow.add_node(
    "safety",
    safety_response,
)

workflow.add_node(
    "planning",
    planning_response,
)

workflow.add_node(
    "general_hiking",
    general_hiking_response,
)

workflow.add_node(
    "general",
    general_response,
)

workflow.add_node(
    "reject",
    reject_response,
)


# =========================================================
# OUTPUT GUARDRAIL
# =========================================================

workflow.add_node(
    "output_guardrail",
    output_guardrail_node,
)

workflow.add_node(
    "output_guardrail_blocked",
    output_guardrail_blocked,
)


# =========================================================
# START
# =========================================================

workflow.add_edge(
    START,
    "input_guardrail",
)


# =========================================================
# INPUT ROUTING
# =========================================================

workflow.add_conditional_edges(
    "input_guardrail",
    route_input_guardrail,
    {
        "continue": "classify",
        "block": "input_guardrail_blocked",
    },
)


# =========================================================
# INPUT BLOCK -> END
# =========================================================

workflow.add_edge(
    "input_guardrail_blocked",
    END,
)


# =========================================================
# CLASSIFIER ROUTING
# =========================================================

workflow.add_conditional_edges(
    "classify",
    route_message,
    {
        "trail_info": "retrieve_trails",
        "difficulty": "retrieve_trails",
        "equipment": "retrieve_trails",
        "safety": "retrieve_trails",
        "planning": "retrieve_trails",
        "general_hiking": "retrieve_trails",

        "general": "general",
        "reject": "reject",
    },
)

workflow.add_conditional_edges(
    "retrieve_trails",
    lambda state: state["trail_type"],
    {
        "trail_info": "trail_info",
        "difficulty": "difficulty",
        "equipment": "equipment",
        "safety": "safety",
        "planning": "planning",
        "general_hiking": "general_hiking",
    },
)


# =========================================================
# RESPONSE -> OUTPUT GUARDRAIL
# =========================================================

workflow.add_edge(
    "trail_info",
    "output_guardrail",
)

workflow.add_edge(
    "difficulty",
    "output_guardrail",
)

workflow.add_edge(
    "equipment",
    "output_guardrail",
)

workflow.add_edge(
    "safety",
    "output_guardrail",
)

workflow.add_edge(
    "planning",
    "output_guardrail",
)

workflow.add_edge(
    "general_hiking",
    "output_guardrail",
)

workflow.add_edge(
    "general",
    "output_guardrail",
)

workflow.add_edge(
    "reject",
    "output_guardrail",
)


# =========================================================
# OUTPUT GUARDRAIL ROUTING
# =========================================================

workflow.add_conditional_edges(
    "output_guardrail",
    route_output_guardrail,
    {
        "allow": END,
        "block": "output_guardrail_blocked",
    },
)


# =========================================================
# OUTPUT BLOCK -> END
# =========================================================

workflow.add_edge(
    "output_guardrail_blocked",
    END,
)


# =========================================================
# CHECKPOINTER
# =========================================================

checkpointer = MemorySaver()


# =========================================================
# COMPILE
# =========================================================

hiking_graph = workflow.compile(
    checkpointer=checkpointer,
)


# =========================================================
# NEW CHAT
# =========================================================

@router.post("/new-chat")
def new_chat(response: Response):

    thread_id = str(uuid4())

    response.set_cookie(
        key="hiking_thread_id",
        value=thread_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24,
    )

    return {
        "message": "New hiking chat created",
    }


# =========================================================
# ASK
# =========================================================

@router.post("/ask")
async def send_message(
    request: UserMessage,
    http_request: Request,
):

    # =====================================================
    # THREAD
    # =====================================================

    thread_id = http_request.cookies.get(
        "hiking_thread_id"
    )

    if not thread_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "No active hiking chat found. "
                "Please create a new chat first."
            ),
        )

    # =====================================================
    # STREAM CONTEXT
    # =====================================================

    stream_context = StreamContext()

    # =====================================================
    # CONFIG
    # =====================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # =====================================================
    # INPUT STATE
    # =====================================================

    input_state = {

        "messages": [
            HumanMessage(
                content=request.message
            )
        ],

        "category": "reject",

        "trail_type": "general_hiking",

        "classification_confidence": 0.0,

        "retrieved_trails": [],

        "response": "",

        "input_allowed": True,

        "input_guardrail_message": "",

        "output_allowed": True,

        "output_guardrail_message": "",
    }

    # =====================================================
    # SSE GENERATOR
    # =====================================================

    async def event_generator():

        request_start = time.perf_counter()

        graph_task = asyncio.create_task(
            hiking_graph.ainvoke(
                input_state,
                config=config,
                context=stream_context,
            )
        )

        try:

            while True:

                # -----------------------------------------
                # If graph crashes before producing an event
                # -----------------------------------------

                if graph_task.done() and stream_context.queue.empty():

                    exception = graph_task.exception()

                    if exception:

                        raise exception

                    break

                try:

                    event = await asyncio.wait_for(
                        stream_context.queue.get(),
                        timeout=0.5,
                    )

                except asyncio.TimeoutError:

                    if graph_task.done():

                        exception = graph_task.exception()

                        if exception:
                            raise exception

                        break

                    continue

                event_type = event["type"]

                # =========================================
                # CLASSIFICATION
                # =========================================

                if event_type == "classification":

                    yield (
                        "event: classification\n"
                        f"data: {json.dumps(event)}\n\n"
                    )

                # =========================================
                # LLM CHUNK
                # =========================================

                elif event_type == "chunk":

                    yield (
                        "event: chunk\n"
                        f"data: {json.dumps(event)}\n\n"
                    )

                # =========================================
                # GUARDRAIL
                # =========================================

                elif event_type == "guardrail_block":

                    yield (
                        "event: guardrail_block\n"
                        f"data: {json.dumps(event)}\n\n"
                    )

                # =========================================
                # ERROR
                # =========================================

                elif event_type == "error":

                    yield (
                        "event: error\n"
                        f"data: {json.dumps(event)}\n\n"
                    )

                    break

                # =========================================
                # STREAM DONE
                # =========================================

                elif event_type == "stream_done":

                    yield (
                        "event: stream_done\n"
                        f"data: {json.dumps(event)}\n\n"
                    )
                    continue

            # =================================================
            # WAIT FOR COMPLETE GRAPH
            # =================================================

            result = await graph_task

            total_time = (
                time.perf_counter()
                - request_start
            )

            print(
                "[HIKING]"
                f" Completed in {total_time:.3f}s"
            )

            print(
                "[HIKING]"
                f" Final category={result.get('category')}"
                f" trail_type={result.get('trail_type')}"
                f" confidence="
                f"{result.get('classification_confidence', 0):.2f}"
            )

            # =================================================
            # FINAL DONE
            # =================================================

            yield (
                "event: done\n"
                f"data: {json.dumps({
                    'type': 'done',
                    'category': result.get('category'),
                    'trail_type': result.get('trail_type'),
                    'confidence': result.get(
                        'classification_confidence',
                        0.0,
                    ),
                })}\n\n"
            )

        except asyncio.CancelledError:

            if not graph_task.done():

                graph_task.cancel()

                try:
                    await graph_task
                except asyncio.CancelledError:
                    pass

            raise

        except Exception as exc:

            print(
                f"[HIKING] Streaming error: {exc}"
            )

            if not graph_task.done():

                graph_task.cancel()

                try:
                    await graph_task
                except asyncio.CancelledError:
                    pass

            yield (
                "event: error\n"
                f"data: {json.dumps({
                    'type': 'error',
                    'message': str(exc),
                })}\n\n"
            )

    # =====================================================
    # SSE RESPONSE
    # =====================================================

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
