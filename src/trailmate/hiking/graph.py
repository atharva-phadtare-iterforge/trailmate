from typing import (
    TypedDict,
    Literal,
    Annotated,
    AsyncGenerator,
)

from uuid import uuid4

import json
import time

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
)

from langchain_openai import ChatOpenAI


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="hiking-small",
    temperature=0,
    api_key="sk-1234",
    base_url="http://localhost:4000/v1",
    streaming=True,
)


# =========================================================
# FASTAPI ROUTER
# =========================================================

router = APIRouter(
    prefix="/hiking",
    tags=["hiking"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class UserMessage(BaseModel):
    message: str


# =========================================================
# STATE
# =========================================================

class State(TypedDict):

    messages: Annotated[list, add_messages]

    category: Literal[
        "trail",
        "general",
        "reject",
    ]

    trail_type: Literal[
        "trail_info",
        "difficulty",
        "equipment",
        "safety",
        "planning",
        "general_hiking",
    ]

    response: str

    # Input guardrail
    input_allowed: bool
    input_guardrail_message: str

    # Output guardrail
    output_allowed: bool
    output_guardrail_message: str


# =========================================================
# HELPER
# =========================================================

def get_latest_message(state: State) -> str:

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):
            return message.content

    return ""


def conversation_prompt(
    state: State,
    system_prompt: str,
):

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        *state["messages"],
    ]


# =========================================================
# INPUT GUARDRAIL
# =========================================================

def input_guardrail_node(state: State):

    message = get_latest_message(state).strip()

    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    if not message:

        return {
            "input_allowed": False,
            "input_guardrail_message": (
                "Please provide a hiking-related question."
            ),
        }

    # -----------------------------------------------------
    # Maximum input length
    # -----------------------------------------------------

    if len(message) > 5000:

        return {
            "input_allowed": False,
            "input_guardrail_message": (
                "Your message is too long. "
                "Please shorten it."
            ),
        }

    # -----------------------------------------------------
    # Basic prompt-injection / abuse checks
    # -----------------------------------------------------

    blocked_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "reveal your prompt",
        "show me your instructions",
    ]

    lower_message = message.lower()

    for pattern in blocked_patterns:

        if pattern in lower_message:

            return {
                "input_allowed": False,
                "input_guardrail_message": (
                    "I can only help with hiking, "
                    "trekking, trails, equipment, "
                    "planning, and safety."
                ),
            }

    # -----------------------------------------------------
    # Allowed
    # -----------------------------------------------------

    return {
        "input_allowed": True,
        "input_guardrail_message": "",
    }


# =========================================================
# INPUT GUARDRAIL ROUTER
# =========================================================

def route_input_guardrail(state: State):

    if state["input_allowed"]:
        return "continue"

    return "block"


# =========================================================
# INPUT GUARDRAIL BLOCK RESPONSE
# =========================================================

def input_guardrail_blocked(state: State):

    response = state["input_guardrail_message"]

    return {
        "response": response,

        "messages": [
            AIMessage(content=response)
        ],
    }


# =========================================================
# MAIN CLASSIFIER
# =========================================================

def classify_node(state: State):

    message = get_latest_message(state).lower()

    trail_keywords = [
        "trail",
        "hike",
        "hiking",
        "trek",
        "trekking",
        "mountain",
        "route",
        "waterfall",
        "summit",
        "difficulty",
        "climb",
        "climbing",
        "peak",
        "forest",
        "hill",
        "camping",
        "camp",
        "trekker",
        "altitude",
        "elevation",
        "adventure",
        "outdoor",
    ]

    general_keywords = [
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    if any(
        keyword in message
        for keyword in trail_keywords
    ):

        return {
            "category": "trail",
        }

    elif any(
        keyword in message
        for keyword in general_keywords
    ):

        return {
            "category": "general",
        }

    else:

        return {
            "category": "reject",
        }


# =========================================================
# HIKING QUESTION CLASSIFIER
# =========================================================

def analyze_trail_node(state: State):

    message = get_latest_message(state).lower()

    # =====================================================
    # PLANNING
    # =====================================================

    planning_keywords = [
        "plan",
        "planning",
        "itinerary",
        "schedule",
        "day plan",
        "trip plan",
        "hiking plan",
        "hike plan",
        "trek plan",
        "trekking plan",
        "hiking trip",
        "trekking trip",
        "plan a hike",
        "plan a trek",
        "plan my hike",
        "plan my trek",
        "prepare for a hike",
        "prepare for a trek",
        "prepare for hiking",
        "prepare for trekking",
        "organize a hike",
        "organise a hike",
        "organize a trek",
        "organise a trek",
    ]

    if any(
        keyword in message
        for keyword in planning_keywords
    ):

        return {
            "trail_type": "planning",
        }

    # =====================================================
    # DIFFICULTY
    # =====================================================

    difficulty_keywords = [
        "difficulty",
        "difficult",
        "easy",
        "hard",
        "beginner",
        "advanced",
        "moderate",
        "fitness",
        "fit",
        "fitness level",
        "level",
        "challenging",
        "challenge",
        "tough",
        "steep",
        "steepness",
    ]

    if any(
        keyword in message
        for keyword in difficulty_keywords
    ):

        return {
            "trail_type": "difficulty",
        }

    # =====================================================
    # EQUIPMENT
    # =====================================================

    equipment_keywords = [
        "equipment",
        "gear",
        "carry",
        "bring",
        "shoes",
        "shoe",
        "hiking shoes",
        "trekking shoes",
        "backpack",
        "bag",
        "clothes",
        "clothing",
        "jacket",
        "raincoat",
        "rain jacket",
        "torch",
        "flashlight",
        "headlamp",
        "water bottle",
        "bottle",
        "trekking pole",
        "trekking poles",
        "stick",
        "first aid",
        "first-aid",
    ]

    if any(
        keyword in message
        for keyword in equipment_keywords
    ):

        return {
            "trail_type": "equipment",
        }

    # =====================================================
    # SAFETY
    # =====================================================

    safety_keywords = [
        "safe",
        "safety",
        "danger",
        "dangerous",
        "risk",
        "monsoon",
        "rain",
        "rainy",
        "weather",
        "storm",
        "accident",
        "emergency",
        "wild animal",
        "wild animals",
        "snake",
        "snakes",
        "injury",
        "injured",
        "medical",
        "rescue",
        "rescue team",
    ]

    if any(
        keyword in message
        for keyword in safety_keywords
    ):

        return {
            "trail_type": "safety",
        }

    # =====================================================
    # TRAIL INFORMATION
    # =====================================================

    trail_info_keywords = [
        "route",
        "distance",
        "location",
        "where",
        "how long",
        "time",
        "trail",
        "trek",
        "hike",
        "mountain",
        "waterfall",
        "summit",
        "peak",
        "elevation",
        "altitude",
        "height",
        "starting point",
        "start point",
        "base",
        "base point",
        "entry",
        "entry fee",
        "best time",
        "best season",
        "season",
        "opening",
        "closed",
        "hours",
    ]

    if any(
        keyword in message
        for keyword in trail_info_keywords
    ):

        return {
            "trail_type": "trail_info",
        }

    return {
        "trail_type": "general_hiking",
    }


# =========================================================
# ROUTING
# =========================================================

def route_message(state: State):

    return state["category"]


def route_trail(state: State):

    return state["trail_type"]


# =========================================================
# COMMON STREAMING LLM FUNCTION
# =========================================================

async def stream_llm_response(
    state: State,
    system_prompt: str,
):

    messages = conversation_prompt(
        state,
        system_prompt,
    )

    full_response = ""

    async for chunk in llm.astream(messages):

        if not chunk.content:
            continue

        if isinstance(chunk.content, str):

            text = chunk.content

        else:

            text = str(chunk.content)

        full_response += text

    return {
        "response": full_response,

        "messages": [
            AIMessage(content=full_response)
        ],
    }


# =========================================================
# TRAIL RESPONSE
# =========================================================

async def trail_response(state: State):

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking trail assistant.

Use the entire conversation history to understand
the user's current question.

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

If the user refers to:
"it", "that trail", "there", "this place", etc.,
use previous conversation context.

Be concise and helpful.

IMPORTANT:
- Do not answer unrelated questions.
- If you don't know a specific fact, say you are not certain.
- Do not invent exact trail information.
- Do not pretend to have live weather or trail conditions.
- Do not make up opening hours, prices, distances, or elevations.
""",
    )


# =========================================================
# DIFFICULTY RESPONSE
# =========================================================

async def difficulty_response(state: State):

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
    )


# =========================================================
# EQUIPMENT RESPONSE
# =========================================================

async def equipment_response(state: State):

    return await stream_llm_response(
        state,
        """
You are TrailMate, a hiking equipment assistant.

Use the entire conversation history to understand
the trail, duration, difficulty, weather, and
other relevant context.

Answer the user's question about hiking equipment.

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

Consider trail, weather, duration, and difficulty.

Be concise and practical.

IMPORTANT:
- Do not answer unrelated questions.
- Do not recommend unnecessary equipment.
- Do not invent specific trail conditions.
""",
    )


# =========================================================
# SAFETY RESPONSE
# =========================================================

async def safety_response(state: State):

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
    )


# =========================================================
# PLANNING RESPONSE
# =========================================================

async def planning_response(state: State):

    return await stream_llm_response(
        state,
        """
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
""",
    )


# =========================================================
# GENERAL HIKING RESPONSE
# =========================================================

async def general_hiking_response(state: State):

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
Only discuss hiking, trekking, trails, mountains,
outdoor preparation, or closely related topics.
""",
    )


# =========================================================
# GENERAL TEMPLATE
# =========================================================

def general_response(state: State):

    response = (
        "Hello! I'm TrailMate. "
        "I can help you with hiking trails, "
        "trekking, hiking plans, equipment, "
        "difficulty, and safety."
    )

    return {
        "response": response,

        "messages": [
            AIMessage(content=response)
        ],
    }


# =========================================================
# REJECT TEMPLATE
# =========================================================

def reject_response(state: State):

    response = (
        "I'm sorry, but I can only help with "
        "hiking, trekking, trails, hiking plans, "
        "equipment, difficulty, and safety."
    )

    return {
        "response": response,

        "messages": [
            AIMessage(content=response)
        ],
    }


# =========================================================
# OUTPUT GUARDRAIL
# =========================================================

def output_guardrail_node(state: State):

    response = state.get(
        "response",
        "",
    )

    # -----------------------------------------------------
    # Empty response
    # -----------------------------------------------------

    if not response.strip():

        return {
            "output_allowed": False,

            "output_guardrail_message": (
                "The assistant could not generate "
                "a valid response."
            ),
        }

    # -----------------------------------------------------
    # Maximum output length
    # -----------------------------------------------------

    if len(response) > 20000:

        return {
            "output_allowed": False,

            "output_guardrail_message": (
                "The generated response was too long."
            ),
        }

    # -----------------------------------------------------
    # Basic forbidden output checks
    # -----------------------------------------------------

    blocked_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "api key",
        "sk-",
    ]

    lower_response = response.lower()

    for pattern in blocked_patterns:

        if pattern in lower_response:

            return {
                "output_allowed": False,

                "output_guardrail_message": (
                    "The generated response failed "
                    "the output safety check."
                ),
            }

    # -----------------------------------------------------
    # Allowed
    # -----------------------------------------------------

    return {
        "output_allowed": True,

        "output_guardrail_message": "",
    }


# =========================================================
# OUTPUT GUARDRAIL ROUTER
# =========================================================

def route_output_guardrail(state: State):

    if state["output_allowed"]:
        return "allow"

    return "block"


# =========================================================
# OUTPUT GUARDRAIL BLOCK
# =========================================================

def output_guardrail_blocked(state: State):

    response = state["output_guardrail_message"]

    return {
        "response": response,

        "messages": [
            AIMessage(content=response)
        ],
    }


# =========================================================
# BUILD GRAPH
# =========================================================

workflow = StateGraph(State)


# =========================================================
# ADD INPUT GUARDRAIL
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
# ADD CLASSIFICATION
# =========================================================

workflow.add_node(
    "classify",
    classify_node,
)

workflow.add_node(
    "analyze_trail",
    analyze_trail_node,
)


# =========================================================
# ADD RESPONSE NODES
# =========================================================

workflow.add_node(
    "trail",
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
# ADD OUTPUT GUARDRAIL
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
# INPUT GUARDRAIL ROUTING
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
# MAIN CLASSIFICATION
# =========================================================

workflow.add_conditional_edges(
    "classify",
    route_message,
    {
        "trail": "analyze_trail",
        "general": "general",
        "reject": "reject",
    },
)


# =========================================================
# HIKING ROUTING
# =========================================================

workflow.add_conditional_edges(
    "analyze_trail",
    route_trail,
    {
        "trail_info": "trail",
        "difficulty": "difficulty",
        "equipment": "equipment",
        "safety": "safety",
        "planning": "planning",
        "general_hiking": "general_hiking",
    },
)


# =========================================================
# ALL RESPONSE NODES -> OUTPUT GUARDRAIL
# =========================================================

workflow.add_edge(
    "trail",
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
# CREATE NEW CHAT
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
    # CONFIG
    # =====================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # =====================================================
    # INPUT
    # =====================================================

    input_state = {

        "messages": [
            HumanMessage(
                content=request.message
            )
        ],

        "category": "reject",

        "trail_type": "general_hiking",

        "response": "",

        "input_allowed": True,

        "input_guardrail_message": "",

        "output_allowed": True,

        "output_guardrail_message": "",
    }

    # =====================================================
    # SSE GENERATOR
    # =====================================================

    async def event_generator() -> AsyncGenerator[str, None]:

        request_start = time.perf_counter()

        first_token_time = None

        chunk_count = 0

        llm_streamed = False

        try:

            # =================================================
            # LANGGRAPH STREAM
            # =================================================

            async for event in hiking_graph.astream_events(
                input_state,
                config=config,
                version="v2",
            ):

                event_type = event["event"]

                event_name = event.get(
                    "name",
                    "",
                )

                # =============================================
                # LLM TOKEN
                # =============================================

                if event_type == "on_chat_model_stream":

                    chunk = event["data"]["chunk"]

                    content = chunk.content

                    if not content:
                        continue

                    if isinstance(
                        content,
                        str,
                    ):

                        text = content

                    else:

                        text = str(content)

                    # -----------------------------------------
                    # TTFT
                    # -----------------------------------------

                    if first_token_time is None:

                        first_token_time = (
                            time.perf_counter()
                        )

                        ttft = (
                            first_token_time
                            - request_start
                        )

                        print(
                            f"[HIKING] "
                            f"TTFT: {ttft:.3f}s"
                        )

                    # -----------------------------------------
                    # SEND IMMEDIATELY
                    # -----------------------------------------

                    llm_streamed = True

                    chunk_count += 1

                    payload = {
                        "type": "chunk",
                        "source": "llm",
                        "content": text,
                    }

                    yield (
                        "event: chunk\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                # =============================================
                # INPUT GUARDRAIL BLOCK
                # =============================================

                elif (
                    event_type == "on_chain_end"
                    and event_name
                    == "input_guardrail_blocked"
                ):

                    output = event["data"].get(
                        "output"
                    )

                    if not output:
                        continue

                    text = output.get(
                        "response",
                        "",
                    )

                    if not text:
                        continue

                    payload = {
                        "type": "guardrail_block",
                        "guardrail": "input",
                        "content": text,
                    }

                    yield (
                        "event: guardrail_block\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                # =============================================
                # TEMPLATE RESPONSE
                # =============================================

                elif (
                    event_type == "on_chain_end"
                    and event_name in {
                        "general",
                        "reject",
                    }
                ):

                    # These are non-LLM template responses.

                    if llm_streamed:
                        continue

                    output = event["data"].get(
                        "output"
                    )

                    if not output:
                        continue

                    text = output.get(
                        "response",
                        "",
                    )

                    if not text:
                        continue

                    payload = {
                        "type": "chunk",
                        "source": "template",
                        "content": text,
                    }

                    yield (
                        "event: chunk\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                # =============================================
                # OUTPUT GUARDRAIL
                # =============================================

                elif (
                    event_type == "on_chain_end"
                    and event_name
                    == "output_guardrail"
                ):

                    output = event["data"].get(
                        "output"
                    )

                    if not output:
                        continue

                    allowed = output.get(
                        "output_allowed",
                        True,
                    )

                    message = output.get(
                        "output_guardrail_message",
                        "",
                    )

                    # -----------------------------------------
                    # OUTPUT PASSED
                    # -----------------------------------------

                    if allowed:

                        payload = {
                            "type": "output_guardrail",
                            "status": "passed",
                        }

                        yield (
                            "event: output_guardrail\n"
                            f"data: {json.dumps(payload)}\n\n"
                        )

                    # -----------------------------------------
                    # OUTPUT FAILED
                    # -----------------------------------------

                    else:

                        payload = {
                            "type": "output_guardrail",
                            "status": "blocked",
                            "message": message,
                        }

                        yield (
                            "event: output_guardrail\n"
                            f"data: {json.dumps(payload)}\n\n"
                        )

                # =============================================
                # OUTPUT GUARDRAIL BLOCK RESPONSE
                # =============================================

                elif (
                    event_type == "on_chain_end"
                    and event_name
                    == "output_guardrail_blocked"
                ):

                    output = event["data"].get(
                        "output"
                    )

                    if not output:
                        continue

                    text = output.get(
                        "response",
                        "",
                    )

                    if not text:
                        continue

                    payload = {
                        "type": "guardrail_block",
                        "guardrail": "output",
                        "content": text,
                    }

                    yield (
                        "event: guardrail_block\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

            # =================================================
            # COMPLETE
            # =================================================

            total_time = (
                time.perf_counter()
                - request_start
            )

            print(
                f"[HIKING] "
                f"Completed in {total_time:.3f}s | "
                f"chunks={chunk_count}"
            )

            # =================================================
            # DONE
            # =================================================

            payload = {
                "type": "done",
            }

            yield (
                "event: done\n"
                f"data: {json.dumps(payload)}\n\n"
            )

        # =====================================================
        # ERROR
        # =====================================================

        except Exception as exc:

            print(
                f"[HIKING] Streaming error: {exc}"
            )

            payload = {
                "type": "error",
                "message": str(exc),
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(payload)}\n\n"
            )

    # =====================================================
    # SSE
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
