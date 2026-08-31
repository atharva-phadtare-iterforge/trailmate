from typing import TypedDict, Literal, Annotated, AsyncGenerator
from uuid import uuid4
import json

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="hiking-small",
    temperature=0,
    api_key="sk-1234",
    base_url="http://localhost:4000/v1",
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


# =========================================================
# HELPER
# =========================================================

def get_latest_message(state: State) -> str:
    """
    Return the latest human/user message.
    """

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):
            return message.content

    return ""


def conversation_prompt(
    state: State,
    system_prompt: str,
):
    """
    Create the LLM message list.

    The complete conversation history is included.
    """

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        *state["messages"],
    ]


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
# TRAIL RESPONSE
# =========================================================

def trail_response(state: State):

    result = llm.invoke(
        conversation_prompt(
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

If the user refers to something using:
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
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# DIFFICULTY RESPONSE
# =========================================================

def difficulty_response(state: State):

    result = llm.invoke(
        conversation_prompt(
            state,
            """
You are TrailMate, a hiking difficulty assistant.

Use the entire conversation history to understand
the user's current question and the trail they
are referring to.

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
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# EQUIPMENT RESPONSE
# =========================================================

def equipment_response(state: State):

    result = llm.invoke(
        conversation_prompt(
            state,
            """
You are TrailMate, a hiking equipment assistant.

Use the entire conversation history to understand
the trail, duration, difficulty, weather, and
other relevant context.

Answer the user's question about hiking equipment or gear.

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

Consider the trail, weather, duration, and difficulty
mentioned in the conversation.

Be concise and practical.

IMPORTANT:
- Do not answer unrelated questions.
- Do not recommend unnecessary equipment.
- Do not invent specific trail conditions.
- Give general recommendations when trail details are unknown.
""",
        )
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# SAFETY RESPONSE
# =========================================================

def safety_response(state: State):

    result = llm.invoke(
        conversation_prompt(
            state,
            """
You are TrailMate, a hiking safety assistant.

Use the entire conversation history to understand
the user's current situation and the trail they
are discussing.

Answer the user's hiking safety question.

Focus on:
- weather awareness
- staying on marked trails
- carrying sufficient water
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
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# 7. PLANNING RESPONSE
# =========================================================

def planning_response(state: State):

    result = llm.invoke(
        conversation_prompt(
            state,
            """
You are TrailMate, a hiking planning assistant.

Use the entire conversation history.

The user wants a hiking or trekking plan.

Create the plan DIRECTLY.

DO NOT ask follow-up questions.

DO NOT gather requirements.

Use information already provided by the user
and previous conversation.

If information is missing:
- make reasonable general assumptions
- mention important assumptions
- continue creating the plan

Do not invent exact trail facts.

If a specific trail is mentioned and you do not
have reliable information about it, say that
exact trail details should be verified separately.

Create a practical plan.

Include relevant sections such as:

1. Overview
2. Preparation
3. Suggested schedule
4. Route approach
5. What to carry
6. Food and water
7. Clothing
8. Safety
9. Backup plan

Keep the plan concise, practical, and easy to follow.
""",
        )
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# GENERAL HIKING RESPONSE
# =========================================================

def general_hiking_response(state: State):

    result = llm.invoke(
        conversation_prompt(
            state,
            """
You are TrailMate, a hiking assistant.

Use the entire conversation history to understand
the user's question and previous context.

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
- Only discuss hiking, trekking, trails, mountains,
  outdoor preparation, or closely related topics.
- Do not answer unrelated questions.
""",
        )
    )

    return {
        "response": result.content,
        "messages": [
            AIMessage(content=result.content)
        ],
    }


# =========================================================
# GENERAL TEMPLATE RESPONSE
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
# 10. REJECT TEMPLATE RESPONSE
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
# BUILD GRAPH
# =========================================================

workflow = StateGraph(State)


# =========================================================
# ADD NODES
# =========================================================

workflow.add_node(
    "classify",
    classify_node,
)

workflow.add_node(
    "analyze_trail",
    analyze_trail_node,
)

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
# START
# =========================================================

workflow.add_edge(
    START,
    "classify",
)


# =========================================================
# MAIN ROUTING
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
# END EDGES
# =========================================================

workflow.add_edge(
    "trail",
    END,
)

workflow.add_edge(
    "difficulty",
    END,
)

workflow.add_edge(
    "equipment",
    END,
)

workflow.add_edge(
    "safety",
    END,
)

workflow.add_edge(
    "planning",
    END,
)

workflow.add_edge(
    "general_hiking",
    END,
)

workflow.add_edge(
    "general",
    END,
)

workflow.add_edge(
    "reject",
    END,
)


checkpointer = MemorySaver()


# =========================================================
# COMPILE GRAPH
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
# STREAMING ASK ENDPOINT
# =========================================================

@router.post("/ask")
async def send_message(
    request: UserMessage,
    http_request: Request,
):
    """
    Send a message to the current hiking conversation.

    The client does NOT provide thread_id.

    The server reads thread_id from the cookie.
    """

    # =====================================================
    # GET THREAD ID FROM COOKIE
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
    # LANGGRAPH CONFIG
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

        "response": "",
    }

    # =====================================================
    # SSE EVENT GENERATOR
    # =====================================================

    async def event_generator() -> AsyncGenerator[str, None]:

        llm_tokens_sent = False
        template_sent = False

        try:
            async for event in hiking_graph.astream_events(
                input_state,
                config=config,
                version="v2",
            ):

                event_type = event["event"]
                event_name = event.get("name", "")

                if event_type == "on_chat_model_stream":

                    chunk = event["data"]["chunk"]

                    content = chunk.content

                    if not content:
                        continue

                    # Usually content is a string.
                    if isinstance(content, str):

                        text = content

                    else:

                        text = str(content)

                    llm_tokens_sent = True

                    payload = {
                        "type": "chunk",
                        "source": "llm",
                        "content": text,
                    }

                    yield (
                        "event: chunk\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                elif (
                    event_type == "on_chain_end"
                    and event_name in {
                        "general",
                        "reject",
                    }
                ):

                    if llm_tokens_sent:
                        continue

                    if template_sent:
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

                    template_sent = True

                    payload = {
                        "type": "chunk",
                        "source": "template",
                        "content": text,
                    }

                    yield (
                        "event: chunk\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

            payload = {
                "type": "done",
            }

            yield (
                "event: done\n"
                f"data: {json.dumps(payload)}\n\n"
            )

        except Exception as exc:

            payload = {
                "type": "error",
                "message": str(exc),
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(payload)}\n\n"
            )

    # =====================================================
    # SSE RESPONSE
    # =====================================================

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
