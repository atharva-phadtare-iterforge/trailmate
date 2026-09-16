from typing import (
    TypedDict,
    Literal,
    Annotated,
)

from uuid import uuid4
import asyncio
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
    BaseMessage,
)

from langgraph.runtime import Runtime
from litellm import acompletion

from ..init.service_client import search_trails

from ..agent.guardrails import (
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

GUARDRAIL_CHECK_SIZE = 200


# =========================================================
# TYPES
# =========================================================

AgentAction = Literal[
    "answer",
    "search",
    "clarify"
]


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    agent_action: AgentAction

    agent_reason: str

    retrieved_trails: list

    response: str

    input_allowed: bool

    input_guardrail_message: str

    output_allowed: bool

    output_guardrail_message: str

    clarification_question: str


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
# MESSAGE HELPERS
# =========================================================

def get_latest_message(
    state: State,
) -> str:

    for message in reversed(
        state["messages"]
    ):

        if isinstance(
            message,
            HumanMessage,
        ):

            content = message.content

            if isinstance(
                content,
                str,
            ):

                return content

            return str(content)

    return ""


def message_to_dict(
    message: BaseMessage,
) -> dict:

    if isinstance(
        message,
        HumanMessage,
    ):

        return {
            "role": "user",
            "content": message.content,
        }

    if isinstance(
        message,
        AIMessage,
    ):

        return {
            "role": "assistant",
            "content": message.content,
        }

    return {
        "role": "user",
        "content": str(
            message.content
        ),
    }


# =========================================================
# CONVERSATION PROMPT
# =========================================================

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

    for message in state["messages"]:

        messages.append(
            message_to_dict(
                message
            )
        )

    return messages


# =========================================================
# INPUT GUARDRAIL
# =========================================================

def input_guardrail_node(
    state: State,
):

    message = get_latest_message(
        state
    )

    result = validate_input(
        message
    )

    print(
        "[HIKING INPUT GUARDRAIL]"
        f" allowed={result['allowed']}"
    )

    return {
        "input_allowed": result["allowed"],
        "input_guardrail_message":
            result["message"],
    }


# =========================================================
# INPUT GUARDRAIL ROUTER
# =========================================================

def route_input_guardrail(
    state: State,
):

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

    response = (
        state[
            "input_guardrail_message"
        ]
    )

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

    generation_start = time.perf_counter()

    await send_sse_event(
        stream_context,
        "answer_start",
        title="GENERATING ANSWER",
        message="Preparing your TrailMate response...",
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

            # =============================================
            # INCREMENTAL OUTPUT GUARDRAIL
            # =============================================

            if (
                len(guardrail_buffer)
                >= GUARDRAIL_CHECK_SIZE
            ):

                result = validate_output(
                    guardrail_buffer
                )

                if not result["allowed"]:

                    guardrail_message = (
                        result["message"]
                    )

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
                        "response":
                            guardrail_message,

                        "output_allowed":
                            False,

                        "output_guardrail_message":
                            guardrail_message,

                        "messages": [
                            AIMessage(
                                content=
                                    guardrail_message
                            )
                        ],
                    }

                guardrail_buffer = ""

            # =============================================
            # SEND CHUNK
            # =============================================

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

            guardrail_message = (
                final_result["message"]
            )

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
                "response":
                    guardrail_message,

                "output_allowed":
                    False,

                "output_guardrail_message":
                    guardrail_message,

                "messages": [
                    AIMessage(
                        content=
                            guardrail_message
                    )
                ],
            }

        # =================================================
        # COMPLETE
        # =================================================

        duration_ms = int(
            (time.perf_counter() - generation_start) * 1000
        )

        await send_sse_event(
            stream_context,
            "answer_end",
            title="ANSWER READY",
            message="Response generated successfully.",
            duration_ms=duration_ms,
        )

        print(
            "[HIKING LLM]"
            f" Completed response"
            f" length={len(full_response)}"
        )

        return {
            "response":
                full_response,

            "output_allowed":
                True,

            "output_guardrail_message":
                "",

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
# AGENT
# =========================================================

async def agent_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    print(
        "[HIKING AGENT]"
        " deciding what to do"
    )

    await send_sse_event(
        runtime.context,
        "agent_start",
        title="TRAILMATE AGENT",
        message="Analyzing your request...",
    )

    # =====================================================
    # BUILD CONVERSATION
    # =====================================================

    conversation = []

    for message in state["messages"]:

        if isinstance(
            message,
            HumanMessage,
        ):

            conversation.append(
                {
                    "role": "user",
                    "content": message.content,
                }
            )

        elif isinstance(
            message,
            AIMessage,
        ):

            conversation.append(
                {
                    "role": "assistant",
                    "content": message.content,
                }
            )

    # =====================================================
    # AGENT PROMPT
    # =====================================================

    agent_prompt = """
You are the decision-making agent for TrailMate,
a hiking trail assistant.

Your job is to decide what TrailMate should do with
the user's latest message.

You have exactly three possible actions:

---------------------------------------------------------
ACTION 1: "answer"
---------------------------------------------------------

Use "answer" when the question can be answered using
general hiking knowledge or the existing conversation.

Examples:

- What is hiking?
- What should I carry on a day hike?
- Why should I use hiking poles?
- How should I prepare for a hike?

---------------------------------------------------------
ACTION 2: "search"
---------------------------------------------------------

Use "search" when TrailMate's trail dataset contains
information needed to answer the request.

Examples:

- Tell me about Hidden Spring Trail.
- Which trail is easiest?
- What is the distance of Thunder Gorge?
- I want an easy trail near water.
- Which trail is good for beginners?
- What trails have waterfalls?
- Find me a short trail.

Use search when the user provides enough information
to perform a useful trail search.

---------------------------------------------------------
ACTION 3: "clarify"
---------------------------------------------------------

Use "clarify" when the user's request is too vague
to perform a useful search without making an arbitrary
assumption.

Examples:

- Find me a good trail.
- Find me a nice hike.
- I want a trail.
- Recommend something for me.

The clarification question should ask for the most
useful missing preference.

For example:

"What difficulty are you looking for — easy, moderate,
or challenging?"

---------------------------------------------------------
IMPORTANT
---------------------------------------------------------

Do not expose hidden reasoning or chain-of-thought.

The "reason" field must be a short user-safe explanation
of the decision.

Return ONLY valid JSON.

{
  "action": "answer" or "search" or "clarify",
  "reason": "short user-safe explanation",
  "clarification_question": "question if action is clarify, otherwise empty string"
}
"""

    messages = [
        {
            "role": "system",
            "content": agent_prompt,
        }
    ]

    messages.extend(
        conversation
    )

    # =====================================================
    # CALL LLM
    # =====================================================

    try:

        response = await acompletion(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            messages=messages,
            temperature=0,
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        print(
            "[HIKING AGENT]"
            f" raw response: {content}"
        )

        # =================================================
        # CLEAN RESPONSE
        # =================================================

        content = content.strip()

        if content.startswith("```"):

            lines = content.splitlines()

            if (
                lines
                and lines[0].startswith("```")
            ):

                lines = lines[1:]

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):

                lines = lines[:-1]

            content = "\n".join(
                lines
            ).strip()

        # =================================================
        # PARSE JSON
        # =================================================

        # Some small/local models occasionally omit the final
        # closing brace even when the JSON structure is otherwise
        # complete. Repair only this simple truncation case.
        if (
            content.startswith("{")
            and not content.endswith("}")
        ):
            content += "}"

        decision = json.loads(
            content
        )

        action = decision.get(
            "action"
        )

        reason = decision.get(
            "reason",
            "",
        )

        clarification_question = decision.get(
            "clarification_question",
            "",
        )

        if not isinstance(
                clarification_question,
                str,
        ):

            clarification_question = str(
                clarification_question
        )
        # =================================================
        # VALIDATE ACTION
        # =================================================

        if action not in (
            "answer",
            "search",
            "clarify",
        ):
            raise ValueError(
                "Invalid agent action: "
                f"{action}"
            )

        # =================================================
        # VALIDATE REASON
        # =================================================

        if not isinstance(
            reason,
            str,
        ):

            reason = str(
                reason
            )

        print(
            "[HIKING AGENT]"
            f" action={action}"
            f" reason={reason}"
        )

        decision_titles = {
            "answer": "ANSWER DIRECTLY",
            "search": "SEARCH REQUIRED",
            "clarify": "CLARIFICATION NEEDED",
        }

        await send_sse_event(
            runtime.context,
            "agent_decision",
            action=action,
            title=decision_titles[action],
            message=reason,
        )

        return {
            "agent_action": action,
            "agent_reason": reason,
            "clarification_question":
                clarification_question,
        }

    except Exception as exc:

        print(
            "[HIKING AGENT]"
            f" error: {exc}"
        )

        # =================================================
        # SAFE FALLBACK
        # =================================================

        return {
            "agent_action":
                "search",

            "agent_reason": (
                "The agent could not confidently "
                "determine whether the question "
                "can be answered directly, so "
                "trail search will be used."
            ),
        }


# =========================================================
# CLARIFICATION NODE
# =========================================================

async def agent_clarify_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    question = state.get(
        "clarification_question",
        "",
    )

    print(
        "[HIKING AGENT]"
        f" Asking clarification: {question}"
    )

    await send_sse_event(
        runtime.context,
        "clarify",
        title="CLARIFICATION NEEDED",
        message=question,
    )

    await send_sse_event(
        runtime.context,
        "chunk",
        source="agent",
        content=question,
    )

    return {
        "response": question,
        "messages": [
            AIMessage(
                content=question
            )
        ],
    }

# =========================================================
# SEARCH NODE
# =========================================================

async def search_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    messages = state["messages"]

    latest_message = messages[-1]

    if not isinstance(
        latest_message,
        HumanMessage,
    ):
        return {
            "retrieved_trails": []
        }

    query = latest_message.content

    search_start = time.perf_counter()

    await send_sse_event(
        runtime.context,
        "tool_start",
        tool="trail_search",
        title="TRAIL SEARCH",
        message="Calling the TrailMate search service...",
        query=query,
    )

    print(
        "[HIKING SEARCH]"
        f" Searching for: {query}"
    )

    await send_sse_event(
        runtime.context,
        "tool_progress",
        tool="trail_search",
        title="SEARCHING",
        message="Finding the most relevant trails...",
    )

    try:

        results = search_trails(
            query=query,
            top_k=3,
        )

        duration_ms = int(
            (time.perf_counter() - search_start) * 1000
        )

        print(
            "[HIKING SEARCH]"
            f" Found {len(results)} results"
        )

        await send_sse_event(
            runtime.context,
            "tool_end",
            tool="trail_search",
            title="SEARCH COMPLETE",
            message=f"Found {len(results)} matching trails.",
            result_count=len(results),
            duration_ms=duration_ms,
            results=results,
        )

        return {
            "retrieved_trails": results
        }

    except Exception as exc:

        print(
            "[HIKING SEARCH]"
            f" Error: {exc}"
        )

        await send_sse_event(
            runtime.context,
            "tool_error",
            tool="trail_search",
            message=str(exc),
        )

        return {
            "retrieved_trails": []
        }

# =========================================================
# ANSWER FROM SEARCH
# =========================================================

async def answer_from_search_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    trails = state.get(
        "retrieved_trails",
        [],
    )

    search_context = json.dumps(
        trails,
        indent=2,
        default=str,
    )

    system_prompt = f"""
You are TrailMate, a hiking trail assistant.

Answer the user's question using the trail search
results below.

The trail search results are the source of truth for
trail-specific information.

IMPORTANT:

- Do not invent trail names.
- Do not invent distances.
- Do not invent difficulty levels.
- Do not invent elevations.
- Do not invent locations.
- Do not invent trail conditions.
- Do not invent weather.
- Do not invent facts that are not present in the
  search results.

If the search results do not contain enough information
to answer the question, clearly say that the available
trail information is insufficient.

Be concise and helpful.

Trail search results:

{search_context}
"""

    return await stream_llm_response(
        state,
        system_prompt,
        runtime.context,
    )


# =========================================================
# AGENT ROUTER
# =========================================================

def route_agent(
    state: State,
):

    return state[
        "agent_action"
    ]


# =========================================================
# DIRECT ANSWER NODE
# =========================================================

async def agent_answer_node(
    state: State,
    runtime: Runtime[StreamContext],
):

    print(
        "[HIKING AGENT]"
        " answering directly"
    )

    system_prompt = """
You are TrailMate, a helpful hiking assistant.

Answer the user's latest question directly.

You can use:

- general hiking knowledge
- information already present in the conversation

Be concise, helpful, and practical.

IMPORTANT:

- Do not invent specific facts about named trails.
- Do not invent distances.
- Do not invent elevations.
- Do not invent trail conditions.
- Do not invent weather.
- Do not pretend to have live trail information.
- If specific trail information is unavailable,
  clearly say that you do not have that information.
- Only answer hiking and closely related questions.
"""

    return await stream_llm_response(
        state,
        system_prompt,
        runtime.context,
    )


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

    result = validate_output(
        response
    )

    print(
        "[HIKING OUTPUT GUARDRAIL]"
        f" allowed={result['allowed']}"
    )

    return {
        "output_allowed":
            result["allowed"],

        "output_guardrail_message":
            result["message"],
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

    response = (
        state[
            "output_guardrail_message"
        ]
    )

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
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(
    State
)


# =========================================================
# NODES
# =========================================================

workflow.add_node(
    "input_guardrail",
    input_guardrail_node,
)

workflow.add_node(
    "input_guardrail_blocked",
    input_guardrail_blocked,
)

workflow.add_node(
    "agent",
    agent_node,
)

workflow.add_node(
    "agent_answer",
    agent_answer_node,
)

workflow.add_node(
    "agent_clarify",
    agent_clarify_node,
)

workflow.add_node(
    "search",
    search_node,
)

workflow.add_node(
    "answer_from_search",
    answer_from_search_node,
)

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
        "continue":
            "agent",

        "block":
            "input_guardrail_blocked",
    },
)


workflow.add_edge(
    "input_guardrail_blocked",
    END,
)


# =========================================================
# AGENT ROUTING
# =========================================================

workflow.add_conditional_edges(
    "agent",
    route_agent,
    {
        "answer":
            "agent_answer",

        "search":
            "search",

        "clarify":
            "agent_clarify",
    },
)


# =========================================================
# DIRECT ANSWER → OUTPUT GUARDRAIL
# =========================================================

workflow.add_edge(
    "agent_answer",
    "output_guardrail",
)


# =========================================================
# SEARCH → ANSWER FROM SEARCH
# =========================================================

workflow.add_edge(
    "search",
    "answer_from_search",
)

# =========================================================
# CLARIFY → ANSWER FROM SEARCH
# =========================================================
workflow.add_edge(
    "agent_clarify",
    "output_guardrail",
)


# =========================================================
# SEARCH ANSWER → OUTPUT GUARDRAIL
# =========================================================

workflow.add_edge(
    "answer_from_search",
    "output_guardrail",
)


# =========================================================
# OUTPUT GUARDRAIL ROUTING
# =========================================================

workflow.add_conditional_edges(
    "output_guardrail",
    route_output_guardrail,
    {
        "allow":
            END,

        "block":
            "output_guardrail_blocked",
    },
)


workflow.add_edge(
    "output_guardrail_blocked",
    END,
)


# =========================================================
# CHECKPOINTER
# =========================================================

checkpointer = MemorySaver()


hiking_graph = workflow.compile(
    checkpointer=checkpointer
)


# =========================================================
# NEW CHAT
# =========================================================

@router.post(
    "/new-chat"
)
def new_chat(
    response: Response,
):

    thread_id = str(
        uuid4()
    )

    response.set_cookie(
        key="hiking_thread_id",
        value=thread_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24,
    )

    return {
        "message":
            "New hiking chat created",
    }


# =========================================================
# ASK
# =========================================================

@router.post(
    "/ask"
)
async def send_message(
    request: UserMessage,
    http_request: Request,
):

    # =====================================================
    # THREAD
    # =====================================================

    thread_id = (
        http_request
        .cookies
        .get(
            "hiking_thread_id"
        )
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

    stream_context = (
        StreamContext()
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
    # INPUT STATE
    # =====================================================

    input_state = {

        "messages": [
            HumanMessage(
                content=request.message
            )
        ],

        "agent_action":
            "search",

        "agent_reason":
            "",

        "clarification_question":
            "",

        "retrieved_trails":
            [],

        "response":
            "",

        "input_allowed":
            True,

        "input_guardrail_message":
            "",

        "output_allowed":
            True,

        "output_guardrail_message":
            "",
    }

    # =====================================================
    # SSE GENERATOR
    # =====================================================

    async def event_generator():

        request_start = (
            time.perf_counter()
        )

        # =================================================
        # START GRAPH
        # =================================================

        graph_task = (
            asyncio.create_task(
                hiking_graph.ainvoke(
                    input_state,
                    config=config,
                    context=stream_context,
                )
            )
        )

        try:

            # =============================================
            # CONSUME SSE EVENTS
            # =============================================

            while True:

                # -----------------------------------------
                # GRAPH FINISHED
                # -----------------------------------------

                if (
                    graph_task.done()
                    and
                    stream_context.queue.empty()
                ):

                    exception = (
                        graph_task.exception()
                    )

                    if exception:

                        raise exception

                    break

                # -----------------------------------------
                # WAIT FOR EVENT
                # -----------------------------------------

                try:

                    event = (
                        await asyncio.wait_for(
                            stream_context.queue.get(),
                            timeout=0.5,
                        )
                    )

                except asyncio.TimeoutError:

                    if graph_task.done():

                        exception = (
                            graph_task.exception()
                        )

                        if exception:

                            raise exception

                        break

                    continue

                event_type = (
                    event["type"]
                )

                # =========================================
                # LLM CHUNK
                # =========================================

                if event_type == "chunk":

                    yield (
                        "event: chunk\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                # =========================================
                # AGENT START
                # =========================================

                elif event_type == "agent_start":

                    yield (
                        "event: agent_start\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # AGENT DECISION
                # =========================================

                elif event_type == "agent_decision":

                    yield (
                        "event: agent_decision\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # CLARIFICATION
                # =========================================

                elif event_type == "clarify":

                    yield (
                        "event: clarify\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # TOOL START
                # =========================================

                elif event_type == "tool_start":

                    yield (
                        "event: tool_start\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                # =========================================
                # TOOL PROGRESS
                # =========================================

                elif event_type == "tool_progress":

                    yield (
                        "event: tool_progress\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # TOOL END
                # =========================================

                elif event_type == "tool_end":

                    yield (
                        "event: tool_end\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                # =========================================
                # TOOL ERROR
                # =========================================

                elif event_type == "tool_error":

                    yield (
                        "event: tool_error\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                # =========================================
                # ANSWER START
                # =========================================

                elif event_type == "answer_start":

                    yield (
                        "event: answer_start\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # ANSWER END
                # =========================================

                elif event_type == "answer_end":

                    yield (
                        "event: answer_end\\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\\n\\n"
                    )

                # =========================================
                # GUARDRAIL
                # =========================================

                elif (
                    event_type
                    == "guardrail_block"
                ):

                    yield (
                        "event: guardrail_block\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                # =========================================
                # ERROR
                # =========================================

                elif (
                    event_type
                    == "error"
                ):

                    yield (
                        "event: error\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                    break

                # =========================================
                # STREAM DONE
                # =========================================

                elif (
                    event_type
                    == "stream_done"
                ):

                    yield (
                        "event: stream_done\n"
                        f"data: "
                        f"{json.dumps(event)}"
                        "\n\n"
                    )

                    continue

            # =================================================
            # WAIT FOR COMPLETE GRAPH
            # =================================================

            result = await graph_task

            # =================================================
            # TIMING
            # =================================================

            total_time = (
                time.perf_counter()
                - request_start
            )

            print(
                "[HIKING]"
                f" Completed in "
                f"{total_time:.3f}s"
            )

            # =================================================
            # FINAL AGENT DECISION
            # =================================================

            print(
                "[HIKING]"
                f" Final agent_action="
                f"{result.get('agent_action')}"
            )

            print(
                "[HIKING]"
                f" Final agent_reason="
                f"{result.get('agent_reason')}"
            )

            # =================================================
            # FINAL DONE EVENT
            # =================================================

            done_data = {

                "type":
                    "done",

                "agent_action":
                    result.get(
                        "agent_action"
                    ),

                "agent_reason":
                    result.get(
                        "agent_reason"
                    ),

                "total_time":
                    total_time,
            }

            yield (
                "event: done\n"
                f"data: "
                f"{json.dumps(done_data)}"
                "\n\n"
            )

        # =====================================================
        # CLIENT CANCELLED
        # =====================================================

        except asyncio.CancelledError:

            if not graph_task.done():

                graph_task.cancel()

                try:

                    await graph_task

                except asyncio.CancelledError:

                    pass

            raise

        # =====================================================
        # ERROR
        # =====================================================

        except Exception as exc:

            print(
                "[HIKING]"
                f" Streaming error: {exc}"
            )

            if not graph_task.done():

                graph_task.cancel()

                try:

                    await graph_task

                except asyncio.CancelledError:

                    pass

    # =====================================================
    # SSE RESPONSE
    # =====================================================

    return StreamingResponse(

        event_generator(),

        media_type=
            "text/event-stream",

        headers={

            "Cache-Control":
                "no-cache, no-transform",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no",
        },
    )