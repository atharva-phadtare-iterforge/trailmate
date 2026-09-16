import json
from uuid import uuid4

from fastapi import APIRouter, Cookie, Response
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from trailmate.agent.agent import trailmate_agent
from trailmate.agent.guardrails import validate_input, validate_output

router = APIRouter()


class AskRequest(BaseModel):
    message: str


def sse_event(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# IMPORTANT:
# Both paths are registered so this router works whether it is mounted as:
#     app.include_router(router)
# or:
#     app.include_router(router, prefix="/hiking")
#
# If you use the second form, /new-chat becomes /hiking/new-chat.
# If you use the first form, /hiking/new-chat is also available.
@router.post("/new-chat")
@router.post("/hiking/new-chat")
def new_chat(response: Response):
    thread_id = str(uuid4())

    response.set_cookie(
        key="hiking_thread_id",
        value=thread_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24,
        path="/",
    )

    return {
        "message": "New hiking chat created",
        "thread_id": thread_id,
    }


@router.post("/hiking/ask")
async def ask(
    request: AskRequest,
    response: Response,
    hiking_thread_id: str | None = Cookie(default=None),
):
    # Generate a thread for the first message when no cookie exists.
    thread_id = hiking_thread_id or str(uuid4())
    created_thread_cookie = hiking_thread_id is None

    # =========================================================
    # INPUT GUARDRAIL
    # =========================================================

    input_result = validate_input(request.message)

    if not input_result["allowed"]:

        async def rejected_stream():
            yield sse_event(
                {
                    "type": "error",
                    "message": input_result.get(
                        "message", "Request rejected."
                    ),
                    "reason": input_result.get(
                        "reason", "Input safety check failed."
                    ),
                }
            )
            yield sse_event({"type": "done"})

        rejected_response = StreamingResponse(
            rejected_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

        if created_thread_cookie:
            rejected_response.set_cookie(
                key="hiking_thread_id",
                value=thread_id,
                httponly=True,
                samesite="lax",
                secure=False,
                max_age=60 * 60 * 24,
                path="/",
            )

        return rejected_response

    async def event_stream():
        try:
            yield sse_event({"type": "agent_start"})

            input_state = {
                "messages": [HumanMessage(content=request.message)],
                "search_performed": False,
            }

            answer_started_sent = False

            # Collect the complete generated response so the output
            # guardrail can validate it before the final safety event.
            full_answer = ""

            async for event in trailmate_agent.astream(
                input_state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=["updates", "custom"],
            ):
                if not isinstance(event, tuple):
                    continue

                mode, payload = event

                if mode == "custom":
                    if not isinstance(payload, dict):
                        continue

                    if payload.get("type") == "answer_chunk":
                        text = payload.get("text", "")

                        # Collect the complete response for the
                        # output guardrail.
                        full_answer += text

                        # Direct answers do not have a tool update before
                        # streaming, so make sure the UI gets answer_start too.
                        if not answer_started_sent:
                            answer_started_sent = True
                            yield sse_event({"type": "answer_start"})

                        yield sse_event(
                            {
                                "type": "chunk",
                                "text": text,
                            }
                        )
                    continue

                if mode != "updates" or not isinstance(payload, dict):
                    continue

                if "agent" in payload:
                    agent_update = payload["agent"]
                    if not isinstance(agent_update, dict):
                        continue

                    messages = agent_update.get("messages", [])
                    if not messages:
                        continue

                    last_message = messages[-1]
                    tool_calls = getattr(last_message, "tool_calls", None)

                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name")

                            if tool_name != "search_trails_tool":
                                continue

                            query = tool_call.get("args", {}).get(
                                "query", ""
                            )

                            yield sse_event(
                                {
                                    "type": "agent_decision",
                                    "action": "search",
                                    "reason": (
                                        "The request contains searchable "
                                        "trail requirements."
                                    ),
                                }
                            )

                            yield sse_event(
                                {
                                    "type": "tool_start",
                                    "tool": tool_name,
                                    "detail": (
                                        f"Searching TrailMate for: {query}"
                                    ),
                                }
                            )
                    else:
                        yield sse_event(
                            {
                                "type": "agent_decision",
                                "action": "answer",
                                "reason": (
                                    "No TrailMate database search is needed "
                                    "for this request."
                                ),
                            }
                        )

                if "tools" in payload:
                    tool_update = payload["tools"]

                    if not isinstance(tool_update, dict):
                        continue

                    for tool_message in tool_update.get("messages", []):
                        content = str(
                            getattr(tool_message, "content", "")
                        )

                        results = []

                        try:
                            parsed_results = json.loads(content)

                            if isinstance(parsed_results, list):
                                results = parsed_results

                        except json.JSONDecodeError:
                            pass

                        yield sse_event(
                            {
                                "type": "tool_end",
                                "tool": "search_trails_tool",
                                "count": len(results),
                                "results": results,
                            }
                        )

                        if not answer_started_sent:
                            answer_started_sent = True
                            yield sse_event(
                                {"type": "answer_start"}
                            )

            # =========================================================
            # OUTPUT GUARDRAIL
            # =========================================================

            output_result = validate_output(full_answer)

            if not output_result["allowed"]:

                # Tell the frontend that the generated response
                # failed the output safety check.
                yield sse_event(
                    {
                        "type": "guardrail",
                        "status": "blocked",
                        "reason": output_result.get(
                            "reason",
                            "Output safety check failed.",
                        ),
                    }
                )

                # Send the safe fallback message instead of
                # continuing with the generated response.
                yield sse_event(
                    {
                        "type": "chunk",
                        "text": output_result.get(
                            "message",
                            "The generated response could not be displayed.",
                        ),
                    }
                )

            else:

                # The generated response passed the actual
                # output guardrail.
                yield sse_event(
                    {
                        "type": "guardrail",
                        "status": "passed",
                        "detail": (
                            "The generated response passed the "
                            "output safety check."
                        ),
                    }
                )

            yield sse_event({"type": "answer_end"})
            yield sse_event({"type": "done"})

        except Exception as exc:
            yield sse_event(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

            yield sse_event({"type": "done"})

    stream_response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

    # IMPORTANT:
    # Cookies set on the injected FastAPI Response object are NOT reliably
    # transferred when a different StreamingResponse object is returned.
    # Set the cookie on the actual response that goes over the network.
    if created_thread_cookie:
        stream_response.set_cookie(
            key="hiking_thread_id",
            value=thread_id,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=60 * 60 * 24,
            path="/",
        )

    return stream_response