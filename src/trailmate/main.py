from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from .hiking.graph import router as hiking_router
from fastapi.responses import StreamingResponse


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="TrailMate",
    description="AI hiking assistant",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

HIKING_DIR = BASE_DIR / "hiking"

TEMPLATES_DIR = HIKING_DIR / "templates"

STATIC_DIR = HIKING_DIR / "static"


# =========================================================
# STATIC FILES
# =========================================================

app.mount("/hiking/static",StaticFiles(directory=STATIC_DIR),name="hiking-static")


# =========================================================
# STATE
# =========================================================

class State(TypedDict):
    message: str
    response: str


# =========================================================
# REQUEST MODEL
# =========================================================

class MessageRequest(BaseModel):
    message: str


llm = ChatOpenAI(
    model="hiking-small",
    temperature=0,
    api_key="sk-1234",
    base_url="http://localhost:4000/v1",
    streaming=True,
)


# =========================================================
# SIMPLE TEST GRAPH
# =========================================================

def entry_node(state: State):

    return {
        "message": state["message"]
    }


async def processing_node(state: State):

    answer = ""
    message = state["message"]

    async for chunk in llm.astream(message):
        answer += chunk.content

    return {
        "response":
            f"{answer}"
    }


def response_node(state: State):

    return {
        "response": state["response"]
    }


builder = StateGraph(State)


builder.add_node(
    "entry",
    entry_node,
)

builder.add_node(
    "processing",
    processing_node,
)

builder.add_node(
    "response",
    response_node,
)


builder.add_edge(
    START,
    "entry",
)

builder.add_edge(
    "entry",
    "processing",
)

builder.add_edge(
    "processing",
    "response",
)

builder.add_edge(
    "response",
    END,
)


graph = builder.compile()


# =========================================================
# SIMPLE MESSAGE API
# =========================================================

@app.post("/message")
async def send_message(request: MessageRequest):

    async def event_generator():

        async for event in graph.astream_events(
            {
                "message": request.message,
                "response": "",
            },
            version="v2",
        ):

            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )



# =========================================================
# FRONTEND
# =========================================================

@app.get("/")
async def home():

    return FileResponse( TEMPLATES_DIR/"index.html")


# =========================================================
# HIKING ROUTER
# =========================================================

app.include_router(
    hiking_router
)
