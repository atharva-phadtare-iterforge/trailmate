from fastapi import FastAPI
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from .hiking.graph import router as hiking_router


app = FastAPI()

class State(TypedDict):
    message: str
    response: str

class MessageRequest(BaseModel):
    message: str

def entry_node(state: State):
    return {
        "message": state["message"]
    }


def processing_node(state: State):
    message = state["message"]

    return {
        "response": f"TrailMate received your message: {message}"
    }


def response_node(state: State):
    return {
        "response": state["response"]
    }


builder = StateGraph(State)
builder.add_node("entry", entry_node)
builder.add_node("processing", processing_node)
builder.add_node("response", response_node)

builder.add_edge(START, "entry")
builder.add_edge("entry", "processing")
builder.add_edge("processing", "response")
builder.add_edge("response", END)

graph = builder.compile()

@app.post("/message")
def send_message(request: MessageRequest):
    result = graph.invoke({
        "message": request.message,
        "response": ""
    })

    return {
        "message": result["message"],
        "response": result["response"]
    }

app.include_router(hiking_router)