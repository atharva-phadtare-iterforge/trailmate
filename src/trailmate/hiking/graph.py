from typing import TypedDict, Literal
from fastapi import APIRouter
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel


router = APIRouter(
    prefix="/hiking",
    tags=["hiking"]
)

class UserMessage(BaseModel):
    message: str

# -------------------------
# State
# -------------------------

class State(TypedDict):
    message: str
    category: Literal["trail", "general", "reject"]
    response: str

# -------------------------
# Classify Node
# -------------------------

def classify_node(state: State):
    message = state['message']

    trail_keywords = ["trail","hike","hiking","trek","trekking","mountain","route","waterfall","summit","difficulty",]

    general_keywords = ["hello","hi","hey","thanks","thank you","good morning","good evening",]

    if any(keyword in message for keyword in trail_keywords):
        return {
            "category" : "trail"
        }

    elif any(keyword in message for keyword in general_keywords):
        return {
            "category": "general"
        }

    else:
        return {
            "category": "reject"
    }

# -------------------------
# Response Nodes
# -------------------------

def trail_response(state: State):
    return {
        "response": (
            "TrailMate can help you with hiking trails, "
            "including trail difficulty, routes, and hiking locations."
        )
    }


def general_response(state: State):
    return {
        "response": (
            "Hello! I'm TrailMate. "
            "I can help you with hiking trails and related questions."
        )
    }


def reject_response(state: State):
    return {
        "response": (
            "I'm sorry, but I can only help with "
            "hiking trails and related questions."
        )
    }


# -------------------------
# Routing
# -------------------------

def route_message(state: State):
    return state["category"]

workflow = StateGraph(State)

workflow.add_node("classify", classify_node)
workflow.add_node("trail", trail_response)
workflow.add_node("general", general_response)
workflow.add_node("reject", reject_response)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_message,
    {
        "trail": "trail",
        "general": "general",
        "reject": "reject",
    }
)

workflow.add_edge("trail", END)
workflow.add_edge("general", END)
workflow.add_edge("reject", END)

hiking_graph = workflow.compile()


@router.post("/ask")
def send_message(request: UserMessage):
    result = hiking_graph.invoke({
        "message" : request.message,
        "category": "",
        "response": ""
    })

    return {
        "message": result["message"],
        "category": result["category"],
        "response": result["response"]
    }