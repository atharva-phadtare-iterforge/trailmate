# TrailMate

TrailMate is a simple hiking assistant built with **FastAPI** and **LangGraph**.

## Features

- Rule-based message classification
- `trail` — hiking-related questions
- `general` — greetings/chit-chat
- `reject` — unrelated messages
- LangGraph conditional routing
- Separate `/hiking` API router

## Flow

Message → Classify → Trail / General / Reject → Response

## Run

```powershell
uv add fastapi uvicorn langgraph pydantic
```

```powershell
uvicorn trailmate.main:app --reload
```


### Swagger UI:

```powershell
http://127.0.0.1:8000/docs
```

#### Endpoint

```powershell
POST /hiking/ask
```


### Example request:

```powershell
{
  "message": "I want to find a hiking trail"
}
```

### TrailMate uses plain Python keyword rules for classification, with no model involved.