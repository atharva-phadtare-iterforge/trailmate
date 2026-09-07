from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .hiking.graph import router as hiking_router


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

app.mount(
    "/hiking/static",
    StaticFiles(directory=STATIC_DIR),
    name="hiking-static",
)


# =========================================================
# FRONTEND
# =========================================================

@app.get("/")
async def home():
    return FileResponse(TEMPLATES_DIR / "index.html")


# =========================================================
# HIKING ROUTER
# =========================================================

app.include_router(hiking_router)