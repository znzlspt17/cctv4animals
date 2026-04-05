"""FastAPI web dashboard application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.requests import Request

from config import settings
from database import get_db, get_events, get_last_state, init_db

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    camera_id: str
    current_count: int
    in_count: int
    out_count: int


class EventResponse(BaseModel):
    id: int
    person_id: int
    camera_id: str
    direction: str
    count_change: int
    current_count: int
    confidence: float | None = None
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None
    snapshot_path: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    fps: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages multiple WebSocket connections for broadcasting."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WS client connected (total={})", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WS client disconnected (total={})", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Send JSON message to all connected clients."""
        stale: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


counter_manager = ConnectionManager()
stream_manager = ConnectionManager()

# Global reference – main.py can inject a PerformanceMonitor instance here.
_performance_monitor: Any = None


def set_performance_monitor(monitor: Any) -> None:
    """Called by main.py to inject the PerformanceMonitor instance."""
    global _performance_monitor
    _performance_monitor = monitor


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard starting — initializing DB …")
    init_db()
    yield
    logger.info("Dashboard shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="People Counter Dashboard", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status", response_model=StatusResponse)
def api_status(
    camera_id: str = Query(default=settings.CAMERA_ID),
    db: Session = Depends(get_db),
):
    state = get_last_state(db, camera_id)
    return StatusResponse(
        camera_id=camera_id,
        current_count=state["current_count"],
        in_count=state["in_count"],
        out_count=state["out_count"],
    )


@app.get("/api/events", response_model=list[EventResponse])
def api_events(
    camera_id: str = Query(default=settings.CAMERA_ID),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    events = get_events(db, camera_id, start_time=start, end_time=end, limit=limit)
    return events


@app.get("/api/stats", response_model=StatsResponse)
def api_stats():
    if _performance_monitor is None:
        return StatsResponse()
    stats = _performance_monitor.get_stats()
    return StatsResponse(**stats)


# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------


@app.websocket("/ws/counter")
async def ws_counter(websocket: WebSocket):
    await counter_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; clients only receive broadcasts
            await websocket.receive_text()
    except WebSocketDisconnect:
        counter_manager.disconnect(websocket)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await stream_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
