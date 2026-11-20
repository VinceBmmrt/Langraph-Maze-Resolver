"""Web server: level data, the graph diagram, and a live SSE stream of the cat's run."""

import uuid
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from maze_agent.agent import LIVES, graph
from maze_agent.brain import Brain, LLMBrain
from maze_agent.config import MODEL
from maze_agent.levels import LEVELS, Level, build
from maze_agent.runner import play

STATIC = Path(__file__).parent.parent / "static"

app = FastAPI(title="Maze Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@cache
def get_brain() -> Brain:
    return LLMBrain(MODEL)


def level_or_404(number: int) -> Level:
    if not 1 <= number <= len(LEVELS):
        raise HTTPException(404, f"No level {number}")
    return LEVELS[number - 1]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/levels")
def levels() -> dict:
    return {"model": MODEL, "lives": LIVES, "levels": [{"number": lv.number, "title": lv.title} for lv in LEVELS]}


@app.get("/api/levels/{number}")
def level(number: int) -> dict:
    lv = level_or_404(number)
    maze, world = build(lv)
    return {"number": lv.number, "title": lv.title, "maze": maze.to_dict(), "world": world.model_dump(mode="json")}


@app.get("/api/graph")
def graph_diagram() -> dict:
    return {"mermaid": graph.get_graph().draw_mermaid()}


@app.get("/api/levels/{number}/play", response_class=EventSourceResponse)
async def play_level(number: int, brain: Annotated[Brain, Depends(get_brain)]) -> AsyncIterator[ServerSentEvent]:
    """Stream the run; the browser closing the stream cancels the graph."""
    lv = level_or_404(number)
    async for event in play(lv, brain, str(uuid.uuid4())):
        yield ServerSentEvent(data=event, event=event["type"])
    yield ServerSentEvent(data={}, event="done")
