"""Web server: levels, models, the graph diagram, eval results, and live SSE streams of runs."""

import json
import uuid
from collections.abc import AsyncIterator, Callable
from functools import cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from maze_agent.agent import STORE, graph, lessons_namespace
from maze_agent.brain import Brain, LLMBrain
from maze_agent.config import MODEL
from maze_agent.levels import LEVELS, Level, build
from maze_agent.models import MODELS
from maze_agent.runner import context, play

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "static"
EVAL_RESULTS = ROOT / "evals" / "results.json"

app = FastAPI(title="Maze Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
runs: dict[str, tuple[int, str]] = {}  # thread_id -> (level, model), to resume paused runs


@cache
def llm_brain(model: str) -> Brain:
    return LLMBrain(model)


def brain_factory() -> Callable[[str], Brain]:
    """Dependency: tests swap it for a scripted brain."""
    return llm_brain


Brains = Annotated[Callable[[str], Brain], Depends(brain_factory)]


def level_or_404(number: int) -> Level:
    if not 1 <= number <= len(LEVELS):
        raise HTTPException(404, f"No level {number}")
    return LEVELS[number - 1]


def model_or_400(model: str = MODEL) -> str:
    if model not in MODELS or not MODELS[model].available:
        raise HTTPException(400, f"Model {model} is unknown or has no API key")
    return model


# Validated as dependencies, so errors are plain HTTP errors raised before any event is streamed.
ValidLevel = Annotated[Level, Depends(level_or_404)]
ValidModel = Annotated[str, Depends(model_or_400)]


def run_or_404(thread_id: str) -> tuple[int, str]:
    if thread_id not in runs:
        raise HTTPException(404, "Unknown run")
    return runs[thread_id]


def memory_key(level: int, model: str) -> str:
    return f"level-{level}:{model}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/levels")
def levels() -> dict:
    models = [{"id": m.id, "label": m.label, "available": m.available} for m in MODELS.values()]
    return {
        "default_model": MODEL,
        "models": models,
        "levels": [{"number": lv.number, "title": lv.title, "lives": lv.lives} for lv in LEVELS],
    }


@app.get("/api/levels/{number}")
def level(number: int) -> dict:
    lv = level_or_404(number)
    maze, world = build(lv)
    return {
        "number": lv.number,
        "title": lv.title,
        "lives": lv.lives,
        "maze": maze.to_dict(),
        "world": world.model_dump(mode="json"),
    }


@app.get("/api/lessons")
async def lessons(level: int, model: str) -> dict:
    items = await STORE.asearch(lessons_namespace(memory_key(level, model)), limit=100)
    items.sort(key=lambda item: item.created_at)
    return {"lessons": [item.value["lesson"] for item in items]}


@app.delete("/api/lessons")
async def forget(level: int, model: str) -> dict:
    namespace = lessons_namespace(memory_key(level, model))
    for item in await STORE.asearch(namespace, limit=100):
        await STORE.adelete(namespace, item.key)
    return {"lessons": []}


@app.get("/api/graph")
def graph_diagram() -> dict:
    return {"mermaid": graph.get_graph().draw_mermaid()}


@app.get("/api/eval")
def eval_results() -> dict:
    return json.loads(EVAL_RESULTS.read_text()) if EVAL_RESULTS.exists() else {"rows": []}


@app.get("/api/levels/{number}/play", response_class=EventSourceResponse)
async def play_level(lv: ValidLevel, model: ValidModel, brains: Brains) -> AsyncIterator[ServerSentEvent]:
    """Start a run. The browser closing the stream cancels the graph."""
    thread_id = str(uuid.uuid4())
    runs[thread_id] = (lv.number, model)
    yield ServerSentEvent(data={"thread_id": thread_id, "model": model}, event="start")
    async for event in _stream(lv, model, brains, thread_id):
        yield event


@app.get("/api/runs/{thread_id}/resume", response_class=EventSourceResponse)
async def resume_run(
    run: Annotated[tuple[int, str], Depends(run_or_404)], answer: str, brains: Brains, thread_id: str
) -> AsyncIterator[ServerSentEvent]:
    """Answer the cat's call for help: the paused graph resumes from its checkpoint."""
    number, model = run
    async for event in _stream(LEVELS[number - 1], model, brains, thread_id, resume=answer):
        yield event


async def _stream(
    lv: Level, model: str, brains: Callable[[str], Brain], thread_id: str, resume: str | None = None
) -> AsyncIterator[ServerSentEvent]:
    ctx = context(lv, brains(model), memory_key=memory_key(lv.number, model), ask_human=True)
    async for event in play(lv, ctx, thread_id, resume=resume, tokens=True):
        yield ServerSentEvent(data=event, event=event["type"])
    yield ServerSentEvent(data={}, event="done")
