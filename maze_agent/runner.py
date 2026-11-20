"""Run one level and turn graph updates into JSON-ready events (shared by the CLI and the web server)."""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig

from maze_agent.agent import Context, graph, start_state
from maze_agent.brain import Brain
from maze_agent.levels import Level, build


async def play(level: Level, brain: Brain, thread_id: str) -> AsyncIterator[dict[str, Any]]:
    """Yield a `think` event per LLM decision and an `act` event per move."""
    maze, world = build(level)
    ctx = Context(maze=maze, brain=brain)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 3 * ctx.max_steps}
    async for chunk in graph.astream(start_state(maze, world), config, context=ctx, stream_mode="updates"):
        for node, update in chunk.items():
            yield {"type": node, **_jsonable(update)}


def _jsonable(update: dict[str, Any]) -> dict[str, Any]:
    world = update.get("world")
    if world is not None:
        update = {**update, "world": world.model_dump(mode="json")}
    return {k: v for k, v in update.items() if k != "visits"}
