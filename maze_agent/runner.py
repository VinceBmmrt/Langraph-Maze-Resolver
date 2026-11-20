"""Run one level and turn the graph stream into JSON-ready events (shared by the CLI, the server and the eval)."""

from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StreamMode

from maze_agent.agent import Context, graph, start_state
from maze_agent.brain import Brain
from maze_agent.levels import Level, build


def context(level: Level, brain: Brain, *, memory_key: str, ask_human: bool = False, learn: bool = True) -> Context:
    maze, _ = build(level)
    return Context(
        maze=maze, brain=brain, max_steps=level.max_steps, memory_key=memory_key, ask_human=ask_human, learn=learn
    )


async def play(
    level: Level, ctx: Context, thread_id: str, *, resume: str | None = None, tokens: bool = False
) -> AsyncIterator[dict[str, Any]]:
    """Yield `think`, `act`, `ask_human`, `reflect` events, `interrupt` when the cat asks for help,
    and (with `tokens`) a `token` event per streamed LLM chunk of the `think` node.

    Pass `resume` (the player's answer) to continue a run paused by an interrupt.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 4 * ctx.max_steps}
    inputs = Command(resume=resume) if resume is not None else start_state(ctx.maze, build(level)[1], level.lives)
    modes: list[StreamMode] = ["updates", "messages"] if tokens else ["updates"]
    async for chunk in graph.astream(inputs, config, context=ctx, stream_mode=modes, version="v2"):
        if chunk["type"] == "messages":
            message, meta = chunk["data"]
            if meta.get("langgraph_node") == "think" and (text := _text(message)):
                yield {"type": "token", "text": text}
            continue
        for node, update in cast(dict[str, Any], chunk["data"]).items():
            if node == "__interrupt__":
                yield {"type": "interrupt", **update[0].value}
            elif update:
                yield {"type": node, **_jsonable(update)}


def _text(message: Any) -> str:
    """Text delta of a streamed chunk: plain content (OpenAI) or text blocks (Anthropic)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def _jsonable(update: dict[str, Any]) -> dict[str, Any]:
    world = update.get("world")
    if world is not None:
        update = {**update, "world": world.model_dump(mode="json")}
    return {k: v for k, v in update.items() if k not in ("visits", "recent")}
