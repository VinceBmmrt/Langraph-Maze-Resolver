"""The LangGraph agent.

START -> think -> act -> think ...          the cat explores
                     -> ask_human -> think  stuck: pause with interrupt() until the player answers
                     -> reflect -> END      failed: write a lesson to the long-term Store
                     -> END                 escaped
"""

import uuid
from dataclasses import dataclass
from operator import add
from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel

from maze_agent.brain import Brain, Usage
from maze_agent.maze import Maze, Pos
from maze_agent.models import cost
from maze_agent.perception import mental_map, observe
from maze_agent.world import World, move_cat, move_dogs

LIVES = 3
LLM_TIMEOUT = 30
MAX_LESSONS = 3
ASK_COOLDOWN = 15  # steps between two calls for help

Status = Literal["running", "won", "dead", "out_of_steps"]
Next = Literal["think", "ask_human", "reflect", "__end__"]


class State(BaseModel):
    pos: Pos = (0, 0)
    world: World = World()
    lives: int = LIVES
    max_lives: int = LIVES
    steps: int = 0
    visits: dict[Pos, int] = {}
    recent: list[Pos] = []  # last positions, to notice the cat going round in circles
    observation: str = ""
    direction: str = ""
    thought: str = ""
    event: str = ""
    status: Status = "running"
    advice: str = ""  # the player's answer to the last call for help
    asked_at: int = -ASK_COOLDOWN
    incidents: Annotated[list[str], add] = []  # reducer: injuries are appended, never replaced
    lesson: str = ""
    tokens: Annotated[int, add] = 0
    cost: Annotated[float, add] = 0.0


@dataclass(frozen=True)
class Context:
    maze: Maze
    brain: Brain
    max_steps: int = 300
    memory_key: str = "default"  # Store namespace for lessons: one per level (and per eval run)
    ask_human: bool = False  # only the web UI has a human to ask
    learn: bool = True


async def think(state: State, runtime: Runtime[Context]) -> dict:
    """Ask the brain for the next move, with the lessons from past attempts and any human advice."""
    ctx = runtime.context
    lessons = await _lessons(runtime) if ctx.learn else []
    observation = observe(
        ctx.maze,
        state.world,
        state.pos,
        steps=state.steps,
        lives=f"{state.lives}/{state.max_lives}",
        visits=state.visits,
        event=state.event,
        lessons=lessons,
        advice=state.advice,
    )
    decision, usage = await ctx.brain.decide(observation)
    return {
        "observation": observation,
        "direction": decision.direction,
        "thought": decision.thought,
        "advice": "",
        **_spend(ctx.brain.model, usage),
    }


def act(state: State, runtime: Runtime[Context]) -> Command[Next]:
    """Apply the move and the world rules, then choose where the graph goes next."""
    ctx = runtime.context
    maze, steps = ctx.maze, state.steps + 1
    outcome = move_cat(maze, state.world, state.pos, state.direction)
    world, bites = move_dogs(maze, outcome.world, outcome.pos, steps)
    lives = max(state.lives - outcome.hurt - bites, 0)
    event = "dog" if bites else outcome.event
    status = _status(outcome.pos == maze.goal, lives, steps, ctx.max_steps)
    update = {
        "pos": outcome.pos,
        "world": world,
        "lives": lives,
        "steps": steps,
        "visits": _visit(state.visits, outcome.pos) if outcome.pos != state.pos else state.visits,
        "recent": [*state.recent, outcome.pos][-8:],
        "event": event,
        "status": status,
    }
    if event in ("spikes", "dog"):
        update["incidents"] = [f"step {steps}: {event} at {outcome.pos}"]
    return Command(update=update, goto=_next(state, ctx, status, outcome.pos, world, update))


def _next(state: State, ctx: Context, status: Status, pos: Pos, world: World, update: dict) -> Next:
    if status == "won":
        return "__end__"
    if status != "running":
        return "reflect" if ctx.learn else "__end__"
    if ctx.ask_human and update["steps"] - state.asked_at >= ASK_COOLDOWN:
        no_lead = mental_map(ctx.maze, world, pos, update["visits"]) is None
        circling = len(update["recent"]) == 8 and len(set(update["recent"])) <= 2
        if no_lead or circling:
            return "ask_human"
    return "think"


def ask_human(state: State) -> dict:
    """Human in the loop: pause the run until the player answers (resumed with Command(resume=...))."""
    answer = interrupt({"question": "I'm stuck! Which way should I go?", "pos": state.pos, "step": state.steps})
    return {"advice": str(answer)[:200], "asked_at": state.steps}


async def reflect(state: State, runtime: Runtime[Context]) -> dict:
    """After a failure, write one lesson to the long-term Store for the next attempt."""
    ctx = runtime.context
    lesson, usage = await ctx.brain.reflect(_summary(state, ctx.maze))
    await _store(runtime).aput(lessons_namespace(ctx.memory_key), str(uuid.uuid4()), {"lesson": lesson})
    return {"lesson": lesson, **_spend(ctx.brain.model, usage)}


def _summary(state: State, maze: Maze) -> str:
    world = state.world
    lines = [
        f"Result: {state.status} after {state.steps} steps, at {state.pos}. Exit at {maze.goal}.",
        f"You explored {len(state.visits)} of {maze.size * maze.size} cells.",
    ]
    if world.door:
        lines.append(
            f"Locked door at {world.door}. " + ("You had the key." if world.has_key else "You never found the key.")
        )
    lines += state.incidents or ["You were never hurt."]
    return "\n".join(lines)


async def _lessons(runtime: Runtime[Context]) -> list[str]:
    items = await _store(runtime).asearch(lessons_namespace(runtime.context.memory_key), limit=100)
    items.sort(key=lambda item: item.created_at)
    return [item.value["lesson"] for item in items[-MAX_LESSONS:]]


def lessons_namespace(memory_key: str) -> tuple[str, str]:
    """Store namespace of the lessons for one memory key (namespace labels cannot contain dots)."""
    return ("lessons", memory_key.replace(".", "_"))


def _store(runtime: Runtime[Context]) -> BaseStore:
    if runtime.store is None:
        raise RuntimeError("the graph must be compiled with a store")
    return runtime.store


def _spend(model: str, usage: Usage) -> dict:
    return {
        "tokens": usage.input_tokens + usage.output_tokens,
        "cost": cost(model, usage.input_tokens, usage.output_tokens),
    }


def _visit(visits: dict[Pos, int], cell: Pos) -> dict[Pos, int]:
    return {**visits, cell: visits.get(cell, 0) + 1}


def _status(escaped: bool, lives: int, steps: int, max_steps: int) -> Status:
    if escaped:
        return "won"
    if lives <= 0:
        return "dead"
    return "out_of_steps" if steps >= max_steps else "running"


SERDE = JsonPlusSerializer(allowed_msgpack_modules=[("maze_agent.world", "World"), ("maze_agent.world", "Dog")])
STORE = InMemoryStore()

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(think, timeout=LLM_TIMEOUT, retry_policy=RetryPolicy(max_attempts=3))
    .add_node(act, destinations=("think", "ask_human", "reflect", "__end__"))
    .add_node(ask_human)
    .add_node(reflect, timeout=LLM_TIMEOUT, retry_policy=RetryPolicy(max_attempts=3))
    .add_edge(START, "think")
    .add_edge("think", "act")
    .add_edge("ask_human", "think")
    .add_edge("reflect", "__end__")
    .compile(checkpointer=InMemorySaver(serde=SERDE), store=STORE)
)


def start_state(maze: Maze, world: World, lives: int = LIVES) -> State:
    return State(pos=maze.start, world=world, visits={maze.start: 1}, lives=lives, max_lives=lives)
