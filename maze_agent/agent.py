"""LangGraph agent: `think` (LLM) -> `act` (world rules) -> `think` ... until the cat escapes or dies."""

from dataclasses import dataclass
from operator import add
from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy
from pydantic import BaseModel

from maze_agent.brain import Brain
from maze_agent.maze import Maze, Pos, nearest_unvisited, step
from maze_agent.world import World, move_cat, move_dogs

LIVES = 3
HEARING = 3
LLM_TIMEOUT = 30

Status = Literal["running", "won", "dead", "out_of_steps"]

EVENTS = {
    "bump": "you bumped into a wall",
    "locked": "the door is locked, you need the key",
    "spikes": "you stepped on spikes and lost a life",
    "dog": "a dog bit you and you lost a life",
    "key": "you picked up the key",
}


class State(BaseModel):
    pos: Pos = (0, 0)
    world: World = World()
    lives: int = LIVES
    steps: int = 0
    visits: dict[Pos, int] = {}
    observation: str = ""
    direction: str = ""
    thought: str = ""
    event: str = ""
    status: Status = "running"
    tokens: Annotated[int, add] = 0


@dataclass(frozen=True)
class Context:
    maze: Maze
    brain: Brain
    max_steps: int = 300


def observe(state: State, maze: Maze) -> str:
    """What the cat perceives, as text for the LLM."""
    world, pos = state.world, state.pos
    lines = [f"You are at {pos}. The exit is at {maze.goal}. Step {state.steps}. Lives: {state.lives}/{LIVES}."]
    if world.door:
        lines.append("You carry the key." if world.has_key else "You do not have the key yet.")
    if state.event in EVENTS:
        lines.append(f"Last turn {EVENTS[state.event]}.")
    lines.append("Open directions:")
    for d in maze.exits(pos):
        cell = step(pos, d)
        lines.append(f"- {d} -> {cell}, visited {state.visits.get(cell, 0)} times{_hazard(world, maze, cell)}")
    lines += [f"You hear a dog at {dog.pos}." for dog in world.dogs if 1 < _distance(dog.pos, pos) <= HEARING]
    if hint := nearest_unvisited(maze, pos, set(state.visits), world.locked | set(world.spikes)):
        cell, dist, first = hint
        lines.append(f"Mental map: nearest unexplored opening is {cell}, {dist} steps away (go {first} first).")
    return "\n".join(lines)


def _hazard(world: World, maze: Maze, cell: Pos) -> str:
    if cell == maze.goal:
        return " EXIT!"
    if world.dog_at(cell):
        return " DOG!"
    if cell in world.spikes:
        return " SPIKES!"
    if cell == world.key:
        return " KEY"
    if cell == world.door:
        return " open door" if world.has_key else " LOCKED DOOR"
    return ""


def _distance(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


async def think(state: State, runtime: Runtime[Context]) -> dict:
    """Ask the brain for the next direction."""
    observation = observe(state, runtime.context.maze)
    decision, tokens = await runtime.context.brain.decide(observation)
    return {"observation": observation, "direction": decision.direction, "thought": decision.thought, "tokens": tokens}


def act(state: State, runtime: Runtime[Context]) -> Command[Literal["think", "__end__"]]:
    """Apply the move, let the dogs play, then loop or stop."""
    maze, steps = runtime.context.maze, state.steps + 1
    outcome = move_cat(maze, state.world, state.pos, state.direction)
    world, bites = move_dogs(maze, outcome.world, outcome.pos, steps)
    lives = state.lives - outcome.hurt - bites
    status = _status(outcome.pos == maze.goal, lives, steps, runtime.context.max_steps)
    update = {
        "pos": outcome.pos,
        "world": world,
        "lives": max(lives, 0),
        "steps": steps,
        "visits": {**state.visits, outcome.pos: state.visits.get(outcome.pos, 0) + 1},
        "event": "dog" if bites else outcome.event,
        "status": status,
    }
    goto: Literal["think", "__end__"] = "think" if status == "running" else "__end__"
    return Command(update=update, goto=goto)


def _status(escaped: bool, lives: int, steps: int, max_steps: int) -> Status:
    if escaped:
        return "won"
    if lives <= 0:
        return "dead"
    return "out_of_steps" if steps >= max_steps else "running"


SERDE = JsonPlusSerializer(allowed_msgpack_modules=[("maze_agent.world", "World"), ("maze_agent.world", "Dog")])

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(think, timeout=LLM_TIMEOUT, retry_policy=RetryPolicy(max_attempts=3))
    .add_node(act)
    .add_edge(START, "think")
    .add_edge("think", "act")
    .compile(checkpointer=InMemorySaver(serde=SERDE))
)


def start_state(maze: Maze, world: World) -> State:
    return State(pos=maze.start, world=world, visits={maze.start: 1})
