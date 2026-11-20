import asyncio
import json
import re
import uuid

from maze_agent.agent import State, start_state
from maze_agent.brain import Decision, Usage
from maze_agent.levels import build, get_level
from maze_agent.maze import DIRECTIONS, generate, shortest_path, step
from maze_agent.perception import observe
from maze_agent.runner import context, play
from maze_agent.world import Dog, World


class FakeBrain:
    """Scripted brain: no API key needed. Records every observation it receives."""

    model = "openai:gpt-5.4-mini"

    def __init__(self, directions: list[str] | None = None, usage: Usage | None = None):
        self.directions = iter(directions or [])
        self.usage = usage or Usage()
        self.seen: list[str] = []

    def choose(self, observation: str) -> str:
        return next(self.directions, "north")

    async def decide(self, observation: str) -> tuple[Decision, Usage]:
        self.seen.append(observation)
        return Decision.model_validate({"thought": "scripted", "direction": self.choose(observation)}), self.usage

    async def reflect(self, summary: str) -> tuple[str, Usage]:
        return "Stop bumping into the north wall.", self.usage


class HintBrain(FakeBrain):
    """Follows the mental map hint, or the exit when it is next door."""

    def choose(self, observation: str) -> str:
        match = re.search(r"- (\w+) -> .* EXIT!", observation) or re.search(r"go (\w+) first", observation)
        assert match
        return match.group(1)


def run(level_number, brain, *, resume=None, thread=None, **ctx_options):
    """All events of one run (or of its resumption)."""
    level = get_level(level_number)
    ctx = context(level, brain, memory_key=ctx_options.pop("memory_key", str(uuid.uuid4())), **ctx_options)

    async def collect():
        return [e async for e in play(level, ctx, thread or str(uuid.uuid4()), resume=resume)]

    return asyncio.run(collect())


def final(events):
    return next(e for e in reversed(events) if e["type"] == "act")


def route(cells):
    moves = {v: k for k, v in DIRECTIONS.items()}
    return [moves[(b[0] - a[0], b[1] - a[1])] for a, b in zip(cells, cells[1:], strict=False)]


def view(state: State, maze) -> str:
    return observe(maze, state.world, state.pos, steps=state.steps, lives="3/3", visits=state.visits)


def test_shortest_route_wins_level_1_and_costs_add_up():
    maze, _ = build(get_level(1))
    path = shortest_path(maze, maze.start, maze.goal)
    events = run(1, FakeBrain(route(path), Usage(1000, 100)))
    last = final(events)
    assert (last["status"], last["steps"], last["lives"]) == ("won", len(path) - 1, 3)
    thinks = [e for e in events if e["type"] == "think"]
    assert sum(e["tokens"] for e in thinks) == 1100 * (len(path) - 1)
    assert abs(thinks[0]["cost"] - (1000 * 0.75 + 100 * 4.50) / 1e6) < 1e-12


def test_mental_map_explores_to_the_exit():
    assert final(run(2, HintBrain()))["status"] == "won"


def test_failure_writes_a_lesson_that_the_next_attempt_reads():
    get_level(1)
    memory = str(uuid.uuid4())
    first = run(1, FakeBrain(), memory_key=memory)
    assert final(first)["status"] == "out_of_steps"
    assert next(e for e in first if e["type"] == "reflect")["lesson"] == "Stop bumping into the north wall."
    brain = FakeBrain()
    run(1, brain, memory_key=memory)
    assert "Stop bumping into the north wall." in brain.seen[0]


def test_stuck_cat_asks_the_human_and_resumes_with_the_advice():
    thread, memory = str(uuid.uuid4()), str(uuid.uuid4())
    brain = FakeBrain()  # bumps north forever: circling on one cell
    events = run(1, brain, thread=thread, memory_key=memory, ask_human=True)
    assert events[-1]["type"] == "interrupt" and "stuck" in events[-1]["question"]
    resumed = run(1, brain, resume="go south!", thread=thread, memory_key=memory, ask_human=True)
    assert resumed[0]["type"] == "ask_human" and resumed[0]["advice"] == "go south!"
    assert 'A human friend advises you: "go south!"' in brain.seen[len(brain.seen) - 1 - _thinks_after(resumed)]


def _thinks_after(events):
    """How many decisions came after the first one that followed the advice."""
    return sum(1 for e in events if e["type"] == "think") - 1


def test_runner_events_are_json():
    json.dumps(run(1, HintBrain()))


def test_observation_shows_neighbours():
    maze, world = build(get_level(3))
    spike = world.spikes[0]
    near = next(c for c in shortest_path(maze, maze.start, spike) if c != spike and spike in _neighbours(maze, c))
    text = view(start_state(maze, world).model_copy(update={"pos": near}), maze)
    assert "SPIKES!" in text and "Mental map" in text


def _neighbours(maze, cell):
    return [(cell[0] + DIRECTIONS[d][0], cell[1] + DIRECTIONS[d][1]) for d in maze.exits(cell)]


def test_mental_map_never_points_at_spikes():
    maze, world = build(get_level(3))
    spike = world.spikes[0]
    near = next(c for c in shortest_path(maze, maze.start, spike) if spike in _neighbours(maze, c))
    visited = set(shortest_path(maze, maze.start, near))
    state = start_state(maze, world).model_copy(update={"pos": near, "visits": dict.fromkeys(visited, 1)})
    assert f"opening is {spike}" not in view(state, maze)


def test_mental_map_leads_back_across_crossed_spikes():
    maze = generate(8, 1)  # perfect maze: the pocket behind the spike has no other exit
    solution = shortest_path(maze, maze.start, maze.goal)
    beyond = {c: shortest_path(maze, maze.start, c) for c in maze.passages if c not in solution}
    spike = next(c for c, path in beyond.items() if any(c in p[:-1] for p in beyond.values()))
    pocket = {c for c, path in beyond.items() if spike in path[:-1]}
    world = World(spikes=(spike,))
    visited = set(beyond[spike]) | pocket
    state = start_state(maze, world).model_copy(update={"pos": next(iter(pocket)), "visits": dict.fromkeys(visited, 1)})
    assert "Mental map" in view(state, maze)


def test_observation_warns_of_a_dog_next_to_a_cell():
    maze = generate(6, 3)
    a = maze.start
    b = step(a, maze.exits(a)[0])
    c = step(b, next(d for d in maze.exits(b) if step(b, d) != a))
    world = World(dogs=(Dog(pos=c, home=c),))
    assert "a dog is right next to it!" in view(start_state(maze, world), maze)


def test_observation_tells_dog_heading():
    maze = generate(6, 3)
    world = World(dogs=(Dog(pos=(1, 1), home=(2, 2), prev=(2, 1)),))
    assert "You hear a dog at (1, 1), coming closer." in view(start_state(maze, world), maze)


def test_eval_summary_counts_first_try_and_learning():
    from maze_agent.eval import summarize

    lost = {"model": "m", "level": 1, "attempt": 1, "status": "dead", "steps": 9, "lives_lost": 3, "cost": 0.01}
    won = {**lost, "status": "won", "steps": 14, "lives_lost": 0, "cost": 0.02}
    (row,) = summarize([won, lost, {**won, "attempt": 2}])  # two chains: a win, then a loss and a win
    assert (row["first_try"], row["with_learning"], row["attempts"], row["steps"]) == (0.5, 1.0, 3, 14)
    assert abs(row["cost_per_win"] - 0.025) < 1e-12 and row["optimal"] == 12
