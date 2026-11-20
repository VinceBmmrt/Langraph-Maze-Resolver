import asyncio
import re

from langchain_core.runnables import RunnableConfig

from maze_agent.agent import Context, graph, observe, start_state
from maze_agent.brain import Decision
from maze_agent.levels import build, get_level
from maze_agent.maze import DIRECTIONS, generate, shortest_path, step
from maze_agent.world import Dog, World


class RouteBrain:
    """Walks a fixed list of directions, then keeps bumping north."""

    def __init__(self, directions: list[str]):
        self.directions = iter(directions)

    async def decide(self, observation: str) -> tuple[Decision, int]:
        return Decision.model_validate({"direction": next(self.directions, "north"), "thought": "scripted"}), 1


class HintBrain:
    """Follows the mental map hint, else the exit when it is next door."""

    async def decide(self, observation: str) -> tuple[Decision, int]:
        match = re.search(r"- (\w+) -> .* EXIT!", observation) or re.search(r"go (\w+) first", observation)
        assert match
        return Decision.model_validate({"direction": match.group(1), "thought": "hint"}), 1


def route(cells):
    moves = {v: k for k, v in DIRECTIONS.items()}
    return [moves[(b[0] - a[0], b[1] - a[1])] for a, b in zip(cells, cells[1:], strict=False)]


def play(level_number, brain, max_steps=300):
    maze, world = build(get_level(level_number))
    config: RunnableConfig = {"configurable": {"thread_id": f"t{level_number}-{id(brain)}"}, "recursion_limit": 1000}
    ctx = Context(maze=maze, brain=brain, max_steps=max_steps)
    return asyncio.run(graph.ainvoke(start_state(maze, world), config, context=ctx)), maze


def test_shortest_route_wins_level_1():
    maze, _ = build(get_level(1))
    path = shortest_path(maze, maze.start, maze.goal)
    final, _ = play(1, RouteBrain(route(path)))
    assert (final["status"], final["steps"], final["lives"]) == ("won", len(path) - 1, 3)
    assert final["tokens"] == len(path) - 1  # reducer sums tokens of every step


def test_wall_bumping_runs_out_of_steps():
    final, maze = play(1, RouteBrain([]), max_steps=5)
    assert final["status"] == "out_of_steps"
    assert final["pos"] == maze.start


def test_mental_map_explores_to_the_exit():
    final, _ = play(2, HintBrain())
    assert final["status"] == "won"


def test_observation_shows_neighbours():
    maze, world = build(get_level(3))
    spike = world.spikes[0]
    near = next(c for c in shortest_path(maze, maze.start, spike) if c != spike and spike in _neighbours(maze, c))
    text = observe(start_state(maze, world).model_copy(update={"pos": near}), maze)
    assert "SPIKES!" in text and "Mental map" in text


def _neighbours(maze, cell):
    return [(cell[0] + DIRECTIONS[d][0], cell[1] + DIRECTIONS[d][1]) for d in maze.exits(cell)]


def test_mental_map_never_points_at_spikes():
    maze, world = build(get_level(3))
    spike = world.spikes[0]
    near = next(c for c in shortest_path(maze, maze.start, spike) if spike in _neighbours(maze, c))
    visited = set(shortest_path(maze, maze.start, near))
    state = start_state(maze, world).model_copy(update={"pos": near, "visits": dict.fromkeys(visited, 1)})
    assert f"opening is {spike}" not in observe(state, maze)


def test_runner_streams_json_events():
    import json

    from maze_agent.runner import play as run_level

    async def collect():
        return [e async for e in run_level(get_level(1), HintBrain(), "runner")]

    events = asyncio.run(collect())
    assert [e["type"] for e in events[:2]] == ["think", "act"]
    assert events[-1]["status"] == "won"
    json.dumps(events)


def test_mental_map_leads_back_across_crossed_spikes():
    maze = generate(8, 1)  # perfect maze: the pocket behind the spike has no other exit
    solution = shortest_path(maze, maze.start, maze.goal)
    beyond = {c: shortest_path(maze, maze.start, c) for c in maze.passages if c not in solution}
    spike = next(c for c, path in beyond.items() if any(c in p[:-1] for p in beyond.values()))
    pocket = {c for c, path in beyond.items() if spike in path[:-1]}
    world = World(spikes=(spike,))
    visited = set(beyond[spike]) | pocket
    state = start_state(maze, world).model_copy(update={"pos": next(iter(pocket)), "visits": dict.fromkeys(visited, 1)})
    assert "Mental map" in observe(state, maze)


def test_observation_warns_of_a_dog_next_to_a_cell():
    maze = generate(6, 3)
    a = maze.start
    b = step(a, maze.exits(a)[0])
    c = step(b, next(d for d in maze.exits(b) if step(b, d) != a))
    world = World(dogs=(Dog(pos=c, home=c),))
    state = start_state(maze, world)
    assert "a dog is right next to it!" in observe(state, maze)


def test_observation_tells_dog_heading():
    maze = generate(6, 3)
    world = World(dogs=(Dog(pos=(1, 1), home=(2, 2), prev=(2, 1)),))
    state = start_state(maze, world)
    assert "You hear a dog at (1, 1), coming closer." in observe(state, maze)
