"""The level ladder: each level adds one new difficulty."""

import random
from dataclasses import dataclass

from maze_agent.maze import Maze, Pos, generate, shortest_path
from maze_agent.world import Dog, World


@dataclass(frozen=True)
class Level:
    number: int
    title: str
    size: int
    seed: int
    spikes: int = 0
    dogs: int = 0
    door: bool = False


LEVELS = (
    Level(1, "First steps", 5, 101),
    Level(2, "Getting lost", 8, 102),
    Level(3, "Spikes", 8, 103, spikes=6),
    Level(4, "Dogs", 10, 104, spikes=4, dogs=2),
    Level(5, "Locked door", 10, 105, spikes=4, dogs=1, door=True),
    Level(6, "The final maze", 12, 106, spikes=8, dogs=3, door=True),
)


def get_level(number: int) -> Level:
    return LEVELS[number - 1]


def build(level: Level) -> tuple[Maze, World]:
    """Generate the maze and place hazards, deterministically from the level seed.

    Hazards sit off the solution path, so a careful cat can always escape unhurt.
    The door sits on the solution path and its key is reachable without crossing it.
    """
    maze = generate(level.size, level.seed)
    rng = random.Random(level.seed)
    solution = shortest_path(maze, maze.start, maze.goal)
    free = [c for c in sorted(maze.passages) if c not in solution]
    rng.shuffle(free)
    door = key = None
    reserved: set[Pos] = set()
    if level.door:
        key, fork = _key_spot(maze, solution, free)
        door = solution[fork + 1]
        reserved |= set(shortest_path(maze, maze.start, key))
    spots = [c for c in free if c not in reserved]
    spikes = tuple(spots[: level.spikes])
    homes = [c for c in spots[level.spikes :] if len(shortest_path(maze, maze.start, c)) > level.size]
    dogs = tuple(Dog(pos=c, home=c) for c in homes[: level.dogs])
    return maze, World(spikes=spikes, dogs=dogs, key=key, door=door)


def _key_spot(maze: Maze, solution: list[Pos], free: list[Pos]) -> tuple[Pos, int]:
    """A key cell and the solution index its branch forks from, preferring forks late in the path."""
    forks = {c: _fork_index(maze, solution, c) for c in free}
    usable = [c for c in free if forks[c] < len(solution) - 2]
    late = [c for c in usable if forks[c] >= len(solution) // 2]
    key = (late or usable)[0]
    return key, forks[key]


def _fork_index(maze: Maze, solution: list[Pos], cell: Pos) -> int:
    """Index of the last solution cell on the way from the start to `cell`."""
    on_path = set(solution)
    return max(solution.index(c) for c in shortest_path(maze, maze.start, cell) if c in on_path)
