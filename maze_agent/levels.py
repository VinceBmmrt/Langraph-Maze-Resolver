"""The level ladder: each level adds one new difficulty."""

import random
from dataclasses import dataclass

from maze_agent.maze import Maze, Pos, add_loops, generate, shortest_path, step
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
    loops: int = 0

    @property
    def max_steps(self) -> int:
        """Step budget: enough to explore the whole maze about twice."""
        return 2 * self.size * self.size


LEVELS = (
    Level(1, "First steps", 7, 101),
    Level(2, "Getting lost", 10, 102),
    Level(3, "Spikes", 12, 103, spikes=8, loops=3),
    Level(4, "Dogs", 14, 104, spikes=8, dogs=3, loops=8),
    Level(5, "Locked door", 16, 105, spikes=10, dogs=3, door=True, loops=10),
    Level(6, "The final maze", 18, 106, spikes=14, dogs=4, door=True, loops=14),
)


def get_level(number: int) -> Level:
    return LEVELS[number - 1]


def build(level: Level) -> tuple[Maze, World]:
    """Generate the maze and place hazards, deterministically from the level seed.

    Hazards sit off the solution path, so a careful cat can always escape unhurt, but close to it,
    so they matter: spikes guard side-branch entrances and dogs live just behind them, free to
    roam onto the path. The door sits on the solution path and its key is reachable without crossing it.
    """
    maze = generate(level.size, level.seed)
    rng = random.Random(level.seed)
    door = key = region = None
    if level.door:
        solution = shortest_path(maze, maze.start, maze.goal)
        key, fork = _key_spot(maze, solution, _shuffled(_off_path(maze, solution), rng))
        door = solution[fork + 1]
        region = _reachable(maze, maze.start, door)
    maze = add_loops(maze, level.loops, rng, region)
    solution = shortest_path(maze, maze.start, maze.goal)
    depth = _depths(maze, solution)
    free = _shuffled(_off_path(maze, solution), rng)
    reserved = set(shortest_path(maze, maze.start, key, frozenset({door}))) if key and door else set()
    spikes = _spike_spots(maze, sorted((c for c in free if c not in reserved), key=lambda c: depth[c]), level.spikes)
    homes = [
        c
        for c in free
        if c not in spikes
        and len(shortest_path(maze, maze.start, c)) > level.size  # far enough that the start stays calm
        and shortest_path(maze, c, maze.goal, frozenset(spikes))  # free to roam onto the cat's path
    ]
    homes.sort(key=lambda c: abs(depth[c] - 2))
    dogs = tuple(Dog(pos=c, home=c) for c in homes[: level.dogs])
    return maze, World(spikes=spikes, dogs=dogs, key=key, door=door)


def _off_path(maze: Maze, solution: list[Pos]) -> list[Pos]:
    on_path = set(solution)
    return [c for c in sorted(maze.passages) if c not in on_path]


def _shuffled(cells: list[Pos], rng: random.Random) -> list[Pos]:
    rng.shuffle(cells)
    return cells


def _reachable(maze: Maze, start: Pos, wall: Pos) -> frozenset[Pos]:
    """Cells reachable from `start` without entering `wall`."""
    seen, queue = {start}, [start]
    for cell in queue:
        for d in maze.exits(cell):
            nxt = step(cell, d)
            if nxt != wall and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return frozenset(seen)


def _spike_spots(maze: Maze, candidates: list[Pos], count: int) -> tuple[Pos, ...]:
    """The first `count` candidates that are not hidden behind an already placed spike."""
    spikes: list[Pos] = []
    for cell in candidates:
        if len(spikes) == count:
            break
        if not set(spikes) & set(shortest_path(maze, cell, maze.goal)):
            spikes.append(cell)
    return tuple(spikes)


def _depths(maze: Maze, solution: list[Pos]) -> dict[Pos, int]:
    """Distance from every cell to the solution path (0 on the path)."""
    depth = dict.fromkeys(solution, 0)
    queue = list(solution)
    for cell in queue:
        for d in maze.exits(cell):
            nxt = step(cell, d)
            if nxt not in depth:
                depth[nxt] = depth[cell] + 1
                queue.append(nxt)
    return depth


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
