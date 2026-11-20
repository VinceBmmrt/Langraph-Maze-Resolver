"""What the cat perceives, written as text for the LLM. Pure functions."""

from maze_agent.maze import Maze, Pos, nearest_unvisited, step
from maze_agent.world import Dog, World

HEARING = 3

EVENTS = {
    "bump": "you bumped into a wall",
    "locked": "the door is locked, you need the key",
    "spikes": "you stepped on spikes and lost a life",
    "dog": "a dog bit you and you lost a life",
    "key": "you picked up the key",
}


def observe(
    maze: Maze,
    world: World,
    pos: Pos,
    *,
    steps: int,
    lives: str,
    visits: dict[Pos, int],
    event: str = "",
    lessons: list[str] | None = None,
    advice: str = "",
) -> str:
    """The cat's view: its cell, its neighbours, the dogs it hears, its mental map, lessons and advice."""
    lines = [f"You are at {pos}. The exit is at {maze.goal}. Step {steps}. Lives: {lives}."]
    if world.door:
        lines.append("You carry the key." if world.has_key else "You do not have the key yet.")
    if event in EVENTS:
        lines.append(f"Last turn {EVENTS[event]}.")
    lines.append("Open directions (or wait):")
    for d in maze.exits(pos):
        cell = step(pos, d)
        lines.append(f"- {d} -> {cell}, visited {visits.get(cell, 0)} times{_hazard(world, maze, cell, pos)}")
    lines += [
        f"You hear a dog at {dog.pos}{_heading(dog, pos)}." for dog in world.dogs if distance(dog.pos, pos) <= HEARING
    ]
    if hint := mental_map(maze, world, pos, visits):
        cell, dist, first = hint
        lines.append(f"Mental map: nearest unexplored opening is {cell}, {dist} steps away (go {first} first).")
    if lessons:
        lines.append("Lessons from your past attempts on this maze:")
        lines += [f"- {lesson}" for lesson in lessons]
    if advice:
        lines.append(f'A human friend advises you: "{advice}"')
    return "\n".join(lines)


def mental_map(maze: Maze, world: World, pos: Pos, visits: dict[Pos, int]) -> tuple[Pos, int, str] | None:
    """Nearest unexplored opening the cat knows of, never through a locked door or onto spikes."""
    return nearest_unvisited(maze, pos, set(visits), world.locked | set(world.spikes))


def distance(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _hazard(world: World, maze: Maze, cell: Pos, cat: Pos) -> str:
    """What sits on a neighbouring cell, plus a warning when a dog could pounce there."""
    if cell == maze.goal:
        return " EXIT!"
    if world.dog_at(cell):
        return " DOG!"
    if cell in world.spikes:
        return " SPIKES!"
    near_dog = any(step(cell, d) != cat and world.dog_at(step(cell, d)) for d in maze.exits(cell))
    warning = ", a dog is right next to it!" if near_dog else ""
    if cell == world.key:
        return " KEY" + warning
    if cell == world.door:
        return (" open door" if world.has_key else " LOCKED DOOR") + warning
    return warning


def _heading(dog: Dog, cat: Pos) -> str:
    if dog.prev is None or distance(dog.pos, cat) == distance(dog.prev, cat):
        return ""
    return ", coming closer" if distance(dog.pos, cat) < distance(dog.prev, cat) else ", moving away"
