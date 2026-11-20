"""World rules: spikes, dogs, key and door. Pure functions over plain data."""

import random
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from maze_agent.maze import Maze, Pos, step


class Dog(BaseModel):
    model_config = ConfigDict(frozen=True)

    pos: Pos
    home: Pos
    prev: Pos | None = None


class World(BaseModel):
    """Hazards of a level. `key is None` means the cat carries it (or there is none)."""

    model_config = ConfigDict(frozen=True)

    spikes: tuple[Pos, ...] = ()
    dogs: tuple[Dog, ...] = ()
    key: Pos | None = None
    door: Pos | None = None
    has_key: bool = False

    @property
    def locked(self) -> frozenset[Pos]:
        """Cells nobody can enter: the door until the cat picks up the key."""
        return frozenset() if self.has_key or self.door is None else frozenset({self.door})

    def dog_at(self, pos: Pos) -> Dog | None:
        return next((d for d in self.dogs if d.pos == pos), None)


@dataclass(frozen=True)
class Outcome:
    """Result of one cat move: where it ends, lives lost and what happened."""

    pos: Pos
    world: World
    hurt: int
    event: str  # moved | waited | bump | locked | spikes | dog | key


def move_cat(maze: Maze, world: World, pos: Pos, direction: str) -> Outcome:
    """Walk one cell, or stay put on "wait". Walls and a locked door keep the cat in place."""
    if direction == "wait":
        return Outcome(pos, world, 0, "waited")
    if direction not in maze.exits(pos):
        return Outcome(pos, world, 0, "bump")
    target = step(pos, direction)
    if target in world.locked:
        return Outcome(pos, world, 0, "locked")
    if target == world.key:
        return Outcome(target, world.model_copy(update={"key": None, "has_key": True}), 0, "key")
    if target in world.spikes:
        return Outcome(target, world, 1, "spikes")
    if dog := world.dog_at(target):
        return Outcome(target, send_home(world, dog), 1, "dog")
    return Outcome(target, world, 0, "moved")


def move_dogs(maze: Maze, world: World, cat: Pos, turn: int) -> tuple[World, int]:
    """On even turns, each dog patrols one cell. Returns the new world and bites.

    Dogs never chase and move at half the cat's speed: they are moving obstacles a careful cat can avoid.
    """
    if turn % 2:
        return world, 0
    rng = random.Random(maze.seed * 1000 + turn)
    bites, dogs = 0, []
    for dog in world.dogs:
        nxt = _patrol_step(maze, world, dog, rng)
        if nxt == cat:
            bites += 1
            nxt = dog.home
        dogs.append(dog.model_copy(update={"pos": nxt, "prev": None if nxt == dog.home else dog.pos}))
    return world.model_copy(update={"dogs": tuple(dogs)}), bites


def send_home(world: World, dog: Dog) -> World:
    dogs = tuple(d.model_copy(update={"pos": d.home}) if d is dog else d for d in world.dogs)
    return world.model_copy(update={"dogs": dogs})


def _patrol_step(maze: Maze, world: World, dog: Dog, rng: random.Random) -> Pos:
    """Keep walking forward, turning back only at a dead end. Dogs avoid spikes, the door and the start."""
    blocked = world.locked | set(world.spikes) | {maze.start}
    options = [step(dog.pos, d) for d in maze.exits(dog.pos) if step(dog.pos, d) not in blocked]
    forward = [c for c in options if c != dog.prev]
    return rng.choice(forward or options or [dog.pos])
