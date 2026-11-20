"""Seeded perfect mazes (recursive backtracker) and grid path search."""

import random
from collections import deque
from dataclasses import dataclass

Pos = tuple[int, int]
DIRECTIONS: dict[str, Pos] = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def step(pos: Pos, direction: str) -> Pos:
    """Cell reached by moving one cell in `direction`."""
    dx, dy = DIRECTIONS[direction]
    return (pos[0] + dx, pos[1] + dy)


@dataclass(frozen=True)
class Maze:
    """Grid maze. `passages[cell]` holds the directions with no wall."""

    size: int
    seed: int
    passages: dict[Pos, frozenset[str]]

    @property
    def start(self) -> Pos:
        return (0, 0)

    @property
    def goal(self) -> Pos:
        return (self.size - 1, self.size - 1)

    def exits(self, pos: Pos) -> list[str]:
        """Open directions from `pos`, in a stable order."""
        return [d for d in DIRECTIONS if d in self.passages[pos]]

    def to_dict(self) -> dict:
        """JSON form for the web UI: one list of open directions per cell, row by row."""
        cells = [[sorted(self.passages[(x, y)]) for x in range(self.size)] for y in range(self.size)]
        return {"size": self.size, "start": self.start, "goal": self.goal, "cells": cells}


def generate(size: int, seed: int) -> Maze:
    """Carve a perfect maze: exactly one path between any two cells."""
    rng = random.Random(seed)
    passages: dict[Pos, set[str]] = {(x, y): set() for x in range(size) for y in range(size)}
    stack, seen = [(0, 0)], {(0, 0)}
    while stack:
        cell = stack[-1]
        options = [d for d in DIRECTIONS if step(cell, d) in passages and step(cell, d) not in seen]
        if not options:
            stack.pop()
            continue
        d = rng.choice(options)
        nxt = step(cell, d)
        passages[cell].add(d)
        passages[nxt].add(OPPOSITE[d])
        seen.add(nxt)
        stack.append(nxt)
    return Maze(size, seed, {cell: frozenset(ds) for cell, ds in passages.items()})


def shortest_path(maze: Maze, start: Pos, goal: Pos, blocked: frozenset[Pos] = frozenset()) -> list[Pos]:
    """Cells from `start` to `goal` (both included), avoiding `blocked`. Empty if unreachable."""
    parents: dict[Pos, Pos | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parents[cell]
            return path[::-1]
        for d in maze.exits(cell):
            nxt = step(cell, d)
            if nxt not in parents and nxt not in blocked:
                parents[nxt] = cell
                queue.append(nxt)
    return []


def nearest_unvisited(
    maze: Maze, pos: Pos, visited: set[Pos], avoid: frozenset[Pos] = frozenset()
) -> tuple[Pos, int, str] | None:
    """Closest unvisited cell reachable through visited ones: (cell, distance, first direction).

    Unvisited cells in `avoid` are never targets. Visited ones stay walkable: the cat already crossed them.
    """
    first: dict[Pos, str] = {}
    seen, queue = {pos}, deque([(pos, 0)])
    while queue:
        cell, dist = queue.popleft()
        for d in maze.exits(cell):
            nxt = step(cell, d)
            if nxt in seen or (nxt in avoid and nxt not in visited):
                continue
            seen.add(nxt)
            first[nxt] = first.get(cell, d)
            if nxt not in visited:
                return nxt, dist + 1, first[nxt]
            queue.append((nxt, dist + 1))
    return None
