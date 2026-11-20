import pytest

from maze_agent.levels import LEVELS, build
from maze_agent.maze import add_loops, generate, shortest_path, step
from maze_agent.world import Dog, World, move_cat, move_dogs


def test_loops_add_exactly_that_many_openings():
    import random

    maze = generate(8, 7)
    looped = add_loops(maze, 5, random.Random(1))
    edges = sum(len(ds) for ds in looped.passages.values()) // 2
    assert edges == 8 * 8 - 1 + 5


def test_maze_is_perfect_and_seeded():
    maze = generate(8, 7)
    edges = sum(len(ds) for ds in maze.passages.values()) // 2
    assert edges == 8 * 8 - 1  # a tree: one path between any two cells
    assert maze == generate(8, 7)


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.title)
def test_level_is_solvable_unhurt(level):
    maze, world = build(level)
    solution = shortest_path(maze, maze.start, maze.goal)
    assert not set(world.spikes) & set(solution)
    assert len(world.spikes) == level.spikes
    assert len(world.dogs) == level.dogs
    for dog in world.dogs:  # every dog can roam onto the cat's path, but lives away from the start
        assert set(shortest_path(maze, dog.pos, maze.goal, frozenset(world.spikes))) & set(solution)
        assert len(shortest_path(maze, maze.start, dog.home)) > level.size
    if level.door:
        assert not shortest_path(maze, maze.start, maze.goal, world.locked)  # the door cannot be bypassed
        assert len(shortest_path(maze, world.key, world.door)) > level.size  # the key is a real detour
        assert world.key and shortest_path(maze, maze.start, world.key, world.locked)


def test_wall_bump_and_locked_door():
    maze = generate(5, 1)
    wall = next(d for d in ("north", "west") if d not in maze.exits((0, 0)))
    assert move_cat(maze, World(), (0, 0), wall).event == "bump"
    d = maze.exits((0, 0))[0]
    locked = World(door=step((0, 0), d), key=(4, 4))
    assert move_cat(maze, locked, (0, 0), d).event == "locked"
    assert move_cat(maze, locked.model_copy(update={"has_key": True}), (0, 0), d).event == "moved"


def test_spikes_and_dogs_hurt():
    maze = generate(5, 1)
    d = maze.exits((0, 0))[0]
    target = step((0, 0), d)
    assert move_cat(maze, World(spikes=(target,)), (0, 0), d).hurt == 1
    bitten = move_cat(maze, World(dogs=(Dog(pos=target, home=(4, 4)),)), (0, 0), d)
    assert (bitten.hurt, bitten.world.dogs[0].pos) == (1, (4, 4))


def test_wait_keeps_the_cat_in_place_unhurt():
    maze = generate(5, 1)
    outcome = move_cat(maze, World(), (0, 0), "wait")
    assert (outcome.pos, outcome.hurt, outcome.event) == ((0, 0), 0, "waited")


def test_dogs_patrol_at_half_speed():
    maze = generate(6, 2)
    cell = next(c for c in maze.passages if len(maze.exits(c)) == 2 and c != maze.start)
    world = World(dogs=(Dog(pos=cell, home=cell),))
    assert move_dogs(maze, world, (5, 5), turn=1)[0] == world  # odd turn: dogs rest
    assert move_dogs(maze, world, (5, 5), turn=2)[0].dogs[0].pos != cell
