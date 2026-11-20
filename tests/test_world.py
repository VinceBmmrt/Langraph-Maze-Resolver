import pytest

from maze_agent.levels import LEVELS, build
from maze_agent.maze import generate, shortest_path, step
from maze_agent.world import Dog, World, move_cat


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
    if level.door:
        assert world.door in solution
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
