"""Terminal run: `uv run python -m maze_agent --level 3`."""

import argparse
import asyncio
import uuid

from maze_agent.brain import LLMBrain
from maze_agent.config import MODEL
from maze_agent.levels import get_level
from maze_agent.runner import play


async def main(level_number: int, model: str) -> None:
    level = get_level(level_number)
    print(f"Level {level.number}: {level.title} ({level.size}x{level.size}) with {model}")
    tokens, event = 0, {}
    async for event in play(level, LLMBrain(model), str(uuid.uuid4())):
        if event["type"] == "think":
            tokens += event["tokens"]
            print(f"      {event['direction']:<5} {event['thought']}")
        else:
            print(f"  {event['steps']:>3} {tuple(event['pos'])} {event['event']:<7} lives={event['lives']}")
    print(f"=> {event.get('status')} in {event.get('steps')} steps, {tokens} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    asyncio.run(main(args.level, args.model))
