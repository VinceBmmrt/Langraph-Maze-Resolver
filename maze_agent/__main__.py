"""Terminal run: `uv run python -m maze_agent --level 3 --model openai:gpt-5.4-nano`."""

import argparse
import asyncio
import uuid

from maze_agent.brain import LLMBrain
from maze_agent.config import MODEL
from maze_agent.levels import get_level
from maze_agent.runner import context, play


async def main(level_number: int, model: str, attempts: int) -> None:
    level = get_level(level_number)
    memory = f"cli-{uuid.uuid4()}"  # lessons are shared between the attempts of this command only
    for attempt in range(1, attempts + 1):
        print(f"Level {level.number}: {level.title} ({level.size}x{level.size}) with {model}, attempt {attempt}")
        ctx = context(level, LLMBrain(model), memory_key=memory)
        cost, last = 0.0, {}
        async for event in play(level, ctx, str(uuid.uuid4())):
            cost += event.get("cost", 0.0)
            if event["type"] == "think":
                print(f"      {event['direction']:<5} {event['thought']}")
            elif event["type"] == "act":
                last = event
                print(f"  {event['steps']:>3} {tuple(event['pos'])} {event['event']:<7} lives={event['lives']}")
            elif event["type"] == "reflect":
                print(f"  lesson: {event['lesson']}")
        print(f"=> {last.get('status')} in {last.get('steps')} steps, ${cost:.4f}")
        if last.get("status") == "won":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--attempts", type=int, default=1, help="retry after a failure, learning from it")
    args = parser.parse_args()
    asyncio.run(main(args.level, args.model, args.attempts))
