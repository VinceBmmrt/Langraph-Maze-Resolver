"""Evaluation harness: models x levels, each with up to N attempts that learn from earlier failures.

    uv run python -m maze_agent.eval --models openai:gpt-5.4-mini openai:gpt-5.4-nano --repeats 2

Writes `evals/results.json` (shown in the web UI). Runs are tagged "eval" for LangSmith when tracing is on.
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from maze_agent.brain import LLMBrain
from maze_agent.config import MODEL
from maze_agent.levels import LEVELS, Level, get_level, optimal_steps
from maze_agent.runner import context, play

RESULTS = Path(__file__).parent.parent / "evals" / "results.json"


async def attempt(level: Level, model: str, memory: str, number: int) -> dict:
    """One run; lessons written on failure are read by the next attempt of the same chain."""
    ctx = context(level, LLMBrain(model), memory_key=memory)
    cost, tokens, last = 0.0, 0, {}
    async for event in play(level, ctx, str(uuid.uuid4()), tags=["eval", model, f"level-{level.number}"]):
        cost += event.get("cost", 0.0)
        tokens += event.get("tokens", 0)
        if event["type"] == "act":
            last = event
    return {
        "model": model,
        "level": level.number,
        "attempt": number,
        "status": last["status"],
        "steps": last["steps"],
        "lives_lost": level.lives - last["lives"],
        "tokens": tokens,
        "cost": cost,
    }


async def chain(level: Level, model: str, attempts: int, gate: asyncio.Semaphore) -> list[dict]:
    """Up to `attempts` tries on one level, stopping at the first win."""
    memory, runs = f"eval-{uuid.uuid4()}", []
    async with gate:
        for number in range(1, attempts + 1):
            runs.append(await attempt(level, model, memory, number))
            print(f"  {model} level {level.number} attempt {number}: {runs[-1]['status']} in {runs[-1]['steps']} steps")
            if runs[-1]["status"] == "won":
                break
    return runs


def summarize(runs: list[dict], chains: list[list[dict]]) -> list[dict]:
    """One row per model and level."""
    rows = []
    for model in dict.fromkeys(r["model"] for r in runs):
        for number in sorted({r["level"] for r in runs if r["model"] == model}):
            mine = [c for c in chains if c[0]["model"] == model and c[0]["level"] == number]
            done = [r for r in runs if r["model"] == model and r["level"] == number]
            wins = [r for r in done if r["status"] == "won"]
            rows.append(
                {
                    "model": model,
                    "level": number,
                    "chains": len(mine),
                    "first_try": sum(c[0]["status"] == "won" for c in mine) / len(mine),
                    "with_learning": sum(c[-1]["status"] == "won" for c in mine) / len(mine),
                    "attempts": len(done),
                    "steps": statistics.mean(r["steps"] for r in wins) if wins else None,
                    "optimal": optimal_steps(get_level(number)),
                    "cost_per_attempt": statistics.mean(r["cost"] for r in done),
                    "cost_per_win": sum(r["cost"] for r in done) / len(wins) if wins else None,
                    "lives_lost": statistics.mean(r["lives_lost"] for r in done),
                }
            )
    return rows


async def main(models: list[str], levels: list[int], repeats: int, attempts: int, concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)
    jobs = [chain(get_level(n), m, attempts, gate) for m in models for n in levels for _ in range(repeats)]
    started = time.perf_counter()
    chains = await asyncio.gather(*jobs)
    runs = [r for c in chains for r in c]
    report = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {"models": models, "levels": levels, "repeats": repeats, "max_attempts": attempts},
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "rows": summarize(runs, chains),
        "runs": runs,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"Wrote {RESULTS} ({len(runs)} runs, ${sum(r['cost'] for r in runs):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[MODEL])
    parser.add_argument("--levels", nargs="+", type=int, default=[lv.number for lv in LEVELS])
    parser.add_argument("--repeats", type=int, default=2, help="independent chains per model and level")
    parser.add_argument("--attempts", type=int, default=3, help="max attempts per chain (learning between them)")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.models, args.levels, args.repeats, args.attempts, args.concurrency))
