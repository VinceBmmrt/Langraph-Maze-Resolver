"""The cat's brain: an LLM that picks the next move from a text observation, and learns from failures."""

from dataclasses import dataclass
from typing import Literal, Protocol, cast

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are a cat escaping a maze on a grid. x grows east, y grows south.
You only see your own cell and its direct neighbours. Each turn you pick one open direction.
Dangers: SPIKES and DOGS each cost a life. Dogs patrol the corridors at half your speed.
You may "wait" one turn to let a dog that is moving away get clear. Never wait for a dog
that is coming closer: back away from it instead. Spikes ALWAYS cost a life, while a dog only
bites if it reaches you, so never step on spikes to escape a dog.
A LOCKED DOOR blocks the way until you pick up the KEY somewhere in the maze.
Your mental map tells you the nearest unexplored opening: follow it to avoid walking in circles.
Prefer never-visited cells heading toward the exit. Step around dangers when you can.
Never pick a closed direction. Lessons from your past attempts, if any, are worth following."""

REFLECT_PROMPT = """You are a cat who just failed to escape a maze. From the summary of your attempt,
write ONE short, concrete lesson for your next attempt on the same maze (places to avoid,
where the key is, which corridor leads on). Max 30 words."""


class Decision(BaseModel):
    """The cat's next move ("wait" stays put for one turn). The thought comes first: think, then act."""

    thought: str = Field(description="One short, playful sentence explaining the choice.")
    direction: Literal["north", "south", "east", "west", "wait"]


class Lesson(BaseModel):
    lesson: str = Field(description="One concrete lesson, max 30 words.")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class Brain(Protocol):
    """Anything that decides moves and writes lessons, reporting the tokens it spent."""

    model: str

    async def decide(self, observation: str) -> tuple[Decision, Usage]: ...

    async def reflect(self, summary: str) -> tuple[str, Usage]: ...


class LLMBrain:
    """A chat model constrained to structured answers (native JSON schema on every provider)."""

    def __init__(self, model: str):
        self.model = model
        llm = init_chat_model(model)
        self.decider = llm.with_structured_output(Decision, method="json_schema", include_raw=True)
        self.reflector = llm.with_structured_output(Lesson, method="json_schema", include_raw=True)

    async def decide(self, observation: str) -> tuple[Decision, Usage]:
        result = cast(dict, await self.decider.ainvoke([("system", SYSTEM_PROMPT), ("human", observation)]))
        return result["parsed"], _usage(result)

    async def reflect(self, summary: str) -> tuple[str, Usage]:
        result = cast(dict, await self.reflector.ainvoke([("system", REFLECT_PROMPT), ("human", summary)]))
        return result["parsed"].lesson, _usage(result)


def _usage(result: dict) -> Usage:
    meta = result["raw"].usage_metadata or {}
    return Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
