"""The cat's brain: an LLM that picks the next move from a text observation."""

from typing import Literal, Protocol, cast

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are a cat escaping a maze on a grid. x grows east, y grows south.
You only see your own cell and its direct neighbours. Each turn you pick one open direction.
Dangers: SPIKES and DOGS each cost a life. Dogs roam and chase you when close.
A LOCKED DOOR blocks the way until you pick up the KEY somewhere in the maze.
Your mental map tells you the nearest unexplored opening: follow it to avoid walking in circles.
Prefer never-visited cells heading toward the exit. Step around dangers when you can.
Never pick a closed direction."""


class Decision(BaseModel):
    """The cat's next move."""

    direction: Literal["north", "south", "east", "west"]
    thought: str = Field(description="One short, playful sentence explaining the choice.")


class Brain(Protocol):
    """Anything that turns an observation into a decision and the tokens it spent."""

    async def decide(self, observation: str) -> tuple[Decision, int]: ...


class LLMBrain:
    """A chat model constrained to answer with a `Decision`."""

    def __init__(self, model: str):
        self.llm = init_chat_model(model).with_structured_output(Decision, include_raw=True)

    async def decide(self, observation: str) -> tuple[Decision, int]:
        result = cast(dict, await self.llm.ainvoke([("system", SYSTEM_PROMPT), ("human", observation)]))
        usage = result["raw"].usage_metadata or {}
        return result["parsed"], usage.get("total_tokens", 0)
