"""The models the cat can run on, with their prices (USD per million tokens)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str  # an init_chat_model id
    label: str
    key_env: str  # the API key it needs
    input_price: float
    output_price: float

    @property
    def available(self) -> bool:
        return bool(os.getenv(self.key_env))

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000


MODELS = {
    m.id: m
    for m in (
        ModelInfo("openai:gpt-5.4-mini", "GPT-5.4 mini", "OPENAI_API_KEY", 0.75, 4.50),
        ModelInfo("openai:gpt-5.4-nano", "GPT-5.4 nano", "OPENAI_API_KEY", 0.20, 1.25),
        ModelInfo("anthropic:claude-haiku-5-5", "Claude Haiku 5.5", "ANTHROPIC_API_KEY", 0.10, 0.50),
    )
}


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of a call; 0 for a model missing from the catalog."""
    info = MODELS.get(model)
    return info.cost(input_tokens, output_tokens) if info else 0.0
