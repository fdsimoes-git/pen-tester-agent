"""Ollama-backed LLM provider."""

import ollama

from .base import ModelProvider


class OllamaProvider(ModelProvider):  # pylint: disable=too-few-public-methods
    """Ollama-backed LLM provider."""

    def __init__(self, model: str = "qwen3.6:35b", num_ctx: int | None = None):
        self.model = model
        # Ollama defaults the runtime context window (num_ctx) to ~4K
        # regardless of the model's trained max, silently truncating longer
        # prompts. Passing an explicit num_ctx ensures the model actually
        # receives the context the agent budgeted for.
        self.num_ctx = num_ctx

    def chat(self, messages: list[dict]) -> str:
        options = {"num_ctx": self.num_ctx} if self.num_ctx else None
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return response["message"]["content"]
