"""Ollama-backed LLM provider."""

import ollama

from .base import ModelProvider

# Ollama defaults the runtime context window (num_ctx) to ~4K regardless of
# the model's trained max, silently truncating longer prompts. Default to a
# window large enough for the agent's default context budget plus generation
# headroom, so even a direct ``OllamaProvider(model=...)`` caller (not just the
# CLI) avoids that truncation. Pass ``num_ctx=None`` to defer to Ollama's own
# default instead.
_DEFAULT_NUM_CTX = 40960


class OllamaProvider(ModelProvider):  # pylint: disable=too-few-public-methods
    """Ollama-backed LLM provider."""

    def __init__(self, model: str = "qwen3.6:35b", num_ctx: int | None = _DEFAULT_NUM_CTX):
        self.model = model
        self.num_ctx = num_ctx

    def chat(self, messages: list[dict]) -> str:
        options = {"num_ctx": self.num_ctx} if self.num_ctx else None
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return response["message"]["content"]
