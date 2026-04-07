"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class ModelProvider(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant's response text."""
