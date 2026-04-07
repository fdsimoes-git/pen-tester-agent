"""Base classes for the tool system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


@dataclass
class ToolResult:
    """Result from a tool execution."""
    output: str
    success: bool = True
    terminates: bool = False


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema describing accepted arguments
    requires_approval: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # skip validation on intermediate ABCs
        for attr in ("name", "description", "parameters"):
            if not hasattr(cls, attr):
                raise TypeError(
                    f"Tool subclass {cls.__name__!r} must define '{attr}'"
                )

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments and return a result."""

    def schema(self) -> dict:
        """Return a serializable description of this tool for the LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry that holds available tools and provides lookup + serialization."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def schema(self) -> list[dict]:
        """Serialize all tools for inclusion in the LLM prompt."""
        return [t.schema() for t in self._tools.values()]

    def schema_text(self) -> str:
        """Return a human-readable text description of all tools."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            lines.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}")
            lines.append("")
        return "\n".join(lines)
