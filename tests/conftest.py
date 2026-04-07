import pytest

from pen_tester_agent.tools import default_registry
from pen_tester_agent.tools.base import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    """A fresh default tool registry for each test."""
    return default_registry()
