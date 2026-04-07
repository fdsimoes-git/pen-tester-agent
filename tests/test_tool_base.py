from pen_tester_agent.tools.base import Tool, ToolRegistry, ToolResult


class DummyTool(Tool):
    name = "dummy"
    description = "A dummy tool for testing"
    parameters = {"x": {"type": "string", "description": "a param"}}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=f"got x={kwargs.get('x')}")


class AnotherTool(Tool):
    name = "another"
    description = "Another tool"
    parameters = {}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="ok")


# --- ToolResult ---

class TestToolResult:
    def test_defaults_to_success(self):
        r = ToolResult(output="hello")
        assert r.success is True
        assert r.output == "hello"

    def test_explicit_failure(self):
        r = ToolResult(output="bad", success=False)
        assert r.success is False


# --- Tool ---

class TestTool:
    def test_schema(self):
        t = DummyTool()
        s = t.schema()
        assert s["name"] == "dummy"
        assert s["description"] == "A dummy tool for testing"
        assert "x" in s["parameters"]

    def test_execute(self):
        t = DummyTool()
        r = t.execute(x="hello")
        assert r.output == "got x=hello"
        assert r.success is True


# --- ToolRegistry ---

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        t = DummyTool()
        reg.register(t)
        assert reg.get("dummy") is t

    def test_get_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(DummyTool())
        reg.register(AnotherTool())
        names = [t.name for t in reg.list_tools()]
        assert "dummy" in names
        assert "another" in names

    def test_schema_returns_list(self):
        reg = ToolRegistry()
        reg.register(DummyTool())
        s = reg.schema()
        assert isinstance(s, list)
        assert len(s) == 1
        assert s[0]["name"] == "dummy"

    def test_schema_text_contains_tool_info(self):
        reg = ToolRegistry()
        reg.register(DummyTool())
        text = reg.schema_text()
        assert "### dummy" in text
        assert "A dummy tool for testing" in text

    def test_register_overwrites_same_name(self):
        reg = ToolRegistry()
        reg.register(DummyTool())
        t2 = DummyTool()
        reg.register(t2)
        assert reg.get("dummy") is t2
        assert len(reg.list_tools()) == 1
