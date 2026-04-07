"""Integration tests for the default registry and tool discovery."""


class TestDefaultRegistry:
    def test_all_builtin_tools_registered(self, registry):
        names = {t.name for t in registry.list_tools()}
        assert names == {"bash", "read_file", "write_file", "http_request", "cve_search", "done"}

    def test_all_tools_have_required_attrs(self, registry):
        for tool in registry.list_tools():
            assert isinstance(tool.name, str) and tool.name
            assert isinstance(tool.description, str) and tool.description
            assert isinstance(tool.parameters, dict)
            assert isinstance(tool.requires_approval, bool)

    def test_all_tools_produce_valid_schema(self, registry):
        schemas = registry.schema()
        assert len(schemas) == 6
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "parameters" in s

    def test_schema_text_nonempty(self, registry):
        text = registry.schema_text()
        assert len(text) > 100

    def test_each_tool_is_callable(self, registry):
        """Every tool should be callable without crashing (even with bad args)."""
        for tool in registry.list_tools():
            r = tool.execute()  # no args — should return an error, not crash
            assert r.output  # should have some output (error message)
