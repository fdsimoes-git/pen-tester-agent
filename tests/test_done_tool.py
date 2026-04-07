from pen_tester_agent.tools.done import DoneTool


class TestDoneTool:
    def test_metadata(self):
        tool = DoneTool()
        assert tool.name == "done"
        assert tool.requires_approval is False

    def test_execute_with_summary(self):
        tool = DoneTool()
        r = tool.execute(summary="scan complete")
        assert r.success is True
        assert r.terminates is True
        assert r.output == "scan complete"

    def test_execute_without_summary(self):
        tool = DoneTool()
        r = tool.execute()
        assert r.terminates is True
        assert r.output == "Task complete."

    def test_schema(self):
        tool = DoneTool()
        s = tool.schema()
        assert s["name"] == "done"
        assert "summary" in s["parameters"]
