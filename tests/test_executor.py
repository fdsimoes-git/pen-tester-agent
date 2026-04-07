from pen_tester_agent.executor import ask_tool_approval


class TestAskToolApproval:
    def test_approve(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is True
        assert args == {"command": "ls"}

    def test_reject(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        args, execute = ask_tool_approval("bash", {"command": "rm -rf /"})
        assert execute is False
        assert args is None

    def test_edit_valid_json(self, monkeypatch):
        inputs = iter(["e", '{"command": "ls -la"}'])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is True
        assert args == {"command": "ls -la"}

    def test_edit_empty_cancels(self, monkeypatch):
        inputs = iter(["e", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is False
        assert args is None

    def test_edit_invalid_json_then_cancel(self, monkeypatch):
        # Invalid JSON prints warning, loops back to prompt, then "n" cancels
        inputs = iter(["e", "not json", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is False
        assert args is None

    def test_retry_on_invalid_input(self, monkeypatch):
        inputs = iter(["x", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is True
        assert args == {"command": "ls"}
