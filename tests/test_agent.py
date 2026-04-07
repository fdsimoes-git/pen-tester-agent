import json

from pen_tester_agent.agent import agent_loop
from pen_tester_agent.providers.base import ModelProvider
from pen_tester_agent.tools import default_registry


def done_action(summary: str) -> str:
    """Helper to build a done ACTION string."""
    return f'ACTION: {json.dumps({"tool": "done", "args": {"summary": summary}})}'


class FakeProvider(ModelProvider):
    """A fake provider that returns pre-scripted responses."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def chat(self, messages: list[dict]) -> str:
        return next(self._responses)


class TestAgentLoop:
    def test_done_on_first_response(self, monkeypatch):
        """Agent exits cleanly when LLM calls done immediately."""
        provider = FakeProvider([done_action("nothing to do")])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)

    def test_done_with_followup(self, monkeypatch, capsys):
        """Agent handles follow-up after done."""
        provider = FakeProvider([
            done_action("listed the files"),
            done_action("also counted them"),
        ])
        registry = default_registry()
        inputs = iter(["count them too", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("list files", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "listed the files" in out
        assert "also counted them" in out

    def test_tool_call_approved(self, monkeypatch, capsys):
        """Agent executes a tool when user approves."""
        action = json.dumps({"tool": "bash", "args": {"command": "echo test123"}})
        provider = FakeProvider([
            f"I'll run echo.\n\nACTION: {action}",
            done_action("ran the command"),
        ])
        registry = default_registry()
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("echo test", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "test123" in out

    def test_tool_call_rejected(self, monkeypatch, capsys):
        """Agent handles user rejecting a tool call."""
        action = json.dumps({"tool": "bash", "args": {"command": "rm -rf /"}})
        provider = FakeProvider([
            f"ACTION: {action}",
            done_action("understood, cancelled"),
        ])
        registry = default_registry()
        inputs = iter(["n", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("do something dangerous", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_unknown_tool(self, monkeypatch, capsys):
        """Agent handles LLM requesting a non-existent tool."""
        action = json.dumps({"tool": "nonexistent", "args": {}})
        provider = FakeProvider([
            f"ACTION: {action}",
            done_action("ok nevermind"),
        ])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "Unknown tool" in out

    def test_invalid_json_action(self, monkeypatch, capsys):
        """Agent handles malformed JSON in ACTION and recovers."""
        provider = FakeProvider([
            'ACTION: {not: valid json}',
            done_action("gave up"),
        ])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "could not parse" in out.lower()
        assert "gave up" in out.lower()

    def test_plain_text_response(self, monkeypatch, capsys):
        """Agent asks user for input when LLM doesn't use a tool."""
        provider = FakeProvider([
            "What IP address should I scan?",
            done_action("ok, quitting"),
        ])
        registry = default_registry()
        inputs = iter(["192.168.1.1", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("scan something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "What IP" in out

    def test_max_iterations(self, monkeypatch, capsys):
        """Agent stops at max iterations."""
        provider = FakeProvider(["thinking..." for _ in range(10)])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "continue")
        agent_loop("loop forever", provider, registry, max_iterations=3)
        out = capsys.readouterr().out
        assert "maximum iterations" in out.lower()

    def test_read_file_tool_integration(self, monkeypatch, capsys, tmp_path):
        """Agent can use the read_file tool end-to-end."""
        target = tmp_path / "info.txt"
        target.write_text("secret data here")
        action = json.dumps({"tool": "read_file", "args": {"path": str(target)}})
        provider = FakeProvider([
            f"I'll read the file.\n\nACTION: {action}",
            done_action("read the file"),
        ])
        registry = default_registry()
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("read that file", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "secret data here" in out

    def test_done_tool_skips_approval(self, monkeypatch, capsys):
        """The done tool should not prompt for approval."""
        provider = FakeProvider([done_action("all finished")])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "all finished" in out
        assert "Execute?" not in out

    def test_large_output_is_truncated_in_context(self, monkeypatch, capsys):
        """Large tool outputs should be truncated in the context sent to LLM."""
        # Use a FakeProvider that captures the messages it receives
        received_messages = []

        class CapturingProvider(ModelProvider):
            def __init__(self, responses):
                self._responses = iter(responses)

            def chat(self, messages):
                received_messages.append(list(messages))
                return next(self._responses)

        action = json.dumps({"tool": "bash", "args": {"command": "echo " + "x" * 5000}})
        provider = CapturingProvider([
            f"ACTION: {action}",
            done_action("done"),
        ])
        registry = default_registry()
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("test", provider, registry, max_iterations=5, max_context_tokens=100000)

        # The second call to chat should have the tool result truncated
        second_call_msgs = received_messages[1]
        tool_result_msg = [m for m in second_call_msgs if "Tool 'bash' output:" in m["content"]]
        assert len(tool_result_msg) == 1
        assert "truncated" in tool_result_msg[0]["content"]
