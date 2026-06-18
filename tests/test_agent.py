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
        # follow-up menu: "2" = Quit
        monkeypatch.setattr("builtins.input", lambda _: "2")
        agent_loop("do something", provider, registry, max_iterations=5)

    def test_done_with_followup(self, monkeypatch, capsys):
        """Agent handles follow-up after done."""
        provider = FakeProvider([
            done_action("listed the files"),
            done_action("also counted them"),
        ])
        registry = default_registry()
        # "0" = Continue chatting, "count them too" = task text, "2" = Quit
        inputs = iter(["0", "count them too", "2"])
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
        # "0" = Approve tool, "2" = Quit follow-up
        inputs = iter(["0", "2"])
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
        # "1" = Reject tool, "2" = Quit follow-up
        inputs = iter(["1", "2"])
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
        # "2" = Quit follow-up
        monkeypatch.setattr("builtins.input", lambda _: "2")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "Unknown tool" in out

    def test_invalid_json_action(self, monkeypatch, capsys):
        """Agent treats malformed ACTION JSON as plain text and recovers."""
        provider = FakeProvider([
            'ACTION: {not: valid json}',
            done_action("gave up"),
        ])
        registry = default_registry()
        # Invalid JSON means no ACTION found, so agent shows as plain text
        # and prompts for user input: "0" = Reply, "continue" = text,
        # "2" = Quit follow-up after done
        inputs = iter(["0", "continue", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "gave up" in out.lower()

    def test_plain_text_response(self, monkeypatch, capsys):
        """Agent asks user for input when LLM doesn't use a tool."""
        provider = FakeProvider([
            "What IP address should I scan?",
            done_action("ok, quitting"),
        ])
        registry = default_registry()
        # "0" = Reply, "192.168.1.1" = text, "2" = Quit follow-up
        inputs = iter(["0", "192.168.1.1", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("scan something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "What IP" in out

    def test_max_iterations(self, monkeypatch, capsys):
        """Agent stops at the cap and offers continue/report/quit."""
        provider = FakeProvider(["thinking..." for _ in range(10)])
        registry = default_registry()
        # Each no-action turn: "0" = Reply, "continue" = text.
        # After 3 turns the cap menu appears: "2" = Quit.
        inputs = iter(["0", "continue", "0", "continue", "0", "continue", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("loop forever", provider, registry, max_iterations=3)
        out = capsys.readouterr().out
        assert "maximum iterations" in out.lower()

    def test_max_iterations_continue(self, monkeypatch, capsys):
        """Choosing 'continue' at the cap runs another batch of iterations."""
        provider = FakeProvider(
            ["thinking..."] + [done_action("finished after continue")]
        )
        registry = default_registry()
        # First (and only) turn of batch 1 replies: "0" + "go on".
        # Cap menu: "0" = Continue. Batch 2 turn 1 hits done, then "2" = Quit.
        inputs = iter(["0", "go on", "0", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("keep going", provider, registry, max_iterations=1)
        out = capsys.readouterr().out
        assert "finished after continue" in out

    def test_planning_step(self, monkeypatch, capsys):
        """With plan=True the first response is pinned as the plan, not run."""
        provider = FakeProvider([
            "PLAN:\n1. Recon the target\n2. Scan open ports",
            done_action("done planning test"),
        ])
        registry = default_registry()
        # "2" = Quit on the post-done follow-up menu.
        monkeypatch.setattr("builtins.input", lambda _: "2")
        agent_loop("scan target", provider, registry, max_iterations=5, plan=True)
        out = capsys.readouterr().out
        assert "Recon" in out          # plan was displayed
        assert "done planning test" in out

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
        # "0" = Approve tool, "2" = Quit follow-up
        inputs = iter(["0", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("read that file", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "secret data here" in out

    def test_done_tool_skips_approval(self, monkeypatch, capsys):
        """The done tool should not prompt for approval."""
        provider = FakeProvider([done_action("all finished")])
        registry = default_registry()
        # "2" = Quit follow-up
        monkeypatch.setattr("builtins.input", lambda _: "2")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "all finished" in out
        assert "Approve" not in out

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
        # "0" = Approve tool, "2" = Quit follow-up
        inputs = iter(["0", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("test", provider, registry, max_iterations=5, max_context_tokens=100000)

        # The second call to chat should have the tool result truncated
        second_call_msgs = received_messages[1]
        tool_result_msg = [m for m in second_call_msgs if "Tool 'bash' output:" in m["content"]]
        assert len(tool_result_msg) == 1
        assert "truncated" in tool_result_msg[0]["content"]

    def test_report_after_done(self, monkeypatch, capsys, tmp_path):
        """Report generation works from the post-done follow-up menu."""
        report_path = tmp_path / "report.txt"
        provider = FakeProvider([
            done_action("scan complete"),
            "## Executive Summary\nNo critical findings.",
        ])
        registry = default_registry()
        # "1" = Generate report, then report path, then done
        inputs = iter(["1", str(report_path)])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("scan 10.0.0.1", provider, registry, max_iterations=5)
        assert report_path.exists()
        content = report_path.read_text()
        assert "Executive Summary" in content

    def test_report_after_plain_text(self, monkeypatch, capsys, tmp_path):
        """Report generation works from the plain-text response menu."""
        report_path = tmp_path / "report.txt"
        provider = FakeProvider([
            "What should I scan?",
            "## Report\nAll clear.",
        ])
        registry = default_registry()
        # "1" = Generate report from user-input menu, then report path
        inputs = iter(["1", str(report_path)])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("scan something", provider, registry, max_iterations=5)
        assert report_path.exists()
        content = report_path.read_text()
        assert "Report" in content

    def test_report_unwritable_path(self, monkeypatch, capsys):
        """Report generation handles unwritable path gracefully."""
        provider = FakeProvider([
            done_action("done"),
            "## Report\nFindings here.",
        ])
        registry = default_registry()
        # "1" = Generate report, bad path
        inputs = iter(["1", "/nonexistent/dir/report.txt"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("test", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "Could not save report" in out
