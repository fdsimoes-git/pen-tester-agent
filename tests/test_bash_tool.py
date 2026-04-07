import pytest

from pen_tester_agent.tools.bash_tool import BashTool


@pytest.fixture
def bash():
    return BashTool()


class TestBashTool:
    def test_metadata(self, bash):
        assert bash.name == "bash"
        assert bash.requires_approval is True

    def test_simple_command(self, bash):
        r = bash.execute(command="echo hello")
        assert r.success is True
        assert "hello" in r.output

    def test_command_with_stderr(self, bash):
        r = bash.execute(command="echo err >&2")
        assert r.success is True
        assert "err" in r.output

    def test_no_command(self, bash):
        r = bash.execute()
        assert r.success is False
        assert "no command" in r.output.lower()

    def test_empty_command(self, bash):
        r = bash.execute(command="")
        assert r.success is False

    def test_failing_command_still_returns_output(self, bash):
        r = bash.execute(command="ls /nonexistent_path_xyz 2>&1")
        assert r.success is True  # subprocess ran fine, just non-zero exit
        assert r.output  # should have error text

    def test_timeout(self, bash):
        r = bash.execute(command="sleep 10", timeout=1)
        assert r.success is False
        assert "timed out" in r.output.lower()

    def test_custom_timeout(self, bash):
        r = bash.execute(command="echo fast", timeout=5)
        assert r.success is True
        assert "fast" in r.output

    def test_no_output_command(self, bash):
        r = bash.execute(command="true")
        assert r.success is True
        assert r.output == "(no output)"
