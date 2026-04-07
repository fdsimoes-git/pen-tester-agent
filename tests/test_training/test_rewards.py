"""Tests for GRPO reward functions."""

from pen_tester_agent.training.rewards import (
    _extract_action,
    command_quality_reward,
    explanation_reward,
    format_reward,
    tool_selection_reward,
)


# --- Helpers ---


def _wrap(text: str) -> list:
    """Wrap text as a single completion in GRPO format."""
    return [[{"content": text}]]


def _wrap_multi(*texts: str) -> list:
    """Wrap multiple completions."""
    return [[{"content": t}] for t in texts]


# --- _extract_action ---


class TestExtractAction:
    def test_valid_action(self):
        text = 'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        result = _extract_action(text)
        assert result == {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}

    def test_action_with_preceding_text(self):
        text = 'I will scan the target.\nACTION: {"tool": "bash", "args": {"command": "nmap -sV 10.0.0.1"}}'
        result = _extract_action(text)
        assert result["tool"] == "bash"

    def test_no_action(self):
        assert _extract_action("Just some text without an action") is None

    def test_invalid_json(self):
        assert _extract_action("ACTION: {not valid json}") is None

    def test_empty_string(self):
        assert _extract_action("") is None


# --- format_reward ---


class TestFormatReward:
    def test_valid_action_with_keys(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        )
        scores = format_reward(completions)
        assert scores == [2.0]

    def test_valid_json_missing_keys(self):
        completions = _wrap('ACTION: {"name": "bash"}')
        scores = format_reward(completions)
        assert scores == [0.5]

    def test_no_action(self):
        completions = _wrap("I don't know how to do that.")
        scores = format_reward(completions)
        assert scores == [-1.0]

    def test_multiple_completions(self):
        completions = _wrap_multi(
            'ACTION: {"tool": "bash", "args": {"command": "ls"}}',
            "no action here",
        )
        scores = format_reward(completions)
        assert scores == [2.0, -1.0]


# --- tool_selection_reward ---


class TestToolSelectionReward:
    def test_correct_tool_for_scan(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        )
        prompts = [[{"role": "user", "content": "Scan the target for open ports"}]]
        scores = tool_selection_reward(completions, prompts=prompts)
        assert scores == [2.0]

    def test_correct_tool_for_read(self):
        completions = _wrap(
            'ACTION: {"tool": "read_file", "args": {"path": "/etc/passwd"}}'
        )
        prompts = [[{"role": "user", "content": "Read the passwd file"}]]
        scores = tool_selection_reward(completions, prompts=prompts)
        assert scores == [2.0]

    def test_wrong_tool(self):
        completions = _wrap(
            'ACTION: {"tool": "write_file", "args": {"path": "x.txt", "content": "hi"}}'
        )
        prompts = [[{"role": "user", "content": "Scan the target for open ports"}]]
        scores = tool_selection_reward(completions, prompts=prompts)
        assert scores == [-1.0]

    def test_invalid_tool_name(self):
        completions = _wrap(
            'ACTION: {"tool": "nonexistent", "args": {}}'
        )
        prompts = [[{"role": "user", "content": "Do something"}]]
        scores = tool_selection_reward(completions, prompts=prompts)
        assert scores == [-1.0]

    def test_no_action(self):
        completions = _wrap("No action here")
        prompts = [[{"role": "user", "content": "Scan ports"}]]
        scores = tool_selection_reward(completions, prompts=prompts)
        assert scores == [0.0]

    def test_no_prompts(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        )
        scores = tool_selection_reward(completions)
        assert scores == [0.0]


# --- command_quality_reward ---


class TestCommandQualityReward:
    def test_security_tool_command(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "nmap -sV 10.0.0.1"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [3.0]

    def test_general_utility_command(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "grep -r password /var/log"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [1.0]

    def test_dangerous_command(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "rm -rf /"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [-2.0]

    def test_non_bash_tool(self):
        completions = _wrap(
            'ACTION: {"tool": "read_file", "args": {"path": "/etc/passwd"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [0.0]

    def test_empty_command(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": ""}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [-1.0]

    def test_sudo_security_tool(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "sudo nmap -sS 10.0.0.1"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [3.0]

    def test_fork_bomb(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": ":(){ :|:& };:"}}'
        )
        scores = command_quality_reward(completions)
        assert scores == [-2.0]


# --- explanation_reward ---


class TestExplanationReward:
    def test_explanation_before_action(self):
        completions = _wrap(
            "I'll perform a port scan on the target to identify open services.\n"
            'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        )
        scores = explanation_reward(completions)
        assert scores == [1.0]

    def test_no_explanation(self):
        completions = _wrap(
            'ACTION: {"tool": "bash", "args": {"command": "nmap 10.0.0.1"}}'
        )
        scores = explanation_reward(completions)
        assert scores == [0.0]

    def test_no_action_at_all(self):
        completions = _wrap("Just talking, no action.")
        scores = explanation_reward(completions)
        assert scores == [0.0]

    def test_very_short_prefix(self):
        completions = _wrap(
            'Ok\nACTION: {"tool": "bash", "args": {"command": "ls"}}'
        )
        scores = explanation_reward(completions)
        assert scores == [0.0]  # "Ok" is <= 10 chars
