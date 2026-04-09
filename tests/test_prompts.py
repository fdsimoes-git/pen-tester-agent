import json

from pen_tester_agent.prompts import find_action, build_system_prompt
from pen_tester_agent.tools import default_registry
from pen_tester_agent.tools.base import ToolRegistry


class TestPatterns:
    def test_action_pattern_simple(self):
        text = 'ACTION: {"tool": "bash", "args": {"command": "ls"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"
        assert parsed["args"]["command"] == "ls"

    def test_action_pattern_with_preceding_text(self):
        text = 'Let me check that.\n\nACTION: {"tool": "read_file", "args": {"path": "/etc/hosts"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "read_file"

    def test_action_pattern_no_match(self):
        text = "I will just explain things."
        m = find_action(text)
        assert m is None

    def test_action_pattern_done_tool(self):
        text = 'ACTION: {"tool": "done", "args": {"summary": "completed the port scan"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "done"
        assert parsed["args"]["summary"] == "completed the port scan"

    def test_action_pattern_nested_json(self):
        text = 'ACTION: {"tool": "http_request", "args": {"url": "https://example.com", "headers": {"Accept": "application/json"}}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["args"]["headers"]["Accept"] == "application/json"


class TestFindAction:
    def test_simple(self):
        text = 'ACTION: {"tool": "bash", "args": {"command": "ls"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"

    def test_nested_json(self):
        text = 'ACTION: {"tool": "http_request", "args": {"url": "https://example.com", "headers": {"Accept": "application/json"}}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["args"]["headers"]["Accept"] == "application/json"

    def test_no_match(self):
        text = "Just some text without any action."
        assert find_action(text) is None

    def test_multiple_actions_takes_first(self):
        text = (
            'I will scan.\n\n'
            'ACTION: {"tool": "bash", "args": {"command": "nmap 192.168.1.1"}}\n'
            'Then write a report.\n\n'
            'ACTION: {"tool": "write_file", "args": {"path": "report.txt", "content": "done"}}\n'
            'ACTION: {"tool": "done", "args": {"summary": "finished"}}'
        )
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"
        assert parsed["args"]["command"] == "nmap 192.168.1.1"

    def test_with_preceding_markdown(self):
        text = (
            '### Explanation\n'
            '- **Tool**: `bash`\n'
            '- **Command**: `nmap -sV 10.0.0.1`\n\n'
            'ACTION: {"tool": "bash", "args": {"command": "nmap -sV 10.0.0.1"}}\n'
        )
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"

    def test_markdown_code_block(self):
        text = (
            "Let's perform a scan.\n\n"
            "ACTION\n\n"
            '```json\n'
            '{\n'
            '  "tool": "bash",\n'
            '  "args": {\n'
            '    "command": "nmap -sV 192.168.86.80"\n'
            '  }\n'
            '}\n'
            '```\n'
        )
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"
        assert parsed["args"]["command"] == "nmap -sV 192.168.86.80"

    def test_action_no_colon(self):
        text = 'ACTION\n{"tool": "bash", "args": {"command": "ls"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"

    def test_string_with_braces(self):
        text = 'ACTION: {"tool": "write_file", "args": {"path": "test.json", "content": "{\\\"key\\\": \\\"val\\\"}"}}'
        m = find_action(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "write_file"


class TestBuildSystemPrompt:
    def test_contains_tool_names(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        assert "bash" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "http_request" in prompt
        assert "cve_search" in prompt
        assert "done" in prompt

    def test_contains_action_format(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        assert "ACTION:" in prompt

    def test_done_tool_documented(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        # done tool should appear in the tool catalog
        assert "done" in prompt
        assert "summary" in prompt

    def test_empty_registry(self):
        reg = ToolRegistry()
        prompt = build_system_prompt(reg)
        # Should still produce a valid prompt, just with no tool docs
        assert "ACTION:" in prompt
