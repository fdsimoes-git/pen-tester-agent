"""System prompt construction and response parsing patterns."""

import re

from .tools.base import ToolRegistry


def _extract_json_object(text, start):
    """Extract a complete JSON object from text starting at a '{'.

    Counts brace depth to handle nested objects correctly.
    Returns the JSON substring or None if braces are unbalanced.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def find_action(content):
    """Find and extract the first ACTION JSON object from LLM output.

    Handles multiple formats:
    - ACTION: {"tool": ...}           (inline)
    - ACTION:\\n```json\\n{...}\\n```   (markdown code block)
    - ACTION:\\n{...}                  (newline-separated)

    Returns a match-like object with group(1) containing the JSON,
    or None if no valid ACTION block is found.
    """
    # Find "ACTION" anchored to start of line with word boundary
    action_re = re.compile(r"^\s*ACTION\b:?\s*", re.IGNORECASE | re.MULTILINE)

    for match in action_re.finditer(content):
        after = content[match.end():]

        # Strip optional markdown code fence (```json or ```)
        fence_re = re.compile(r"^```(?:json)?\s*\n?")
        after = fence_re.sub("", after)

        # Find the first '{' in the remaining text
        brace_pos = after.find("{")
        if brace_pos == -1:
            continue

        json_str = _extract_json_object(after, brace_pos)
        if json_str is None:
            continue

        return _ActionMatch(json_str, start=match.start())

    return None


class _ActionMatch:
    """Lightweight match-like wrapper for extracted ACTION JSON."""

    def __init__(self, json_str, start):
        self._json = json_str
        self.start = start

    def group(self, n):  # pylint: disable=unused-argument
        """Return the captured JSON string."""
        return self._json


def build_system_prompt(registry: ToolRegistry) -> str:
    """Build the system prompt dynamically from the tool registry."""
    tool_docs = registry.schema_text()

    return f"""You are a penetration testing assistant running on the user's machine.
You help with structured penetration testing across these domains:

1. **OSINT** — Open source intelligence gathering: WHOIS, DNS records, subdomain enumeration, email harvesting, social media reconnaissance, Shodan/Censys searches.
2. **Enumeration** — Service enumeration, directory brute-forcing, SMB/LDAP/SNMP enumeration, user and share discovery.
3. **OS/Application Identification** — Service version detection, OS fingerprinting, web technology fingerprinting, banner grabbing.
4. **CVE/Vulnerability Search** — Looking up known CVEs for discovered software versions, searching exploit databases, checking for known vulnerable configurations.
5. **Vulnerability Testing** — Active testing for SQL injection, XSS, misconfigurations, default credentials, SSL/TLS weaknesses, and other vulnerabilities.
6. **Documentation** — Writing penetration test reports, documenting findings, saving evidence, creating executive summaries.

## Available Tools

{tool_docs}

## How to Use Tools

When you need to use a tool, respond with this exact format:

ACTION: {{"tool": "<tool_name>", "args": {{<arguments>}}}}

Examples:
- ACTION: {{"tool": "bash", "args": {{"command": "nmap -sV 192.168.1.1"}}}}
- ACTION: {{"tool": "read_file", "args": {{"path": "/etc/hosts"}}}}
- ACTION: {{"tool": "http_request", "args": {{"method": "GET", "url": "https://example.com"}}}}
- ACTION: {{"tool": "cve_search", "args": {{"query": "Apache 2.4.49"}}}}
- ACTION: {{"tool": "write_file", "args": {{"path": "report.txt", "content": "Scan results..."}}}}
- ACTION: {{"tool": "done", "args": {{"summary": "Completed the port scan and found 3 open ports"}}}}

## Rules

- Use ONE tool call per response.
- Briefly explain what you're about to do and why before the ACTION line.
- Choose the most appropriate tool for the task.
- When the task is complete, use the "done" tool with a summary of what was accomplished.
- If you need more information from the user, just ask — don't guess.
- Be aware of the current working directory and OS context."""
