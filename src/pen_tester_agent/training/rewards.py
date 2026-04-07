"""GRPO reward functions for penetration testing agent fine-tuning.

Each reward function follows the TRL GRPO signature:
    (completions, **kwargs) -> list[float]

Where completions is a list of conversation lists, each containing
a single assistant message dict with a "content" key.
"""

import json
import re

from pen_tester_agent.prompts import ACTION_PATTERN

VALID_TOOLS = {"bash", "read_file", "write_file", "http_request", "cve_search", "done"}

# Security tools we expect to see in bash commands.
SECURITY_TOOLS = {
    "nmap", "nikto", "gobuster", "dirb", "dirbuster", "ffuf",
    "sqlmap", "hydra", "john", "hashcat", "masscan", "amass",
    "subfinder", "httpx", "nuclei", "wpscan", "enum4linux",
    "smbclient", "netcat", "nc", "ncat", "socat", "tcpdump",
    "wireshark", "tshark", "burpsuite", "msfconsole", "metasploit",
    "searchsploit", "exploitdb", "whatweb", "wafw00f", "fierce",
    "dnsrecon", "dig", "whois", "theHarvester", "recon-ng",
    "shodan", "censys", "curl", "wget", "openssl", "ssh",
    "scp", "ftp", "telnet", "snmpwalk", "crackmapexec",
    "impacket", "responder", "bloodhound", "ldapsearch",
    "rpcclient", "smbmap", "feroxbuster", "rustscan",
    "testssl", "sslyze", "arjun", "paramspider",
}

# Commands that are destructive and should be penalized.
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bformat\b",
    r">\s*/dev/sd",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r"\bsystemctl\s+(stop|disable)\s+(sshd|network|firewall)",
    r":(){ :|:& };:",  # fork bomb
]

# Mapping of task keywords to expected tools.
TASK_TOOL_MAP = {
    "scan": "bash",
    "nmap": "bash",
    "port": "bash",
    "enumerate": "bash",
    "directory": "bash",
    "brute": "bash",
    "exploit": "bash",
    "reconnaissance": "bash",
    "recon": "bash",
    "network": "bash",
    "vulnerability": "bash",
    "dns": "bash",
    "subdomain": "bash",
    "web": "bash",
    "ssl": "bash",
    "tls": "bash",
    "read": "read_file",
    "review": "read_file",
    "analyze log": "read_file",
    "config": "read_file",
    "examine": "read_file",
    "write report": "write_file",
    "save": "write_file",
    "create report": "write_file",
    "http": "http_request",
    "api": "http_request",
    "request": "http_request",
    "fetch": "http_request",
    "cve": "cve_search",
    "vulnerability search": "cve_search",
    "exploit search": "cve_search",
}


def _extract_action(text: str) -> dict | None:
    """Extract the ACTION JSON from a completion text."""
    match = ACTION_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, IndexError):
        return None


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for producing a valid ACTION: {...} block.

    +2.0  valid ACTION JSON with 'tool' and 'args' keys
    +0.5  valid JSON but missing expected keys
    -1.0  no ACTION found or unparseable JSON
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = _extract_action(response)
        if action is None:
            scores.append(-1.0)
        elif "tool" in action and "args" in action:
            scores.append(2.0)
        else:
            scores.append(0.5)
    return scores


def tool_selection_reward(completions, prompts=None, **kwargs) -> list[float]:
    """Reward for choosing the right tool for the given task.

    +2.0  correct tool for the task keywords
     0.0  no clear mapping or neutral
    -1.0  clearly wrong tool selection
    """
    scores = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        action = _extract_action(response)

        if action is None or "tool" not in action:
            scores.append(0.0)
            continue

        tool = action["tool"]
        if tool not in VALID_TOOLS:
            scores.append(-1.0)
            continue

        # Extract the user task from prompts if available.
        task_text = ""
        if prompts and i < len(prompts):
            prompt_msgs = prompts[i]
            if isinstance(prompt_msgs, list):
                for msg in prompt_msgs:
                    if msg.get("role") == "user":
                        task_text = msg.get("content", "").lower()
            elif isinstance(prompt_msgs, str):
                task_text = prompt_msgs.lower()

        if not task_text:
            scores.append(0.0)
            continue

        # Check if the selected tool matches any keyword mapping.
        expected = None
        for keyword, expected_tool in TASK_TOOL_MAP.items():
            if keyword in task_text:
                expected = expected_tool
                break

        if expected is None:
            scores.append(0.0)
        elif tool == expected:
            scores.append(2.0)
        else:
            scores.append(-1.0)
    return scores


def command_quality_reward(completions, **kwargs) -> list[float]:
    """Reward for bash command quality in security context.

    +3.0  well-formed command using a recognized security tool
    +1.0  valid command with general utilities (grep, cat, etc.)
     0.0  non-bash tool call (neutral)
    -2.0  dangerous/destructive command
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = _extract_action(response)

        if action is None or action.get("tool") != "bash":
            scores.append(0.0)
            continue

        command = action.get("args", {}).get("command", "")
        if not command:
            scores.append(-1.0)
            continue

        # Check for dangerous patterns first.
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                scores.append(-2.0)
                break
        else:
            # Check if command starts with or contains a security tool.
            cmd_first = command.strip().split()[0] if command.strip() else ""
            # Handle sudo prefix.
            if cmd_first == "sudo" and len(command.strip().split()) > 1:
                cmd_first = command.strip().split()[1]

            if cmd_first in SECURITY_TOOLS:
                scores.append(3.0)
            elif cmd_first in {"grep", "cat", "head", "tail", "awk", "sed",
                               "find", "ls", "ps", "netstat", "ss", "ip",
                               "ifconfig", "route", "iptables", "uname",
                               "whoami", "id", "hostname", "env", "set"}:
                scores.append(1.0)
            else:
                scores.append(0.0)
    return scores


def explanation_reward(completions, **kwargs) -> list[float]:
    """Reward for including an explanation before the ACTION call.

    The pen-tester-agent system prompt requires a brief explanation before each
    ACTION line.

    +1.0  text present before the ACTION line
     0.0  ACTION is the only content
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        match = ACTION_PATTERN.search(response)
        if match is None:
            scores.append(0.0)
            continue

        text_before = response[:match.start()].strip()
        if len(text_before) > 10:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores
