<p align="center">
  <img src="assets/pen-tester-pic.png" alt="Pen-Tester Agent" width="400">
</p>

# pen-tester-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A penetration testing agent powered by local LLMs via [Ollama](https://ollama.com).

`pen-tester-agent` gives a locally-running language model the ability to propose and execute shell commands on your machine — with your approval on every step. It's designed for structured penetration testing workflows across six key domains.

## Penetration Testing Domains

1. **OSINT** — Open source intelligence gathering (WHOIS, DNS, subdomain enumeration, email harvesting)
2. **Enumeration** — Service enumeration, directory brute-forcing, SMB/LDAP/SNMP enumeration
3. **OS/Application Identification** — Version detection, OS fingerprinting, web technology fingerprinting
4. **CVE/Vulnerability Search** — Looking up known CVEs, searching exploit databases
5. **Vulnerability Testing** — Active testing for SQL injection, XSS, misconfigurations, default credentials
6. **Documentation** — Writing penetration test reports, documenting findings, saving evidence

## Installation

```bash
# From PyPI
pip install pen-tester-agent

# Or with pipx (isolated install)
pipx install pen-tester-agent

# Or directly from GitHub
pip install git+https://github.com/fdsimoes-git/pen-tester-agent.git
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- A model pulled (default: `qwen2.5-coder:3b`):
  ```
  ollama pull qwen2.5-coder:3b
  ```

## Usage

```bash
# Interactive mode — prompts you for a task
pen-tester-agent

# Pass a task directly
pen-tester-agent "scan open ports on 192.168.1.1"

# Use a different model
pen-tester-agent --model llama3.1:8b "review nginx access.log for suspicious requests"

# Limit iterations
pen-tester-agent --max-iterations 5 "enumerate subdomains of example.com"

# Running from source (development)
uv run pen-tester-agent
```

## Interactive CLI

When launched without a task, the agent presents an interactive menu (navigate with arrow keys):

- **New penetration test task** — describe a task and the agent works through it step by step
- **Quit**

During a session, every tool call is shown for approval via an arrow-key menu (approve / reject / edit args). Bash command output streams in real-time. A spinner indicates when the LLM is thinking or a non-bash tool is running.

At any interaction point during a session you can choose to generate a report from the session history or quit.

## How it works

1. You describe a task in natural language.
2. The agent (running locally via Ollama) reasons about the task and proposes a tool call (shell command, CVE lookup, file read/write, etc.).
3. You review and approve/edit/reject the action via arrow-key menu.
4. Bash output streams live to the terminal; the full output is fed back to the agent.
5. Repeat until the task is complete or you stop.
6. Generate a structured pentest report from the session at any time.

## Disclaimer

This tool executes shell commands on your machine. Always review proposed commands before approving them. Use responsibly and only on systems you own or have explicit written authorization to test. The authors are not responsible for any misuse or damage.
