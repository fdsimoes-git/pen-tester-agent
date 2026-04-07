# pen-tester-agent

A penetration testing agent powered by local LLMs via [Ollama](https://ollama.com).

`pen-tester-agent` gives a locally-running language model the ability to propose and execute shell commands on your machine — with your approval on every step. It's designed for structured penetration testing workflows across six key domains.

## Penetration Testing Domains

1. **OSINT** — Open source intelligence gathering (WHOIS, DNS, subdomain enumeration, email harvesting)
2. **Enumeration** — Service enumeration, directory brute-forcing, SMB/LDAP/SNMP enumeration
3. **OS/Application Identification** — Version detection, OS fingerprinting, web technology fingerprinting
4. **CVE/Vulnerability Search** — Looking up known CVEs, searching exploit databases
5. **Vulnerability Testing** — Active testing for SQL injection, XSS, misconfigurations, default credentials
6. **Documentation** — Writing penetration test reports, documenting findings, saving evidence

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- A model pulled (default: `qwen2.5-coder:7b`):
  ```
  ollama pull qwen2.5-coder:7b
  ```
- [uv](https://docs.astral.sh/uv/) (for running/developing)

## Usage

```bash
# Interactive mode — prompts you for a task
uv run pen-tester-agent

# Pass a task directly
uv run pen-tester-agent "scan open ports on 192.168.1.1"

# Use a different model
uv run pen-tester-agent --model llama3.1:8b "review nginx access.log for suspicious requests"

# Limit iterations
uv run pen-tester-agent --max-iterations 5 "enumerate subdomains of example.com"
```

Every command the agent proposes is shown to you first. You can approve (`y`), reject (`n`), or edit (`e`) before anything runs.

## How it works

1. You describe a task in natural language.
2. The agent (running locally via Ollama) reasons about the task and proposes a tool call (shell command, CVE lookup, file read/write, etc.).
3. You review and approve/edit/reject the action.
4. The output is fed back to the agent for the next step.
5. Repeat until the task is complete or you stop.

## Disclaimer

This tool executes shell commands on your machine. Always review proposed commands before approving them. Use responsibly and only on systems you own or have explicit written authorization to test. The authors are not responsible for any misuse or damage.
