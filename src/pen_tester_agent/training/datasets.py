"""Penetration testing task dataset generation for GRPO training."""

from pen_tester_agent.prompts import build_system_prompt
from pen_tester_agent.tools import default_registry

# Diverse penetration testing / security tasks for training.
SECURITY_TASKS = [
    # --- Network Reconnaissance ---
    "Scan the target 10.0.0.1 for open ports using a SYN scan.",
    "Run a full TCP port scan on 192.168.1.0/24 and identify running services.",
    "Perform a service version detection scan on 10.0.0.5 ports 80,443,8080.",
    "Scan the top 1000 ports on target 172.16.0.10 and save output to nmap_results.txt.",
    "Perform an OS detection scan on 10.0.0.1.",
    "Run a UDP scan on 192.168.1.1 for common services.",
    "Do an aggressive scan on 10.0.0.50 to enumerate services and scripts.",
    "Use masscan to quickly scan 10.0.0.0/8 for port 445.",
    "Scan 192.168.1.100 for SMB vulnerabilities using nmap scripts.",
    "Run a ping sweep on 10.0.0.0/24 to discover live hosts.",

    # --- DNS & Subdomain Enumeration ---
    "Enumerate subdomains for example.com using subfinder.",
    "Perform a DNS zone transfer attempt against ns1.example.com for example.com.",
    "Run dnsrecon on example.com to discover DNS records.",
    "Look up the WHOIS information for example.com.",
    "Use dig to query all DNS record types for example.com.",
    "Enumerate subdomains for target.com using amass.",
    "Resolve a list of subdomains from subdomains.txt to their IP addresses.",
    "Check for wildcard DNS on example.com.",

    # --- Web Application Testing ---
    "Run directory enumeration on http://10.0.0.1 using gobuster with a common wordlist.",
    "Use ffuf to fuzz for hidden directories on https://target.com.",
    "Scan http://10.0.0.1 for web vulnerabilities using nikto.",
    "Check the HTTP security headers on https://example.com.",
    "Use whatweb to fingerprint the web server at http://10.0.0.1.",
    "Test for SQL injection on http://10.0.0.1/login using sqlmap.",
    "Scan http://10.0.0.1 with nuclei using default templates.",
    "Check if a WAF is present on https://target.com using wafw00f.",
    "Run wpscan against http://10.0.0.1/wordpress/ to enumerate plugins.",
    "Use feroxbuster to discover content on https://target.com recursively.",
    "Send a GET request to https://target.com/api/v1/users and analyze the response.",
    "Test for CORS misconfiguration on https://target.com.",
    "Check the SSL/TLS configuration of target.com using testssl.",

    # --- File & Log Analysis ---
    "Read /var/log/auth.log and look for failed SSH login attempts.",
    "Analyze the Apache access log at /var/log/apache2/access.log for suspicious requests.",
    "Read the nginx configuration at /etc/nginx/nginx.conf and check for misconfigurations.",
    "Examine /etc/passwd for unusual user accounts.",
    "Read the SSH config at /etc/ssh/sshd_config and review security settings.",
    "Analyze the firewall rules in /etc/iptables/rules.v4.",
    "Read the contents of /etc/hosts and check for suspicious entries.",
    "Review the crontab entries in /etc/crontab for persistence mechanisms.",
    "Read /etc/shadow and check for accounts with weak password hashes.",
    "Analyze the web server error log at /var/log/apache2/error.log for attack signatures.",

    # --- OSINT & Information Gathering ---
    "Look up the IP geolocation for 8.8.8.8.",
    "Gather information about example.com using theHarvester.",
    "Use shodan CLI to search for exposed MongoDB instances.",
    "Perform a reverse DNS lookup on 93.184.216.34.",
    "Search for publicly exposed .git directories on target.com.",
    "Check if credentials for example.com appear in known breaches via HIBP API.",
    "Use recon-ng to gather OSINT on example.com.",

    # --- SMB / Active Directory ---
    "Enumerate SMB shares on 192.168.1.10 using smbclient.",
    "Run enum4linux on 10.0.0.1 to gather Windows host information.",
    "Use crackmapexec to check for null sessions on 192.168.1.0/24.",
    "List SMB shares on 10.0.0.5 using smbmap.",
    "Perform an LDAP search on 10.0.0.1 for all user objects.",

    # --- Credential Testing ---
    "Use hydra to brute-force SSH login on 10.0.0.1 with rockyou wordlist.",
    "Test default credentials on http://10.0.0.1:8080/admin.",
    "Run john the ripper on hashes.txt with the default wordlist.",
    "Use hashcat to crack MD5 hashes from hashes.txt.",

    # --- Exploitation & Post-Exploitation ---
    "Search for known exploits for Apache 2.4.49 using searchsploit.",
    "Check if 10.0.0.1 is vulnerable to EternalBlue (MS17-010).",
    "Use openssl to test for Heartbleed on target.com:443.",
    "Enumerate SNMP community strings on 10.0.0.1 using snmpwalk.",
    "Test for shellshock vulnerability on http://10.0.0.1/cgi-bin/test.cgi.",

    # --- CVE / Vulnerability Search ---
    "Search for known CVEs affecting Apache 2.4.49.",
    "Look up CVE-2021-44228 to understand the Log4Shell vulnerability.",
    "Find known vulnerabilities for OpenSSH 8.2.",
    "Search for CVEs related to Microsoft Exchange ProxyShell.",
    "Look up exploits for the version of nginx discovered on the target.",

    # --- Documentation / Reporting ---
    "Write a summary report of the scan findings to report.txt.",
    "Create a vulnerability report documenting the SQL injection found on the login page.",
    "Save the nmap scan results to a structured report file.",
    "Write a penetration test executive summary to executive_summary.txt.",

    # --- Enumeration / Post-Exploitation ---
    "Check which ports are currently listening on this machine.",
    "Display the current network interfaces and their IP addresses.",
    "Show all active network connections and the processes using them.",
    "Check the system for SUID binaries that could be used for privilege escalation.",
    "List all running processes and check for suspicious ones.",
    "Verify the integrity of system binaries using package manager.",
    "Check for world-writable directories in /tmp and /var/tmp.",
    "Examine environment variables for leaked credentials.",
    "Identify kernel version and check for known privilege escalation exploits.",
    "Check which users have sudo privileges on this system.",
]


def _build_system_context() -> str:
    """Build the system prompt that will be prepended to training examples."""
    registry = default_registry()
    return build_system_prompt(registry)


def build_dataset(size: int = 500) -> list[dict]:
    """Build a list of penetration testing task prompts for GRPO training.

    Each item contains the full conversation prompt (system + user)
    so the model learns the exact protocol used at inference time.

    Args:
        size: Number of training examples (tasks are cycled).

    Returns:
        List of dicts with a "prompt" key containing message lists.
    """
    system_prompt = _build_system_context()

    items = []
    for i in range(size):
        task = SECURITY_TASKS[i % len(SECURITY_TASKS)]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        items.append({"prompt": messages})

    return items
