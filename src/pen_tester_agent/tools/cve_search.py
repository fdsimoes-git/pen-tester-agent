"""CVE search tool for vulnerability lookups."""

import re
from urllib.parse import quote

import httpx

from .base import Tool, ToolResult

_CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_CIRCL_API_BASE = "https://cve.circl.lu/api"
_REQUEST_TIMEOUT = 15
_MAX_SUMMARY_CHARS = 500


class CveSearchTool(Tool):
    """Search for known CVEs by ID or keyword."""

    name = "cve_search"
    description = (
        "Search for known CVEs (Common Vulnerabilities and Exposures) by keyword, "
        "product name, or CVE ID. Useful for identifying known vulnerabilities "
        "in discovered software versions during penetration testing."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": (
                "CVE ID (e.g., CVE-2021-44228) or keyword/product "
                "to search (e.g., 'Apache 2.4.49')"
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default: 5)",
            "default": 5,
        },
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "").strip()
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(output="Error: no query provided", success=False)

        try:
            max_results = max(1, int(max_results))
        except (TypeError, ValueError):
            max_results = 5

        if _CVE_ID_PATTERN.match(query):
            return self._lookup_cve_id(query.upper())
        return self._search_keyword(query, max_results)

    def _lookup_cve_id(self, cve_id: str) -> ToolResult:
        """Look up a specific CVE by its ID."""
        url = f"{_CIRCL_API_BASE}/cve/{cve_id}"
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
                resp = client.get(url)

            if resp.status_code == 404:
                return ToolResult(output=f"CVE {cve_id} not found.", success=True)
            resp.raise_for_status()

            data = resp.json()
            return ToolResult(output=_format_cve(data))
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error querying CVE API: {exc}", success=False)

    def _search_keyword(self, keyword: str, max_results: int) -> ToolResult:
        """Search CVEs by keyword/product name."""
        url = f"{_CIRCL_API_BASE}/search/{quote(keyword, safe='')}"
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
                resp = client.get(url)

            if resp.status_code == 404:
                return ToolResult(
                    output=f"No CVEs found for '{keyword}'.", success=True,
                )
            resp.raise_for_status()

            data = resp.json()
            if not data:
                return ToolResult(
                    output=f"No CVEs found for '{keyword}'.", success=True,
                )

            results = data[:max_results] if isinstance(data, list) else [data]
            lines = [f"Found CVEs for '{keyword}':\n"]
            for entry in results:
                lines.append(_format_cve(entry))
                lines.append("")

            return ToolResult(output="\n".join(lines))
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error querying CVE API: {exc}", success=False)


def _format_cve(data: dict) -> str:
    """Format a single CVE entry for display."""
    cve_id = data.get("id", data.get("cve", "Unknown"))
    summary = data.get("summary", data.get("descriptions", "No description available"))
    if isinstance(summary, list):
        summary = summary[0].get("value", "") if summary else ""
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "..."

    cvss = data.get("cvss", data.get("cvss-time", "N/A"))
    if cvss is None:
        cvss = "N/A"

    references = data.get("references", [])
    ref_str = ""
    if references:
        refs = references[:3]
        ref_str = "\n  References: " + ", ".join(refs)

    return f"  {cve_id}\n  CVSS: {cvss}\n  {summary}{ref_str}"
