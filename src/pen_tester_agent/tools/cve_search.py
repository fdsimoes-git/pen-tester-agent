"""CVE search tool for vulnerability lookups (CIRCL CVE API, v5 schema)."""

import re
import time
from urllib.parse import quote

import httpx

from .base import Tool, ToolResult

_CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_CIRCL_API_BASE = "https://cve.circl.lu/api"
_REQUEST_TIMEOUT = 15
_MAX_SUMMARY_CHARS = 500
# CVSS metric keys in CVE v5 records, most-preferred first.
_CVSS_KEYS = ("cvssV4_0", "cvssV3_1", "cvssV3_0", "cvssV2_0")
# CIRCL rate-limits bursts; retry transient throttling/unavailability.
_USER_AGENT = "pen-tester-agent/1.0 (+https://github.com/fdsimoes-git/pen-tester-agent)"
_RETRY_STATUS = (429, 503)
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0
_MAX_BACKOFF = 8.0


class CveSearchTool(Tool):
    """Search for known CVEs by ID or by vendor/product."""

    name = "cve_search"
    description = (
        "Search for known CVEs (Common Vulnerabilities and Exposures). "
        "Pass a CVE ID (e.g. 'CVE-2021-44228') for a direct lookup, or a "
        "'vendor product' pair (e.g. 'apache http_server', 'openbsd openssh') "
        "to list CVEs affecting that product. Free-text version strings are "
        "NOT supported by the search backend — give the vendor and product "
        "names, not a version (look up specific versions by reading each CVE)."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": (
                "A CVE ID (e.g. 'CVE-2021-44228'), or a 'vendor product' "
                "pair (e.g. 'apache http_server')."
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
        return self._search_product(query, max_results)

    def _lookup_cve_id(self, cve_id: str) -> ToolResult:
        """Look up a specific CVE by its ID."""
        url = f"{_CIRCL_API_BASE}/cve/{cve_id}"
        try:
            resp = _get(url)
            if resp.status_code == 404:
                return ToolResult(output=f"CVE {cve_id} not found.", success=True)
            resp.raise_for_status()

            data = resp.json()
            if not data:
                return ToolResult(output=f"CVE {cve_id} not found.", success=True)
            return ToolResult(output=_format_cve(data))
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error querying CVE API: {exc}", success=False)

    def _search_product(self, query: str, max_results: int) -> ToolResult:
        """Search CVEs by vendor/product via CIRCL's /search/{vendor}/{product}."""
        parts = query.split()
        if len(parts) < 2:
            return ToolResult(
                output=(
                    f"Product search needs a 'vendor product' pair "
                    f"(e.g. 'apache http_server'); got '{query}'. For a single "
                    "CVE, pass its ID instead."
                ),
                success=False,
            )

        vendor = quote(parts[0].lower(), safe="")
        product = quote("_".join(parts[1:]).lower(), safe="")
        url = f"{_CIRCL_API_BASE}/search/{vendor}/{product}"
        try:
            resp = _get(url)
            if resp.status_code == 404:
                return ToolResult(
                    output=f"No CVEs found for '{query}'.", success=True,
                )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error querying CVE API: {exc}", success=False)

        records = _extract_search_records(payload)
        if not records:
            return ToolResult(output=f"No CVEs found for '{query}'.", success=True)
        return ToolResult(
            output=_format_search_output(query, payload, records, max_results)
        )


def _format_search_output(query, payload, records, max_results) -> str:
    """Render a product-search result list."""
    total = payload.get("total_count", len(records)) \
        if isinstance(payload, dict) else len(records)
    shown = min(max_results, len(records))
    lines = [f"Found {total} CVE(s) for '{query}' (showing {shown}):\n"]
    for record in records[:max_results]:
        lines.append(_format_cve(record))
        lines.append("")
    return "\n".join(lines)


def _get(url: str) -> httpx.Response:
    """GET with a descriptive User-Agent and backoff on 429/503.

    CIRCL throttles bursts, so transient 429/503 responses are retried
    (honoring Retry-After when present) to keep a session's CVE lookups
    from failing spuriously. Connection errors and other statuses are
    returned/raised to the caller unchanged.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=_REQUEST_TIMEOUT, headers=headers) as client:
        for attempt in range(_MAX_RETRIES):
            resp = client.get(url)
            if resp.status_code not in _RETRY_STATUS:
                return resp
            time.sleep(_retry_delay(resp.headers.get("Retry-After"), attempt))
        return client.get(url)


def _retry_delay(retry_after, attempt: int) -> float:
    """Seconds before the next retry: honor Retry-After, else exp backoff."""
    if retry_after:
        try:
            return min(float(retry_after), _MAX_BACKOFF)
        except (TypeError, ValueError):
            pass
    return min(_BACKOFF_BASE * (2 ** attempt), _MAX_BACKOFF)


def _extract_search_records(payload) -> list:
    """Pull CVE v5 records out of CIRCL's search envelope.

    Shape: {"results": {"<source>": [[cve_id, {record}], ...]}, ...}.
    Each item is a ``[cve_id, record]`` pair (older CIRCL builds returned the
    record directly), so both forms are handled.
    """
    if not isinstance(payload, dict):
        return []
    results = payload.get("results", {})
    records = []
    if isinstance(results, dict):
        sources = results.values()
    elif isinstance(results, list):
        sources = [results]
    else:
        return []

    for source_items in sources:
        if not isinstance(source_items, list):
            continue
        for item in source_items:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                records.append(item[1])
            elif isinstance(item, dict):
                records.append(item)
    return records


def _format_cve(data: dict) -> str:
    """Format a single CVE v5 record for display."""
    if not isinstance(data, dict):
        return "  Unknown\n  CVSS: N/A\n  No description available"

    meta = data.get("cveMetadata", {})
    containers = data.get("containers", {})
    cna = containers.get("cna", {})

    cve_id = meta.get("cveId") or data.get("id") or "Unknown"

    summary = _first_english_description(cna.get("descriptions", [])) \
        or "No description available"
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "..."

    cvss = _extract_cvss(containers)

    refs = [r.get("url") for r in cna.get("references", []) if r.get("url")]
    ref_str = ""
    if refs:
        ref_str = "\n  References: " + ", ".join(refs[:3])

    return f"  {cve_id}\n  CVSS: {cvss}\n  {summary}{ref_str}"


def _first_english_description(descriptions) -> str:
    """Return the first English (or any) description value."""
    if not isinstance(descriptions, list):
        return ""
    for desc in descriptions:
        if isinstance(desc, dict) and str(desc.get("lang", "en")).lower().startswith("en"):
            return desc.get("value", "")
    for desc in descriptions:
        if isinstance(desc, dict) and desc.get("value"):
            return desc["value"]
    return ""


def _extract_cvss(containers: dict) -> str:
    """Find a CVSS base score across cna + adp metrics (preferred version first).

    In CVE v5 the score often lives in the ADP (CISA/authorized publisher)
    container rather than the CNA, so both are scanned.
    """
    metric_blocks = list(containers.get("cna", {}).get("metrics", []) or [])
    for adp in containers.get("adp", []) or []:
        metric_blocks.extend(adp.get("metrics", []) or [])

    for key in _CVSS_KEYS:
        for block in metric_blocks:
            if isinstance(block, dict) and isinstance(block.get(key), dict):
                score = block[key].get("baseScore")
                severity = block[key].get("baseSeverity")
                if score is not None:
                    return f"{score} ({severity})" if severity else str(score)
    return "N/A"
