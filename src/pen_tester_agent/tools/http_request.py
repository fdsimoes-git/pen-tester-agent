"""HTTP request tool."""

import httpx

from .base import Tool, ToolResult

MAX_RESPONSE_BODY_CHARS = 5000


class HttpRequestTool(Tool):
    """Make HTTP requests to URLs."""

    name = "http_request"
    description = (
        "Send an HTTP request and return the response. Useful for probing "
        "web applications, testing APIs for vulnerabilities, checking "
        "HTTP security headers, and fingerprinting web services."
    )
    parameters = {
        "method": {
            "type": "string",
            "description": "HTTP method: GET, POST, PUT, DELETE, HEAD, OPTIONS",
            "default": "GET",
        },
        "url": {
            "type": "string",
            "description": "The URL to request",
        },
        "headers": {
            "type": "object",
            "description": "Optional HTTP headers as key-value pairs",
            "default": {},
        },
        "body": {
            "type": "string",
            "description": "Optional request body",
            "default": "",
        },
        "timeout": {
            "type": "integer",
            "description": "Request timeout in seconds (default: 30)",
            "default": 30,
        },
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url", "")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", "")
        timeout = kwargs.get("timeout", 30)

        if not url:
            return ToolResult(output="Error: no url provided", success=False)

        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body if body else None,
                )

            lines = [
                f"HTTP {response.status_code} {response.reason_phrase}",
                "",
                "--- Response Headers ---",
            ]
            for key, value in response.headers.items():
                lines.append(f"{key}: {value}")

            lines.append("")
            lines.append("--- Response Body ---")

            body_text = response.text
            if len(body_text) > MAX_RESPONSE_BODY_CHARS:
                body_text = (
                    body_text[:MAX_RESPONSE_BODY_CHARS] + "\n... (truncated)"
                )
            lines.append(body_text)

            return ToolResult(output="\n".join(lines))
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error: {exc}", success=False)
