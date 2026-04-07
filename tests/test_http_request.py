import httpx
import pytest

from pen_tester_agent.tools.http_request import HttpRequestTool


@pytest.fixture
def http_tool():
    return HttpRequestTool()


class TestHttpRequestTool:
    def test_metadata(self, http_tool):
        assert http_tool.name == "http_request"
        assert http_tool.requires_approval is True

    def test_no_url(self, http_tool):
        r = http_tool.execute(method="GET")
        assert r.success is False
        assert "no url" in r.output.lower()

    def test_get_request(self, http_tool, httpx_mock):
        httpx_mock.add_response(url="https://example.com/api", text='{"ok": true}')
        r = http_tool.execute(method="GET", url="https://example.com/api")
        assert r.success is True
        assert "200" in r.output
        assert '{"ok": true}' in r.output

    def test_post_request(self, http_tool, httpx_mock):
        httpx_mock.add_response(url="https://example.com/api", text="created", status_code=201)
        r = http_tool.execute(method="POST", url="https://example.com/api", body='{"data": 1}')
        assert r.success is True
        assert "201" in r.output

    def test_default_method_is_get(self, http_tool, httpx_mock):
        httpx_mock.add_response(url="https://example.com/", text="ok")
        r = http_tool.execute(url="https://example.com/")
        assert r.success is True

    def test_custom_headers(self, http_tool, httpx_mock):
        httpx_mock.add_response(url="https://example.com/", text="ok")
        r = http_tool.execute(url="https://example.com/", headers={"Authorization": "Bearer token"})
        assert r.success is True
        req = httpx_mock.get_request()
        assert req.headers["Authorization"] == "Bearer token"

    def test_connection_error(self, http_tool, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("connection refused"))
        r = http_tool.execute(url="https://unreachable.invalid/")
        assert r.success is False
        assert "error" in r.output.lower()

    def test_long_response_truncated(self, http_tool, httpx_mock):
        long_body = "x" * 10000
        httpx_mock.add_response(url="https://example.com/big", text=long_body)
        r = http_tool.execute(url="https://example.com/big")
        assert r.success is True
        assert "truncated" in r.output

    def test_response_headers_shown(self, http_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/",
            text="ok",
            headers={"X-Custom": "value123"},
        )
        r = http_tool.execute(url="https://example.com/")
        assert "x-custom" in r.output.lower()
        assert "value123" in r.output
