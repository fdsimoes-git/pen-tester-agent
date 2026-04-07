import httpx
import pytest

from pen_tester_agent.tools.cve_search import CveSearchTool


@pytest.fixture
def cve_tool():
    return CveSearchTool()


class TestCveSearchTool:
    def test_metadata(self, cve_tool):
        assert cve_tool.name == "cve_search"
        assert cve_tool.requires_approval is True

    def test_no_query(self, cve_tool):
        r = cve_tool.execute()
        assert r.success is False
        assert "no query" in r.output.lower()

    def test_empty_query(self, cve_tool):
        r = cve_tool.execute(query="")
        assert r.success is False
        assert "no query" in r.output.lower()

    def test_lookup_cve_id(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/cve/CVE-2021-44228",
            json={
                "id": "CVE-2021-44228",
                "summary": "Apache Log4j2 remote code execution",
                "cvss": 10.0,
                "references": ["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"],
            },
        )
        r = cve_tool.execute(query="CVE-2021-44228")
        assert r.success is True
        assert "CVE-2021-44228" in r.output
        assert "10.0" in r.output
        assert "Log4j2" in r.output

    def test_lookup_cve_id_not_found(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/cve/CVE-9999-99999",
            status_code=404,
        )
        r = cve_tool.execute(query="CVE-9999-99999")
        assert r.success is True
        assert "not found" in r.output.lower()

    def test_keyword_search(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/Apache%202.4.49",
            json=[
                {
                    "id": "CVE-2021-41773",
                    "summary": "Path traversal in Apache HTTP Server 2.4.49",
                    "cvss": 7.5,
                    "references": [],
                },
                {
                    "id": "CVE-2021-42013",
                    "summary": "Path traversal and RCE in Apache 2.4.49/2.4.50",
                    "cvss": 9.8,
                    "references": [],
                },
            ],
        )
        r = cve_tool.execute(query="Apache 2.4.49", max_results=5)
        assert r.success is True
        assert "CVE-2021-41773" in r.output
        assert "CVE-2021-42013" in r.output

    def test_keyword_search_no_results(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/nonexistentsoftware12345",
            json=[],
        )
        r = cve_tool.execute(query="nonexistentsoftware12345")
        assert r.success is True
        assert "no cves found" in r.output.lower()

    def test_keyword_search_404(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/notfound",
            status_code=404,
        )
        r = cve_tool.execute(query="notfound")
        assert r.success is True
        assert "no cves found" in r.output.lower()

    def test_max_results_limits_output(self, cve_tool, httpx_mock):
        entries = [
            {"id": f"CVE-2021-{i}", "summary": f"Vuln {i}", "cvss": 5.0}
            for i in range(10)
        ]
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/test",
            json=entries,
        )
        r = cve_tool.execute(query="test", max_results=2)
        assert r.success is True
        assert "CVE-2021-0" in r.output
        assert "CVE-2021-1" in r.output
        assert "CVE-2021-2" not in r.output

    def test_max_results_clamped_to_minimum_one(self, cve_tool, httpx_mock):
        entries = [
            {"id": f"CVE-2021-{i}", "summary": f"Vuln {i}", "cvss": 5.0}
            for i in range(5)
        ]
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/test",
            json=entries,
        )
        r = cve_tool.execute(query="test", max_results=0)
        assert r.success is True
        assert "CVE-2021-0" in r.output  # at least 1 result returned

    def test_connection_error(self, cve_tool, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("connection refused"))
        r = cve_tool.execute(query="CVE-2021-44228")
        assert r.success is False
        assert "error" in r.output.lower()

    def test_cve_id_case_insensitive(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/cve/CVE-2021-44228",
            json={
                "id": "CVE-2021-44228",
                "summary": "Log4Shell",
                "cvss": 10.0,
            },
        )
        r = cve_tool.execute(query="cve-2021-44228")
        assert r.success is True
        assert "CVE-2021-44228" in r.output
