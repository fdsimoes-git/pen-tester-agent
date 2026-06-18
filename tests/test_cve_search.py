import os

import httpx
import pytest

from pen_tester_agent.tools.cve_search import CveSearchTool


@pytest.fixture
def cve_tool():
    return CveSearchTool()


def v5_record(cve_id, summary, *, score=None, severity="HIGH", refs=None, in_adp=True):
    """Build a minimal CVE Record v5 dict like CIRCL returns."""
    metric = {}
    if score is not None:
        metric = {"cvssV3_1": {"baseScore": score, "baseSeverity": severity}}
    cna = {
        "descriptions": [{"lang": "en", "value": summary}],
        "references": [{"url": u} for u in (refs or [])],
        "metrics": [] if in_adp else ([metric] if metric else []),
    }
    record = {
        "dataType": "CVE_RECORD",
        "dataVersion": "5.1",
        "cveMetadata": {"cveId": cve_id, "state": "PUBLISHED"},
        "containers": {"cna": cna},
    }
    if in_adp and metric:
        record["containers"]["adp"] = [{"metrics": [metric]}]
    return record


def search_envelope(records):
    """Wrap records in CIRCL's {results: {nvd: [[id, record], ...]}} envelope."""
    return {
        "results": {
            "nvd": [[r["cveMetadata"]["cveId"].lower(), r] for r in records]
        },
        "total_count": len(records),
        "page_size": len(records),
        "page": 1,
    }


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
            json=v5_record(
                "CVE-2021-44228",
                "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI remote code execution",
                score=10, severity="CRITICAL",
                refs=["https://logging.apache.org/log4j/2.x/security.html"],
                in_adp=True,
            ),
        )
        r = cve_tool.execute(query="CVE-2021-44228")
        assert r.success is True
        assert "CVE-2021-44228" in r.output
        assert "10" in r.output            # CVSS base score from the ADP container
        assert "CRITICAL" in r.output
        assert "Log4j2" in r.output

    def test_lookup_cve_id_not_found(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/cve/CVE-9999-99999",
            status_code=404,
        )
        r = cve_tool.execute(query="CVE-9999-99999")
        assert r.success is True
        assert "not found" in r.output.lower()

    def test_product_search(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/apache/http_server",
            json=search_envelope([
                v5_record("CVE-2021-41773", "Path traversal in Apache HTTP Server 2.4.49", score=7.5),
                v5_record("CVE-2021-42013", "Path traversal and RCE in Apache 2.4.49/2.4.50", score=9.8),
            ]),
        )
        r = cve_tool.execute(query="apache http_server", max_results=5)
        assert r.success is True
        assert "CVE-2021-41773" in r.output
        assert "CVE-2021-42013" in r.output
        assert "9.8" in r.output

    def test_product_search_joins_multiword_product(self, cve_tool, httpx_mock):
        # "apache http server" -> vendor=apache, product=http_server
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/apache/http_server",
            json=search_envelope([v5_record("CVE-2021-41773", "x", score=7.5)]),
        )
        r = cve_tool.execute(query="apache http server")
        assert r.success is True
        assert "CVE-2021-41773" in r.output

    def test_product_search_requires_vendor_and_product(self, cve_tool):
        r = cve_tool.execute(query="apache")
        assert r.success is False
        assert "vendor product" in r.output.lower()

    def test_product_search_empty_results(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/nope/nothere",
            json=search_envelope([]),
        )
        r = cve_tool.execute(query="nope nothere")
        assert r.success is True
        assert "no cves found" in r.output.lower()

    def test_product_search_404(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/no/thing",
            status_code=404,
        )
        r = cve_tool.execute(query="no thing")
        assert r.success is True
        assert "no cves found" in r.output.lower()

    def test_max_results_limits_output(self, cve_tool, httpx_mock):
        records = [v5_record(f"CVE-2021-{i:05d}", f"Vuln {i}", score=5.0) for i in range(10)]
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/test/product",
            json=search_envelope(records),
        )
        r = cve_tool.execute(query="test product", max_results=2)
        assert r.success is True
        assert "CVE-2021-00000" in r.output
        assert "CVE-2021-00001" in r.output
        assert "CVE-2021-00002" not in r.output

    def test_max_results_clamped_to_minimum_one(self, cve_tool, httpx_mock):
        records = [v5_record(f"CVE-2021-{i:05d}", f"Vuln {i}", score=5.0) for i in range(5)]
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/search/test/product",
            json=search_envelope(records),
        )
        r = cve_tool.execute(query="test product", max_results=0)
        assert r.success is True
        assert "CVE-2021-00000" in r.output

    def test_connection_error(self, cve_tool, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("connection refused"))
        r = cve_tool.execute(query="CVE-2021-44228")
        assert r.success is False
        assert "error" in r.output.lower()

    def test_cve_id_case_insensitive(self, cve_tool, httpx_mock):
        httpx_mock.add_response(
            url="https://cve.circl.lu/api/cve/CVE-2021-44228",
            json=v5_record("CVE-2021-44228", "Log4Shell", score=10, severity="CRITICAL"),
        )
        r = cve_tool.execute(query="cve-2021-44228")
        assert r.success is True
        assert "CVE-2021-44228" in r.output


@pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="hits the live CIRCL API; set RUN_NETWORK_TESTS=1 to run",
)
class TestCveSearchLive:
    """Real-network smoke tests that catch CIRCL API drift (the failure mode
    that the mocked tests above cannot see). Opt-in via RUN_NETWORK_TESTS=1."""

    def test_live_log4shell_lookup(self, cve_tool):
        r = cve_tool.execute(query="CVE-2021-44228")
        assert r.success is True
        assert "CVE-2021-44228" in r.output
        assert "10" in r.output                     # real CVSS base score
        assert "No description available" not in r.output

    def test_live_product_search(self, cve_tool):
        r = cve_tool.execute(query="apache http_server", max_results=3)
        assert r.success is True
        assert "CVE-" in r.output
        assert "No CVEs found" not in r.output
