import pytest

from pen_tester_agent.tools.read_file import ReadFileTool


@pytest.fixture
def read_file():
    return ReadFileTool()


class TestReadFileTool:
    def test_metadata(self, read_file):
        assert read_file.name == "read_file"
        assert read_file.requires_approval is True

    def test_read_existing_file(self, read_file, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        r = read_file.execute(path=str(f))
        assert r.success is True
        assert "line1" in r.output
        assert "line3" in r.output

    def test_read_with_max_lines(self, read_file, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\n")
        r = read_file.execute(path=str(f), max_lines=2)
        assert r.success is True
        assert "line1" in r.output
        assert "line2" in r.output
        assert "line3" not in r.output

    def test_read_nonexistent(self, read_file):
        r = read_file.execute(path="/nonexistent_xyz_file")
        assert r.success is False
        assert "not found" in r.output.lower()

    def test_read_directory(self, read_file, tmp_path):
        r = read_file.execute(path=str(tmp_path))
        assert r.success is False
        assert "not a file" in r.output.lower()

    def test_no_path(self, read_file):
        r = read_file.execute()
        assert r.success is False
        assert "no path" in r.output.lower()

    def test_empty_file(self, read_file, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        r = read_file.execute(path=str(f))
        assert r.success is True
        assert r.output == "(empty file)"

    def test_tilde_expansion(self, read_file):
        # Should not crash on tilde paths, even if file doesn't exist
        r = read_file.execute(path="~/nonexistent_xyz_file")
        assert r.success is False
        assert "not found" in r.output.lower()
