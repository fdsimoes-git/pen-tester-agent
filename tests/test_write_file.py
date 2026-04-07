import pytest

from pen_tester_agent.tools.write_file import WriteFileTool


@pytest.fixture
def write_file():
    return WriteFileTool()


class TestWriteFileTool:
    def test_metadata(self, write_file):
        assert write_file.name == "write_file"
        assert write_file.requires_approval is True

    def test_write_new_file(self, write_file, tmp_path):
        target = tmp_path / "out.txt"
        r = write_file.execute(path=str(target), content="hello world")
        assert r.success is True
        assert target.read_text() == "hello world"
        assert "11 bytes" in r.output

    def test_overwrite_existing(self, write_file, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("old content")
        r = write_file.execute(path=str(target), content="new content")
        assert r.success is True
        assert target.read_text() == "new content"

    def test_creates_parent_dirs(self, write_file, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        r = write_file.execute(path=str(target), content="deep")
        assert r.success is True
        assert target.read_text() == "deep"

    def test_no_path(self, write_file):
        r = write_file.execute(content="something")
        assert r.success is False
        assert "no path" in r.output.lower()

    def test_empty_content(self, write_file, tmp_path):
        target = tmp_path / "empty.txt"
        r = write_file.execute(path=str(target), content="")
        assert r.success is True
        assert target.read_text() == ""
