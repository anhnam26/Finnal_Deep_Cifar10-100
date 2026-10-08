import io
import hashlib
import urllib.error

import pytest

import download_data as downloader


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(content))}


def test_resume_with_valid_range(tmp_path, monkeypatch):
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")

    def open_response(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_response)
    updates = []
    assert downloader.transfer("https://example.org/data", partial, updates.append) == 6
    assert partial.read_bytes() == b"abcdef"
    assert updates[-1]["resumed"] is True


def test_server_ignoring_range_restarts_safely(tmp_path, monkeypatch):
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"old")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: Response(b"complete"))
    downloader.transfer("https://example.org/data", partial, lambda state: None)
    assert partial.read_bytes() == b"complete"


def test_invalid_range_does_not_modify_partial(tmp_path, monkeypatch):
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(downloader.urllib.request, "urlopen",
                        lambda *a, **k: Response(b"def", 206, {"Content-Range": "bytes 0-2/6"}))
    with pytest.raises(ValueError, match="Content-Range"):
        downloader.transfer("https://example.org/data", partial, lambda state: None)
    assert partial.read_bytes() == b"abc"


def test_existing_partial_promoted_only_after_checksum(tmp_path, monkeypatch):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"abc")
    monkeypatch.setattr(downloader.urllib.request, "urlopen",
                        lambda *a, **k: Response(b"def", 206, {"Content-Range": "bytes 3-5/6"}))
    updates = []
    downloader.fetch_archive("https://example.org/data", archive, hashlib.md5(b"abcdef").hexdigest(), updates.append)
    assert archive.read_bytes() == b"abcdef"
    assert updates[-1]["phase"] == "archive_verified"
    assert not archive.with_suffix(".gz.part").exists()


def test_bad_checksum_never_published(tmp_path, monkeypatch):
    archive = tmp_path / "archive.tar.gz"
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(ValueError, match="MD5 mismatch"):
        downloader.fetch_archive("https://example.org/data", archive, hashlib.md5(b"good").hexdigest(),
                                 lambda state: None, attempts=1)
    assert not archive.exists()
    assert len(list(tmp_path.glob("*.bad-*"))) == 1


def test_retry_after_network_error(tmp_path, monkeypatch):
    calls = []

    def open_response(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary network issue")
        return Response(b"good")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    archive = tmp_path / "archive.tar.gz"
    downloader.fetch_archive("https://example.org/data", archive, hashlib.md5(b"good").hexdigest(), lambda state: None)
    assert archive.read_bytes() == b"good" and len(calls) == 2


def test_incomplete_response_retains_partial(tmp_path, monkeypatch):
    partial = tmp_path / "archive.part"
    monkeypatch.setattr(downloader.urllib.request, "urlopen",
                        lambda *a, **k: Response(b"abc", 200, {"Content-Length": "6"}))
    with pytest.raises(IOError, match="Incomplete response"):
        downloader.transfer("https://example.org/data", partial, lambda state: None)
    assert partial.read_bytes() == b"abc"