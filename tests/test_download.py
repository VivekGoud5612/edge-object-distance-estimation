"""Resumable downloader against a local HTTP server (no internet needed)."""

import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.data.coco import _is_complete_zip
from src.data.download import download_file

PAYLOAD = bytes(range(256)) * 4000   # ~1 MB


class Handler(BaseHTTPRequestHandler):
    support_range = True
    stall = False

    def do_GET(self):
        if self.stall:
            time.sleep(3)
        start = 0
        rng = self.headers.get("Range")
        if rng and self.support_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    def start(**attrs):
        handler = type("H", (Handler,), attrs)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/file.bin"
    servers = []
    yield start
    for s in servers:
        s.shutdown()


def test_fresh_download(server, tmp_path):
    out = download_file(server(), tmp_path / "f.bin", progress=False)
    assert out.read_bytes() == PAYLOAD
    assert not (tmp_path / "f.bin.part").exists()


def test_resume_from_part(server, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:1000])
    download_file(server(), tmp_path / "f.bin", progress=False)
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_server_without_range_restarts(server, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(b"garbage" * 10)
    download_file(server(support_range=False), tmp_path / "f.bin", progress=False)
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_stalled_server_times_out(server, tmp_path):
    with pytest.raises(RuntimeError):
        download_file(server(stall=True), tmp_path / "f.bin", timeout=0.5, retries=1, progress=False)


def test_truncated_zip_detected(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("annotations/instances_val2017.json", "{}" * 1000)
    assert _is_complete_zip(z)
    z.write_bytes(z.read_bytes()[:50])   # simulate an interrupted download
    assert not _is_complete_zip(z)
