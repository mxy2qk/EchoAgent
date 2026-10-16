from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from echo_agent.dispatcher.client import ProjectAPIError, ProjectClient


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ProjectClientRetryTests(unittest.TestCase):
    def test_transient_urlerror_is_retried(self) -> None:
        client = ProjectClient("http://127.0.0.1:9")
        calls = {"count": 0}

        def fake_urlopen(request, timeout=10):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            if calls["count"] < 3:
                raise URLError(TimeoutError("timed out"))
            return _Response(b'{"ok": true}')

        with patch("echo_agent.dispatcher.client.urlopen", fake_urlopen), patch(
            "echo_agent.dispatcher.client.time.sleep"
        ) as sleep:
            self.assertEqual(client._request("GET", "/projects/1"), {"ok": True})
            self.assertEqual(calls["count"], 3)
            self.assertEqual(sleep.call_count, 2)

    def test_connection_reset_and_address_in_use_are_retried(self) -> None:
        client = ProjectClient("http://127.0.0.1:9")
        reasons = [
            ConnectionResetError("connection reset"),
            OSError(10048, "通常每个套接字地址(协议/网络地址/端口)只允许使用一次"),
        ]

        def fake_urlopen(request, timeout=10):  # type: ignore[no-untyped-def]
            if reasons:
                raise URLError(reasons.pop(0))
            return _Response(json.dumps({"ok": True}).encode("utf-8"))

        with patch("echo_agent.dispatcher.client.urlopen", fake_urlopen), patch(
            "echo_agent.dispatcher.client.time.sleep"
        ):
            self.assertEqual(client._request("GET", "/health"), {"ok": True})
        self.assertEqual(reasons, [])

    def test_http_error_is_not_retried(self) -> None:
        client = ProjectClient("http://127.0.0.1:9")
        calls = {"count": 0}

        def fake_urlopen(request, timeout=10):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            raise HTTPError(
                "http://127.0.0.1:9/projects/1",
                404,
                "missing",
                hdrs=None,
                fp=BytesIO(b'{"error":"missing"}'),
            )

        with patch("echo_agent.dispatcher.client.urlopen", fake_urlopen), patch(
            "echo_agent.dispatcher.client.time.sleep"
        ) as sleep:
            with self.assertRaises(ProjectAPIError):
                client._request("GET", "/projects/1")
        self.assertEqual(calls["count"], 1)
        self.assertEqual(sleep.call_count, 0)

    def test_non_transient_urlerror_is_not_retried(self) -> None:
        client = ProjectClient("http://127.0.0.1:9")
        calls = {"count": 0}

        def fake_urlopen(request, timeout=10):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            raise URLError("nodename nor servname provided")

        with patch("echo_agent.dispatcher.client.urlopen", fake_urlopen), patch(
            "echo_agent.dispatcher.client.time.sleep"
        ) as sleep:
            with self.assertRaises(ProjectAPIError):
                client._request("GET", "/projects/1")
        self.assertEqual(calls["count"], 1)
        self.assertEqual(sleep.call_count, 0)


if __name__ == "__main__":
    unittest.main()
