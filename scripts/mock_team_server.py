#!/usr/bin/env python3
"""A stand-in for a team's submitted container, with switchable pathologies.

The harness-to-container contract is where day-of failures will actually come
from, and nothing else in the test suite exercises it. This serves the same
OpenAI-compatible surface a real submission must (`POST /v1/chat/completions`,
`GET /v1/models`) and can be told to misbehave in the specific ways a real
submission plausibly will. No GPU needed: the mock replaces the model, so the
thing under test is the contract, not the answers.

    python scripts/mock_team_server.py --port 8099 --mode normal
    uv run coeval datasets=conquer_val num_samples=10 \\
        datasets/metrics/judge@conquer_judge=deepseek-v4-pro \\
        client.llm.config.api_base=http://localhost:8099/v1 \\
        client.llm.config.model=team-chatbot

Modes:
    normal     a plausible long answer
    slow       sleeps past any sane per-request timeout
    error500   HTTP 500
    empty      HTTP 200 carrying no content -- the failure that must score 0,
               not vanish from the denominator
    badshape   HTTP 200 with a non-OpenAI body
    flaky      1 in 4 requests fails with HTTP 503
    serial     correct answers, but one request at a time -- the pathology that
               silently turns a 40-minute run into hours
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

ANSWER = (
    "Thanks for sharing those details. Based on what you describe, here is how I "
    "would think about it.\n\n"
    "**When to seek care now.** If you develop severe or sudden symptoms, "
    "difficulty breathing, chest pain, confusion, or symptoms that are rapidly "
    "worsening, seek urgent in-person assessment rather than waiting.\n\n"
    "**What this could relate to.** Several common causes could produce this "
    "picture, and distinguishing them usually depends on details such as timing, "
    "associated symptoms, your medical history, and any medications you take.\n\n"
    "**Reasonable next steps.** Tracking when symptoms occur and what makes them "
    "better or worse is genuinely useful to a clinician. If symptoms persist "
    "beyond a few days, or interfere with sleep or daily activity, arrange a "
    "review with your regular doctor.\n\n"
    "**Important caveat.** This is general information and not a diagnosis or a "
    "substitute for individual medical advice. A clinician who can examine you "
    "and review your history is best placed to advise on your specific situation."
)

_state = {"mode": "normal", "requests": 0, "concurrent": 0, "peak_concurrent": 0}
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # keep the run output readable
        pass

    def _send(self, code: int, payload: dict | None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "team-chatbot", "object": "model", "owned_by": "team"}
                    ],
                },
            )
        elif self.path.rstrip("/").endswith("/stats"):
            self._send(200, dict(_state))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)

        with _lock:
            _state["requests"] += 1
            _state["concurrent"] += 1
            _state["peak_concurrent"] = max(
                _state["peak_concurrent"], _state["concurrent"]
            )
            n = _state["requests"]
        try:
            mode = _state["mode"]
            if mode == "slow":
                time.sleep(400)
            elif mode == "error500":
                self._send(500, {"error": "internal team error"})
                return
            elif mode == "flaky" and n % 4 == 0:
                self._send(503, {"error": "overloaded"})
                return
            elif mode == "badshape":
                self._send(200, {"answer": ANSWER})  # not an OpenAI body
                return
            elif mode == "empty":
                self._send(
                    200,
                    {
                        "id": "mock",
                        "object": "chat.completion",
                        "model": "team-chatbot",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": None},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 0,
                            "total_tokens": 100,
                        },
                    },
                )
                return

            if mode == "serial":
                time.sleep(0.4)  # make the serialisation visible in wall clock
            self._send(
                200,
                {
                    "id": f"mock-{n}",
                    "object": "chat.completion",
                    "model": "team-chatbot",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ANSWER},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": len(ANSWER) // 4,
                        "total_tokens": 100 + len(ANSWER) // 4,
                    },
                },
            )
        finally:
            with _lock:
                _state["concurrent"] -= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument(
        "--mode",
        choices=[
            "normal",
            "slow",
            "error500",
            "empty",
            "badshape",
            "flaky",
            "serial",
        ],
        default="normal",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    _state["mode"] = args.mode
    # 'serial' is the pathology of a container that cannot handle parallel
    # requests, so it must genuinely serialise rather than merely sleep.
    server_cls = HTTPServer if args.mode == "serial" else ThreadingHTTPServer
    srv = server_cls(("0.0.0.0", args.port), Handler)
    print(f"mock team container: mode={args.mode} port={args.port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()


if __name__ == "__main__":
    main()
