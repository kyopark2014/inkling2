#!/usr/bin/env python3
"""SageMaker adapter for vLLM OpenAI server (Inkling-2-Mini).

Starts `vllm serve` on 127.0.0.1:8000, then exposes SageMaker routes on 0.0.0.0:8080:
  GET  /ping
  POST /invocations  (messages -> /v1/chat/completions, prompt -> /v1/completions)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
SM_PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
MODEL_PATH = (
    os.environ.get("SM_VLLM_MODEL")
    or os.environ.get("VLLM_MODEL")
    or "/opt/ml/model"
)
READY = threading.Event()
VLLM_PROC: subprocess.Popen[str] | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def is_hf_hub_id(value: str) -> bool:
    """True for ids like org/name (not absolute paths or existing local dirs)."""
    if value.startswith("/") or value.startswith("."):
        return False
    if os.path.exists(value):
        return False
    return "/" in value and " " not in value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_vllm_cmd() -> list[str]:
    tp = os.environ.get("SM_VLLM_TENSOR_PARALLEL_SIZE", "1")
    max_model_len = os.environ.get("SM_VLLM_MAX_MODEL_LEN", "65536")
    gpu_util = os.environ.get("SM_VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        tp,
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-auto-tool-choice",
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_util,
        "--trust-remote-code",
    ]
    if env_flag("SM_VLLM_ENABLE_DSPARK", False):
        speculative = os.environ.get(
            "SM_VLLM_SPECULATIVE_CONFIG",
            '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}',
        )
        cmd.extend(["--speculative-config", speculative])
    extra = os.environ.get("SM_VLLM_EXTRA_ARGS", "").strip()
    if extra:
        # shell-like split without importing shlex complexity for quoted JSON already handled above
        cmd.extend(extra.split())
    return cmd


def wait_for_vllm(timeout_s: float = 1700.0) -> None:
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if VLLM_PROC is not None and VLLM_PROC.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {VLLM_PROC.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    READY.set()
                    log("vLLM is ready")
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("vLLM did not become ready before timeout")


def http_json(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{VLLM_HOST}:{VLLM_PORT}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log("%s - %s" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/ping":
            if not READY.is_set():
                self._send(503, b'{"status":"loading"}')
                return
            code, body, ctype = http_json("GET", "/v1/models")
            self._send(200 if 200 <= code < 300 else 503, body if body else b'{"status":"ok"}', ctype)
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/invocations":
            self._send(404, b'{"error":"not found"}')
            return
        if not READY.is_set():
            self._send(503, b'{"error":"model loading"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}')
            return
        if not isinstance(payload, dict):
            self._send(400, b'{"error":"body must be a json object"}')
            return

        # Ensure model id for OpenAI-compatible clients that omit it.
        payload.setdefault("model", os.environ.get("SERVED_MODEL_NAME", MODEL_PATH))

        if "messages" in payload:
            path = "/v1/chat/completions"
        elif "prompt" in payload:
            path = "/v1/completions"
        else:
            self._send(
                400,
                b'{"error":"body must include messages (chat) or prompt (completions)"}',
            )
            return

        code, body, ctype = http_json("POST", path, payload)
        self._send(code, body, ctype)


def start_vllm() -> subprocess.Popen[str]:
    cmd = build_vllm_cmd()
    log("starting vLLM: " + " ".join(cmd))
    env = os.environ.copy()
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, text=True)


def shutdown(*_args: Any) -> None:
    log("shutting down")
    if VLLM_PROC is not None and VLLM_PROC.poll() is None:
        VLLM_PROC.terminate()
        try:
            VLLM_PROC.wait(timeout=30)
        except subprocess.TimeoutExpired:
            VLLM_PROC.kill()
    sys.exit(0)


def main() -> int:
    global VLLM_PROC
    if is_hf_hub_id(MODEL_PATH):
        log(f"loading model from Hugging Face Hub: {MODEL_PATH}")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    elif not os.path.exists(MODEL_PATH):
        log(f"model path missing: {MODEL_PATH}")
        return 1
    else:
        log(f"loading local model from: {MODEL_PATH}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    VLLM_PROC = start_vllm()
    waiter = threading.Thread(target=wait_for_vllm, daemon=True)
    waiter.start()

    server = ThreadingHTTPServer(("0.0.0.0", SM_PORT), Handler)
    log(f"SageMaker server listening on :{SM_PORT}")
    try:
        server.serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
