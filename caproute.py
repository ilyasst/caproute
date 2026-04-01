#!/usr/bin/env python3
"""
caproute — Capability-routing LLM gateway.

Accepts OpenAI-compatible requests where the "model" field is a capability
(e.g. "powerful", "fast", "thinking", "vision", "style", "adequate").
Routes to the best available backend from ~/.config/llm.json with auto-fallback.

Usage:
    python3 caproute.py                    # port 8800
    CAPROUTE_PORT=9000 python3 caproute.py # custom port
    python3 caproute.py --port 9000        # also works
"""

import argparse
import http.server
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("CAPROUTE_CONFIG", Path.home() / ".config" / "llm.json"))
DEFAULT_PORT = int(os.environ.get("CAPROUTE_PORT", "8800"))
REQUEST_TIMEOUT = int(os.environ.get("CAPROUTE_TIMEOUT", "300"))

# ── Config ───────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_capabilities():
    """Return set of all capabilities defined in config."""
    cfg = load_config()
    caps = set()
    for info in cfg["models"].values():
        caps.update(info["capabilities"])
    return sorted(caps)


def get_models_for_capability(capability):
    """Get ordered list of (model_name, host_info) for a capability."""
    cfg = load_config()
    results = []
    for model_name, info in cfg["models"].items():
        if capability in info["capabilities"]:
            for host_name in info["hosts"]:
                host = cfg["hosts"][host_name]
                results.append({
                    "name": model_name,
                    "base_url": host["base_url"],
                    "api": host["api"],
                    "host": host_name,
                })
    return results


def resolve_model_direct(model_name):
    """If model_name is a real model (not a capability), resolve its host."""
    cfg = load_config()
    if model_name in cfg["models"]:
        info = cfg["models"][model_name]
        for host_name in info["hosts"]:
            host = cfg["hosts"][host_name]
            return [{
                "name": model_name,
                "base_url": host["base_url"],
                "api": host["api"],
                "host": host_name,
            }]
    return []


# ── Backend requests ─────────────────────────────────────────────────

def _proxy_ollama(base_url, model, messages, params, timeout):
    """Send request to an Ollama backend, return OpenAI-shaped response."""
    url = base_url.rstrip("/") + "/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {},
    }
    if params.get("temperature") is not None:
        data["options"]["temperature"] = params["temperature"]
    if params.get("max_tokens") is not None:
        data["options"]["num_predict"] = params["max_tokens"]

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())

    content = result.get("message", {}).get("content", "")
    return _wrap_openai_response(model, content)


def _proxy_openai(base_url, model, messages, params, timeout):
    """Send request to an OpenAI-compatible backend, return response as-is."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if params.get("temperature") is not None:
        data["temperature"] = params["temperature"]
    if params.get("max_tokens") is not None:
        data["max_tokens"] = params["max_tokens"]

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wrap_openai_response(model, content):
    """Wrap raw content into an OpenAI chat completion response."""
    return {
        "id": f"caproute-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def proxy_to_backend(backend, messages, params, timeout):
    """Route a request to a single backend. Returns OpenAI-shaped dict."""
    if backend["api"] == "ollama":
        return _proxy_ollama(backend["base_url"], backend["name"], messages, params, timeout)
    else:
        return _proxy_openai(backend["base_url"], backend["name"], messages, params, timeout)


# ── HTTP handler ─────────────────────────────────────────────────────

class CaprouteHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Compact logging
        print(f"[caproute] {self.address_string()} {args[0] if args else ''}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/":
            self._send_json({"status": "caproute running"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_models(self):
        """List capabilities as models so clients can discover them."""
        caps = get_capabilities()
        models = [{"id": c, "object": "model", "owned_by": "caproute"} for c in caps]
        self._send_json({"object": "list", "data": models})

    def _handle_health(self):
        try:
            cfg = load_config()
            self._send_json({
                "status": "ok",
                "config": str(CONFIG_PATH),
                "capabilities": get_capabilities(),
                "hosts": list(cfg["hosts"].keys()),
            })
        except Exception as e:
            self._send_json({"status": "error", "error": str(e)}, 500)

    def _handle_chat(self):
        body = self._read_body()
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        model_field = req.get("model", "")
        messages = req.get("messages", [])
        if not messages:
            self._send_json({"error": "messages required"}, 400)
            return

        params = {}
        if "temperature" in req:
            params["temperature"] = req["temperature"]
        if "max_tokens" in req:
            params["max_tokens"] = req["max_tokens"]

        timeout = req.get("timeout", REQUEST_TIMEOUT)

        # Resolve: capability name or direct model name
        backends = get_models_for_capability(model_field)
        if not backends:
            backends = resolve_model_direct(model_field)
        if not backends:
            self._send_json({
                "error": f"unknown capability or model: '{model_field}'",
                "available": get_capabilities(),
            }, 400)
            return

        # Try each backend in order
        errors = []
        for backend in backends:
            try:
                print(f"[caproute] {model_field} -> {backend['name']}@{backend['host']}")
                result = proxy_to_backend(backend, messages, params, timeout)
                # Tag which backend actually served it
                result["_caproute"] = {
                    "capability": model_field,
                    "model": backend["name"],
                    "host": backend["host"],
                }
                self._send_json(result)
                return
            except Exception as e:
                err = f"{backend['name']}@{backend['host']}: {e}"
                print(f"[caproute] FAIL {err}")
                errors.append(err)

        self._send_json({
            "error": f"all backends failed for '{model_field}'",
            "tried": errors,
        }, 503)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="caproute — capability-routing LLM gateway")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    parser.add_argument("--config", type=str, default=None, help="path to llm.json config")
    args = parser.parse_args()

    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = Path(args.config)

    # Verify config
    try:
        cfg = load_config()
        caps = get_capabilities()
        print(f"[caproute] Config: {CONFIG_PATH}")
        print(f"[caproute] Capabilities: {', '.join(caps)}")
        print(f"[caproute] Hosts: {', '.join(cfg['hosts'].keys())}")
    except Exception as e:
        print(f"[caproute] ERROR loading config: {e}")
        return 1

    server = http.server.HTTPServer(("0.0.0.0", args.port), CaprouteHandler)
    print(f"[caproute] Listening on http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[caproute] Shutting down.")


if __name__ == "__main__":
    main()
