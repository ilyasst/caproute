#!/usr/bin/env python3
"""
caproute — Capability-routing LLM gateway.

Accepts OpenAI-compatible requests where the "model" field is a capability
(e.g. "powerful", "fast", "thinking", "vision", "style", "adequate").
Auto-discovers models from hosts, routes by capability with fallback.

Config (~/.config/llm.json):
    {
      "hosts": { "mybox": { "url": "http://mybox:11434", "api": "ollama" } },
      "capabilities": { "powerful": ["qwen2.5:32b"], "fast": ["gemma3:4b"] }
    }

Usage:
    python3 caproute.py                    # port 8800
    CAPROUTE_PORT=9000 python3 caproute.py # custom port
    python3 caproute.py --port 9000        # also works
"""

import argparse
import http.server
import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("CAPROUTE_CONFIG", Path.home() / ".config" / "llm.json"))
DEFAULT_PORT = int(os.environ.get("CAPROUTE_PORT", "8800"))
REQUEST_TIMEOUT = int(os.environ.get("CAPROUTE_TIMEOUT", "300"))
DISCOVERY_INTERVAL = int(os.environ.get("CAPROUTE_DISCOVERY_INTERVAL", "60"))

# ── Discovery cache ──────────────────────────────────────────────────
# Maps model_name -> [{"host": name, "url": base_url, "api": type}, ...]
_discovery = {}
_discovery_lock = threading.Lock()
_last_discovery = 0


def _discover_ollama(host_name, base_url):
    """Query an Ollama host for available models."""
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            models.append(name)
            # Also add without :latest suffix for convenience
            if name.endswith(":latest"):
                models.append(name.removesuffix(":latest"))
    return models


def _discover_openai(host_name, base_url):
    """Query an OpenAI-compatible host for available models."""
    url = base_url.rstrip("/") + "/v1/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def run_discovery():
    """Query all hosts and build model -> hosts map."""
    global _discovery, _last_discovery
    cfg = load_config()
    hosts = _get_hosts(cfg)
    new_map = {}

    for host_name, host_info in hosts.items():
        base_url = host_info["url"]
        api = host_info["api"]
        try:
            if api == "ollama":
                models = _discover_ollama(host_name, base_url)
            else:
                models = _discover_openai(host_name, base_url)

            for model in models:
                if model not in new_map:
                    new_map[model] = []
                new_map[model].append({
                    "host": host_name,
                    "url": base_url,
                    "api": api,
                })
            print(f"[discovery] {host_name}: {len(models)} models")
        except Exception as e:
            print(f"[discovery] {host_name}: UNREACHABLE ({e})")

    with _discovery_lock:
        _discovery = new_map
        _last_discovery = time.time()

    return new_map


def _discovery_loop():
    """Background thread that refreshes discovery every DISCOVERY_INTERVAL seconds."""
    while True:
        try:
            run_discovery()
        except Exception as e:
            print(f"[discovery] error: {e}")
        time.sleep(DISCOVERY_INTERVAL)


def get_discovery():
    """Return current discovery cache."""
    with _discovery_lock:
        return dict(_discovery)


# ── Config ───────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _is_new_format(cfg):
    """Detect config format: new has 'capabilities', old has 'models'."""
    return "capabilities" in cfg


def _get_timeout(cfg, capability):
    """Get timeout for a capability from config, with fallback."""
    timeouts = cfg.get("timeouts", {})
    return timeouts.get(capability, timeouts.get("_default", REQUEST_TIMEOUT))


def _get_hosts(cfg):
    """Get hosts dict, normalizing url key across formats."""
    hosts = {}
    for name, info in cfg.get("hosts", {}).items():
        hosts[name] = {
            "url": info.get("url") or info.get("base_url", ""),
            "api": info["api"],
        }
    return hosts


def get_capabilities():
    """Return sorted list of all capabilities defined in config."""
    cfg = load_config()
    if _is_new_format(cfg):
        return sorted(cfg["capabilities"].keys())
    # Old format: extract from models
    caps = set()
    for info in cfg.get("models", {}).values():
        caps.update(info.get("capabilities", []))
    return sorted(caps)


def _get_capability_models(cfg, capability):
    """Get ordered list of model names for a capability."""
    if _is_new_format(cfg):
        return cfg["capabilities"].get(capability, [])
    # Old format: scan models
    return [name for name, info in cfg.get("models", {}).items()
            if capability in info.get("capabilities", [])]


def _get_all_tagged_models(cfg):
    """Get set of all model names that have at least one capability."""
    if _is_new_format(cfg):
        tagged = set()
        for models in cfg["capabilities"].values():
            tagged.update(models)
        return tagged
    return set(cfg.get("models", {}).keys())


def resolve_capability(capability):
    """Resolve a capability to ordered list of backends.

    Looks up which models have this capability, then checks discovery
    to find which hosts currently serve each model.
    """
    cfg = load_config()
    model_list = _get_capability_models(cfg, capability)
    if not model_list:
        return []

    discovered = get_discovery()
    hosts = _get_hosts(cfg)
    backends = []

    for model_name in model_list:
        # Try discovery first (knows actual host availability)
        disc_hosts = discovered.get(model_name, [])
        if disc_hosts:
            for h in disc_hosts:
                backends.append({
                    "name": model_name,
                    "base_url": h["url"],
                    "api": h["api"],
                    "host": h["host"],
                })
        elif not _is_new_format(cfg):
            # Old format fallback: use explicit hosts from config
            model_info = cfg["models"].get(model_name, {})
            for host_name in model_info.get("hosts", []):
                if host_name in hosts:
                    h = hosts[host_name]
                    backends.append({
                        "name": model_name,
                        "base_url": h["url"],
                        "api": h["api"],
                        "host": host_name,
                    })
    return backends


def resolve_model_direct(model_name):
    """Resolve a direct model name (not a capability) via discovery."""
    discovered = get_discovery()
    hosts = discovered.get(model_name, [])
    if hosts:
        return [{
            "name": model_name,
            "base_url": h["url"],
            "api": h["api"],
            "host": h["host"],
        } for h in hosts]

    # Fallback: check old-format config for explicit host mapping
    cfg = load_config()
    if not _is_new_format(cfg) and model_name in cfg.get("models", {}):
        cfg_hosts = _get_hosts(cfg)
        model_info = cfg["models"][model_name]
        return [{
            "name": model_name,
            "base_url": cfg_hosts[h]["url"],
            "api": cfg_hosts[h]["api"],
            "host": h,
        } for h in model_info.get("hosts", []) if h in cfg_hosts]

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
        elif self.path == "/discovery":
            self._handle_discovery()
        elif self.path == "/":
            self._send_json({"status": "caproute running"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        elif self.path == "/discovery/refresh":
            run_discovery()
            self._send_json({"status": "ok", "models": get_discovery()})
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
            discovered = get_discovery()
            tagged = _get_all_tagged_models(cfg)
            untagged = sorted(set(discovered.keys()) - tagged)

            hosts = _get_hosts(cfg)
            caps = get_capabilities()
            cap_models = {c: _get_capability_models(cfg, c) for c in caps}
            self._send_json({
                "status": "ok",
                "config": str(CONFIG_PATH),
                "capabilities": caps,
                "capability_models": cap_models,
                "hosts": list(hosts.keys()),
                "discovered_models": len(discovered),
                "untagged_models": untagged,
                "last_discovery": _last_discovery,
            })
        except Exception as e:
            self._send_json({"status": "error", "error": str(e)}, 500)

    def _handle_discovery(self):
        """Show full discovery state: which models are on which hosts."""
        discovered = get_discovery()
        cfg = load_config()

        # Build reverse map: model -> capabilities
        if _is_new_format(cfg):
            cap_map = cfg["capabilities"]
        else:
            cap_map = {}
            for model_name, info in cfg.get("models", {}).items():
                for cap in info.get("capabilities", []):
                    cap_map.setdefault(cap, []).append(model_name)

        result = {}
        for model, hosts in sorted(discovered.items()):
            caps = [c for c, models in cap_map.items() if model in models]
            result[model] = {
                "hosts": [h["host"] for h in hosts],
                "capabilities": caps if caps else ["untagged"],
            }
        self._send_json({
            "models": result,
            "last_discovery": _last_discovery,
        })

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

        # Per-capability timeout from config, overridable by request
        cfg = load_config()
        default_timeout = _get_timeout(cfg, model_field)
        timeout = req.get("timeout", default_timeout)

        # Resolve: capability name or direct model name
        backends = resolve_capability(model_field)
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
        hosts = _get_hosts(cfg)
        caps = get_capabilities()
        fmt = "new (capabilities)" if _is_new_format(cfg) else "legacy (models)"
        print(f"[caproute] Config: {CONFIG_PATH} ({fmt})")
        print(f"[caproute] Capabilities: {', '.join(caps)}")
        print(f"[caproute] Hosts: {', '.join(hosts.keys())}")
    except Exception as e:
        print(f"[caproute] ERROR loading config: {e}")
        return 1

    # Initial discovery
    print(f"[caproute] Running initial discovery...")
    discovered = run_discovery()
    print(f"[caproute] Discovered {len(discovered)} models across hosts")

    # Background discovery thread
    t = threading.Thread(target=_discovery_loop, daemon=True)
    t.start()
    print(f"[caproute] Discovery refresh every {DISCOVERY_INTERVAL}s")

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), CaprouteHandler)
    server.daemon_threads = True
    print(f"[caproute] Listening on http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[caproute] Shutting down.")


if __name__ == "__main__":
    main()
