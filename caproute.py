#!/usr/bin/env python3
"""
caproute — Capability-routing LLM gateway (probe-routing version).

Instead of blindly trying backends sequentially and waiting for timeouts,
caproute continuously probes every backend to know their real-time status:
- Which backends are alive and responsive right now
- What their current latency looks like
- How many requests are in-flight (concurrency tracking)

Routing then goes directly to the best available backend — no wasted timeouts.

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
import concurrent.futures
import http.server
import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("CAPROUTE_CONFIG", Path.home() / ".config" / "llm.json")
)
DEFAULT_PORT = int(os.environ.get("CAPROUTE_PORT", "8800"))
REQUEST_TIMEOUT = int(os.environ.get("CAPROUTE_TIMEOUT", "60"))
DISCOVERY_INTERVAL = int(os.environ.get("CAPROUTE_DISCOVERY_INTERVAL", "60"))
PROBE_INTERVAL = int(os.environ.get("CAPROUTE_PROBE_INTERVAL", "5"))
PROBE_TIMEOUT = int(os.environ.get("CAPROUTE_PROBE_TIMEOUT", "60"))
IDLE_PROBE_INTERVAL = int(os.environ.get("CAPROUTE_IDLE_PROBE_INTERVAL", "60"))
DOWN_RETRY_INTERVAL = int(os.environ.get("CAPROUTE_DOWN_RETRY_INTERVAL", "30"))

# ── Discovery cache ──────────────────────────────────────────────────
# Maps model_name -> [{"host": name, "url": base_url, "api": type}, ...]
_discovery = {}
_discovery_lock = threading.Lock()
_last_discovery = 0

# ── Backend state tracking ───────────────────────────────────────────
# Key: "host:model" (e.g. "peacewalker:qwen2.5:32b")
# Value: {
#   "latency_ms": float,          # recent probe/request latency
#   "failures": int,              # consecutive failures
#   "last_success": float,        # epoch timestamp
#   "last_probe": float,          # epoch timestamp of last probe
#   "in_flight": int,             # currently active requests
#   "avg_latency_ms": float,      # exponential moving average
#   "status": str,                # "ok", "slow", "down", "unknown"
# }
_backend_state = {}
_backend_lock = threading.Lock()

# ── Concurrency tracking ─────────────────────────────────────────────
_in_flight_lock = threading.Lock()
_in_flight = {}  # "host:model" -> int


def _backend_key(host, model):
    return f"{host}:{model}"


def _get_backend_state(key):
    with _backend_lock:
        if key not in _backend_state:
            _backend_state[key] = {
                "latency_ms": 0,
                "failures": 0,
                "last_success": 0,
                "last_probe": 0,
                "in_flight": 0,
                "avg_latency_ms": 500,  # initial estimate
                "status": "unknown",
            }
        return _backend_state[key]


def _record_success(key, latency_ms):
    """Record a successful request or probe."""
    with _backend_lock:
        s = _backend_state.get(
            key,
            {
                "latency_ms": 0,
                "failures": 0,
                "last_success": 0,
                "last_probe": 0,
                "in_flight": 0,
                "avg_latency_ms": 500,
                "status": "unknown",
            },
        )
        # Exponential moving average (alpha=0.3)
        alpha = 0.3
        s["avg_latency_ms"] = alpha * latency_ms + (1 - alpha) * s.get(
            "avg_latency_ms", 500
        )
        s["latency_ms"] = latency_ms
        s["failures"] = 0
        s["last_success"] = time.time()
        s["last_probe"] = time.time()
        # Status based on latency
        if s["avg_latency_ms"] < 2000:
            s["status"] = "ok"
        elif s["avg_latency_ms"] < 10000:
            s["status"] = "slow"
        else:
            s["status"] = "slow"
        _backend_state[key] = s


def _record_failure(key):
    """Record a failed request or probe.
    Uses exponential backoff: backends marked 'down' are retried periodically
    instead of being permanently dead.
    """
    with _backend_lock:
        s = _backend_state.get(
            key,
            {
                "latency_ms": 0,
                "failures": 0,
                "last_success": 0,
                "last_probe": 0,
                "in_flight": 0,
                "avg_latency_ms": 500,
                "status": "unknown",
            },
        )
        s["failures"] = s.get("failures", 0) + 1
        s["last_probe"] = time.time()
        # After 3 failures mark as down, but still retry periodically
        if s["failures"] >= 3:
            s["status"] = "down"
        else:
            s["status"] = "slow"
        _backend_state[key] = s


def _should_retry_down(key):
    """Check if a down backend should be retried (every 30 seconds)."""
    with _backend_lock:
        s = _backend_state.get(key, {})
    if s.get("status") != "down":
        return True
    last_probe = s.get("last_probe", 0)
    return (time.time() - last_probe) >= DOWN_RETRY_INTERVAL


def _record_idle(key):
    """Record that a model is available but not loaded (Ollama idle)."""
    with _backend_lock:
        s = _backend_state.get(
            key,
            {
                "latency_ms": 0,
                "failures": 0,
                "last_success": 0,
                "last_probe": 0,
                "in_flight": 0,
                "avg_latency_ms": 500,
                "status": "unknown",
            },
        )
        s["status"] = "idle"
        s["failures"] = 0
        s["last_probe"] = time.time()
        _backend_state[key] = s


def _inc_in_flight(key):
    with _in_flight_lock:
        _in_flight[key] = _in_flight.get(key, 0) + 1
    with _backend_lock:
        if key in _backend_state:
            _backend_state[key]["in_flight"] = _in_flight.get(key, 0)


def _dec_in_flight(key):
    with _in_flight_lock:
        _in_flight[key] = max(0, _in_flight.get(key, 1) - 1)
    with _backend_lock:
        if key in _backend_state:
            _backend_state[key]["in_flight"] = _in_flight.get(key, 0)


def backend_score(key):
    """
    Score for routing. Lower = better.
    Combines: avg latency, consecutive failures, in-flight count.
    Down backends get a high score but are still tried (not permanently dead).
    """
    with _backend_lock:
        s = _backend_state.get(
            key,
            {
                "avg_latency_ms": 500,
                "failures": 0,
                "in_flight": 0,
                "status": "unknown",
            },
        )

    status = s.get("status", "unknown")

    failures = s.get("failures", 0)
    avg_lat = s.get("avg_latency_ms", 500)
    in_flight = s.get("in_flight", 0)

    # Base score from latency
    score = avg_lat
    # Penalty for consecutive failures
    score += failures * 5000
    # Penalty for in-flight requests (prefer less loaded backends)
    score += in_flight * 3000
    # Penalty for idle (available but not loaded — will need cold-start)
    if status == "idle":
        score += 10000
    # Penalty for down (recently failed — try last but don't skip entirely)
    elif status == "down":
        score += 50000
    # Small penalty for unknown status (prefer backends we've seen succeed)
    elif status == "unknown":
        score += 1000

    return score


# ── Discovery ────────────────────────────────────────────────────────


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
            t0 = time.time()
            if api == "ollama":
                models = _discover_ollama(host_name, base_url)
            else:
                models = _discover_openai(host_name, base_url)
            latency_ms = (time.time() - t0) * 1000

            for model in models:
                if model not in new_map:
                    new_map[model] = []
                new_map[model].append(
                    {
                        "host": host_name,
                        "url": base_url,
                        "api": api,
                    }
                )
            print(f"[discovery] {host_name}: {len(models)} models ({latency_ms:.0f}ms)")
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


# ── Probing ──────────────────────────────────────────────────────────


def _get_loaded_models_ollama(base_url):
    """Get models currently loaded in memory on an Ollama host."""
    try:
        url = base_url.rstrip("/") + "/api/ps"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        loaded = set()
        for m in data.get("models", []):
            name = m.get("name", "")
            loaded.add(name)
            if name.endswith(":latest"):
                loaded.add(name.removesuffix(":latest"))
        return loaded
    except Exception:
        return set()


def _get_loaded_models_openai(base_url):
    """Get models currently loaded on an OpenAI-compatible host (e.g. llama-server).
    Tries /api/ps first (for hosts that implement Ollama-compatible ps endpoint),
    falls back to /v1/models status check, then to assuming all models are loaded.
    Returns (loaded, total) sets."""
    try:
        url = base_url.rstrip("/") + "/api/ps"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        loaded = set()
        for m in data.get("models", []):
            name = m.get("name", "")
            if name:
                loaded.add(name)
        if loaded:
            return loaded, loaded
    except Exception:
        pass

    try:
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        loaded = set()
        total = set()
        has_status = False
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            total.add(mid)
            status = m.get("status", {})
            if isinstance(status, dict) and "value" in status:
                has_status = True
                if status["value"] == "loaded":
                    loaded.add(mid)
            else:
                loaded.add(mid)
        if not has_status:
            return total, total
        return loaded, total
    except Exception:
        return set(), set()


def _probe_backend(backend):
    """Send a tiny probe request to a backend. Returns latency_ms or None on failure."""
    key = _backend_key(backend["host"], backend["name"])
    try:
        t0 = time.time()
        if backend["api"] == "ollama":
            url = backend["base_url"].rstrip("/") + "/api/chat"
            data = {
                "model": backend["name"],
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 5},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                json.loads(resp.read())
        else:
            url = backend["base_url"].rstrip("/") + "/v1/chat/completions"
            data = {
                "model": backend["name"],
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 5,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                json.loads(resp.read())

        latency_ms = (time.time() - t0) * 1000
        _record_success(key, latency_ms)
        return latency_ms
    except Exception as e:
        _record_failure(key)
        return None


def _probe_loop():
    """
    Continuously probe backends to track real-time health.

    Only probe models that are loaded in memory (via /api/ps for Ollama).
    Idle (not loaded) models are skipped — actual requests will trigger
    cold-start loading if needed.

    Down backends are retried periodically with exponential backoff.

    Skips backends with in-flight requests to avoid interference.
    """
    while True:
        try:
            cfg = load_config()
            hosts = _get_hosts(cfg)
            discovered = get_discovery()
            tagged_models = set()
            for models in cfg.get("capabilities", {}).values():
                tagged_models.update(models)
            if "models" in cfg:
                for model_name, info in cfg.get("models", {}).items():
                    if info.get("capabilities"):
                        tagged_models.add(model_name)

            # Pre-fetch loaded model sets per host (avoid repeated API calls)
            _host_loaded_cache = {}

            backends_to_probe = []
            for model_name in tagged_models:
                disc_hosts = discovered.get(model_name, [])
                for h in disc_hosts:
                    key = _backend_key(h["host"], model_name)
                    state = _get_backend_state(key)
                    in_flight = state.get("in_flight", 0)
                    last_probe = state.get("last_probe", 0)
                    age = time.time() - last_probe
                    status = state.get("status", "unknown")

                    if in_flight > 0:
                        continue

                    # Down backends: retry with exponential backoff
                    if status == "down" and not _should_retry_down(key):
                        continue

                    # Skip recently probed healthy backends
                    if status in ("ok", "slow") and age < PROBE_INTERVAL:
                        continue

                    # Check if model is loaded
                    host_key = (h["host"], h["api"], h["url"])
                    if host_key not in _host_loaded_cache:
                        if h["api"] == "ollama":
                            loaded = _get_loaded_models_ollama(h["url"])
                            _host_loaded_cache[host_key] = loaded
                        else:
                            loaded, _ = _get_loaded_models_openai(h["url"])
                            _host_loaded_cache[host_key] = loaded

                    loaded = _host_loaded_cache[host_key]
                    if model_name not in loaded:
                        # Model not loaded — mark as idle, skip probing
                        # Actual requests will handle cold-start
                        _record_idle(key)
                        continue

                    backends_to_probe.append(
                        {
                            "name": model_name,
                            "base_url": h["url"],
                            "api": h["api"],
                            "host": h["host"],
                        }
                    )

            for backend in backends_to_probe:
                latency = _probe_backend(backend)
                key = _backend_key(backend["host"], backend["name"])
                state = _get_backend_state(key)
                status = state.get("status", "unknown")
                if latency:
                    print(
                        f"[probe] {backend['name']}@{backend['host']}: ok ({latency:.0f}ms)"
                    )
                else:
                    print(f"[probe] {backend['name']}@{backend['host']}: {status}")

        except Exception as e:
            print(f"[probe] error: {e}")

        time.sleep(PROBE_INTERVAL)


# ── Config ───────────────────────────────────────────────────────────


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _is_new_format(cfg):
    return "capabilities" in cfg


def _get_timeout(cfg, capability):
    timeouts = cfg.get("timeouts", {})
    return timeouts.get(capability, timeouts.get("_default", REQUEST_TIMEOUT))


def _get_hosts(cfg):
    hosts = {}
    for name, info in cfg.get("hosts", {}).items():
        hosts[name] = {
            "url": info.get("url") or info.get("base_url", ""),
            "api": info["api"],
        }
    return hosts


# Default fallback chain: each capability falls back to the one above it (higher quality).
# Config can override with a "fallbacks" key.
DEFAULT_FALLBACKS = {
    "fast": "adequate",
    "adequate": "powerful",
    "powerful": "thinking",
    "style": "adequate",
}


def get_capabilities():
    cfg = load_config()
    if _is_new_format(cfg):
        return sorted(cfg["capabilities"].keys())
    caps = set()
    for info in cfg.get("models", {}).values():
        caps.update(info.get("capabilities", []))
    return sorted(caps)


def _get_fallbacks(cfg):
    """Get fallback chain from config, merged with defaults."""
    fallbacks = dict(DEFAULT_FALLBACKS)
    fallbacks.update(cfg.get("fallbacks", {}))
    return fallbacks


def _get_capability_models(cfg, capability):
    if _is_new_format(cfg):
        return cfg["capabilities"].get(capability, [])
    return [
        name
        for name, info in cfg.get("models", {}).items()
        if capability in info.get("capabilities", [])
    ]


def _get_all_tagged_models(cfg):
    if _is_new_format(cfg):
        tagged = set()
        for models in cfg["capabilities"].values():
            tagged.update(models)
        return tagged
    return set(cfg.get("models", {}).keys())


def _resolve_capability_backends(capability):
    """Resolve a capability to backends sorted by probe score. Returns empty list if none available."""
    cfg = load_config()
    model_list = _get_capability_models(cfg, capability)
    if not model_list:
        return []

    discovered = get_discovery()
    hosts = _get_hosts(cfg)
    backends = []

    for model_name in model_list:
        disc_hosts = discovered.get(model_name, [])
        disc_by_host = {h["host"]: h for h in disc_hosts}

        if not _is_new_format(cfg):
            model_info = cfg["models"].get(model_name, {})
            config_hosts = model_info.get("hosts", [])
        else:
            config_hosts = list(disc_by_host.keys())

        for host_name in config_hosts:
            if host_name in disc_by_host:
                h = disc_by_host[host_name]
            elif host_name in hosts:
                h = hosts[host_name]
            else:
                continue

            key = _backend_key(host_name, model_name)
            score = backend_score(key)
            backends.append(
                {
                    "name": model_name,
                    "base_url": h["url"],
                    "api": h["api"],
                    "host": host_name,
                    "_score": score,
                }
            )

    backends.sort(key=lambda b: b["_score"])
    return backends


def _has_healthy_backend(backends):
    """Check if any backend in the list is not down (score < 999999)."""
    return any(b["_score"] < 999999 for b in backends)


def resolve_capability(capability):
    """
    Resolve a capability to backends sorted by probe score.
    If all backends are down, follows the fallback chain upward to higher-quality
    categories until a healthy backend is found.
    Returns (backends, actual_capability) tuple.
    """
    backends = _resolve_capability_backends(capability)
    actual_cap = capability

    if backends and not _has_healthy_backend(backends):
        # All backends for this capability are down — try fallback chain
        cfg = load_config()
        fallbacks = _get_fallbacks(cfg)
        chain = []
        current = capability
        visited = set()
        while current in fallbacks and current not in visited:
            visited.add(current)
            next_cap = fallbacks[current]
            chain.append(next_cap)
            current = next_cap

        for fallback_cap in chain:
            fb_backends = _resolve_capability_backends(fallback_cap)
            if fb_backends and _has_healthy_backend(fb_backends):
                backends = fb_backends
                actual_cap = fallback_cap
                print(f"[caproute] fallback: {capability} -> {actual_cap}")
                break

    return backends, actual_cap


def resolve_model_direct(model_name):
    """Resolve a direct model name (not a capability) via discovery."""
    discovered = get_discovery()
    hosts = discovered.get(model_name, [])
    if hosts:
        result = []
        for h in hosts:
            key = _backend_key(h["host"], model_name)
            result.append(
                {
                    "name": model_name,
                    "base_url": h["url"],
                    "api": h["api"],
                    "host": h["host"],
                    "_score": backend_score(key),
                }
            )
        result.sort(key=lambda b: b["_score"])
        return result, model_name

    cfg = load_config()
    if not _is_new_format(cfg) and model_name in cfg.get("models", {}):
        cfg_hosts = _get_hosts(cfg)
        model_info = cfg["models"][model_name]
        result = []
        for h in model_info.get("hosts", []):
            if h in cfg_hosts:
                key = _backend_key(h, model_name)
                result.append(
                    {
                        "name": model_name,
                        "base_url": cfg_hosts[h]["url"],
                        "api": cfg_hosts[h]["api"],
                        "host": h,
                        "_score": backend_score(key),
                    }
                )
        result.sort(key=lambda b: b["_score"])
        return result, model_name

    return [], model_name


# ── Backend requests ─────────────────────────────────────────────────


def _extract_content(result):
    """Extract text content from a chat completion response.
    Handles reasoning models that output to 'reasoning_content' (OpenAI format)
    or 'reasoning' (Ollama format) instead of 'content'.
    """
    # DEBUG
    print(f"[DEBUG _extract_content] result keys: {result.keys()}")
    # OpenAI format
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        print(f"[DEBUG _extract_content] msg keys: {msg.keys()}")
        content = msg.get("content", "")
        if content:
            return content
        # Fallback: reasoning models put output in reasoning_content (OpenAI),
        # reasoning (OpenAI-compat Ollama), or thinking (native Ollama)
        for key in ("reasoning_content", "reasoning", "thinking"):
            reasoning = msg.get(key, "")
            if reasoning:
                print(
                    f"[DEBUG _extract_content] found reasoning via key={key}, len={len(reasoning)}"
                )
                return reasoning
    # Ollama format (wrapped by _proxy_ollama)
    msg = result.get("message", {})
    if msg:
        content = msg.get("content", "")
        if content:
            return content
        for key in ("reasoning_content", "reasoning", "thinking"):
            reasoning = msg.get(key, "")
            if reasoning:
                return reasoning
    return ""


def _proxy_ollama(base_url, model, messages, params, timeout):
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

    content = _extract_content(result)
    return _wrap_openai_response(model, content)


def _proxy_openai(base_url, model, messages, params, timeout):
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
        result = json.loads(resp.read())

    content = _extract_content(result)
    return _wrap_openai_response(model, content)


def _wrap_openai_response(model, content):
    return {
        "id": f"caproute-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def proxy_to_backend(backend, messages, params, timeout):
    if backend["api"] == "ollama":
        return _proxy_ollama(
            backend["base_url"], backend["name"], messages, params, timeout
        )
    else:
        return _proxy_openai(
            backend["base_url"], backend["name"], messages, params, timeout
        )


# ── HTTP handler ─────────────────────────────────────────────────────


class CaprouteHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected, nothing to do

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
        elif self.path == "/backends":
            self._handle_backends()
        elif self.path == "/":
            self._send_json({"status": "caproute running", "mode": "probe-routing"})
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
            self._send_json(
                {
                    "status": "ok",
                    "config": str(CONFIG_PATH),
                    "capabilities": caps,
                    "capability_models": cap_models,
                    "hosts": list(hosts.keys()),
                    "discovered_models": len(discovered),
                    "untagged_models": untagged,
                    "last_discovery": _last_discovery,
                    "host_health": {
                        k: {
                            "latency_ms": round(v.get("latency_ms", 0)),
                            "failures": v.get("failures", 0),
                        }
                        for k, v in _backend_state.items()
                    },
                }
            )
        except Exception as e:
            self._send_json({"status": "error", "error": str(e)}, 500)

    def _handle_discovery(self):
        discovered = get_discovery()
        cfg = load_config()

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
        self._send_json(
            {
                "models": result,
                "last_discovery": _last_discovery,
            }
        )

    def _handle_backends(self):
        """Show real-time backend state: scores, status, in-flight, latency."""
        with _backend_lock:
            state = dict(_backend_state)
        with _in_flight_lock:
            in_flight = dict(_in_flight)

        result = {}
        for key, s in sorted(state.items()):
            result[key] = {
                "status": s.get("status", "unknown"),
                "score": round(backend_score(key)),
                "avg_latency_ms": round(s.get("avg_latency_ms", 0)),
                "latency_ms": round(s.get("latency_ms", 0)),
                "failures": s.get("failures", 0),
                "in_flight": in_flight.get(key, s.get("in_flight", 0)),
                "last_success": s.get("last_success", 0),
                "last_probe": s.get("last_probe", 0),
            }
        self._send_json({"backends": result})

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

        cfg = load_config()
        default_timeout = _get_timeout(cfg, model_field)
        timeout = req.get("timeout", default_timeout)

        # Resolve: capability name or direct model name
        backends, actual_cap = resolve_capability(model_field)
        if not backends:
            backends, _ = resolve_model_direct(model_field)
        if not backends:
            self._send_json(
                {
                    "error": f"unknown capability or model: '{model_field}'",
                    "available": get_capabilities(),
                },
                400,
            )
            return

        # Try each backend in score order (best first).
        # Use the full timeout per backend since we already know which ones are alive
        # from probing — no need to split timeout across backends.
        errors = []
        for backend in backends:
            key = _backend_key(backend["host"], backend["name"])
            _inc_in_flight(key)
            try:
                t0 = time.time()
                print(
                    f"[caproute] {model_field} -> {backend['name']}@{backend['host']} (score={backend['_score']:.0f})"
                )
                result = proxy_to_backend(backend, messages, params, timeout)
                latency_ms = (time.time() - t0) * 1000
                _record_success(key, latency_ms)
                result["_caproute"] = {
                    "capability": actual_cap,
                    "requested": model_field,
                    "model": backend["name"],
                    "host": backend["host"],
                    "latency_ms": round(latency_ms),
                    "score": round(backend["_score"]),
                }
                self._send_json(result)
                return
            except Exception as e:
                _record_failure(key)
                err = f"{backend['name']}@{backend['host']}: {e}"
                print(f"[caproute] FAIL {err}")
                errors.append(err)
            finally:
                _dec_in_flight(key)

        self._send_json(
            {
                "error": f"all backends failed for '{model_field}'",
                "tried": errors,
            },
            503,
        )


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="caproute — capability-routing LLM gateway (probe-routing)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port to listen on"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="path to llm.json config"
    )
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

    # Background probe thread — continuously checks backend health
    p = threading.Thread(target=_probe_loop, daemon=True)
    p.start()
    print(f"[caproute] Probe interval: {PROBE_INTERVAL}s, timeout: {PROBE_TIMEOUT}s")

    class PooledHTTPServer(http.server.HTTPServer):
        """HTTPServer that dispatches requests to a bounded thread pool."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        daemon_threads = True

        def process_request(self, request, client_address):
            self.pool.submit(self.process_request_thread, request, client_address)

        def process_request_thread(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

        def handle_error(self, request, client_address):
            # Suppress noisy broken pipe / connection reset errors
            import traceback, sys
            _, exc, _ = sys.exc_info()
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, OSError)):
                return
            traceback.print_exc()

    server = PooledHTTPServer(("0.0.0.0", args.port), CaprouteHandler)
    print(f"[caproute] Listening on http://0.0.0.0:{args.port} (probe-routing, 32 workers)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[caproute] Shutting down.")


if __name__ == "__main__":
    main()
