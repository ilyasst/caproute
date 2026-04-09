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
import collections
import concurrent.futures
import http.server
import json
import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("CAPROUTE_CONFIG", Path.home() / ".config" / "llm.json")
)
DEFAULT_PORT = int(os.environ.get("CAPROUTE_PORT", "8800"))
REQUEST_TIMEOUT = int(os.environ.get("CAPROUTE_TIMEOUT", "60"))
DISCOVERY_INTERVAL = int(os.environ.get("CAPROUTE_DISCOVERY_INTERVAL", "60"))
PROBE_INTERVAL = int(os.environ.get("CAPROUTE_PROBE_INTERVAL", "120"))
PROBE_TIMEOUT = int(os.environ.get("CAPROUTE_PROBE_TIMEOUT", "10"))
IDLE_PROBE_INTERVAL = int(os.environ.get("CAPROUTE_IDLE_PROBE_INTERVAL", "60"))
DOWN_RETRY_INTERVAL = int(os.environ.get("CAPROUTE_DOWN_RETRY_INTERVAL", "30"))
SYNC_INTERVAL = int(os.environ.get("CAPROUTE_SYNC_INTERVAL", "60"))

# ── Config sync state ───────────────────────────────────────────────
_sync_state = {
    "last_poll": 0,
    "last_update_from": None,
    "last_update_at": 0,
    "peers": {},  # peer_url -> {"last_seen": ts, "mtime": ts, "error": str|None}
}
_sync_lock = threading.Lock()


def _config_mtime():
    """Get mtime of local config file as epoch float."""
    try:
        return CONFIG_PATH.stat().st_mtime
    except OSError:
        return 0


def _sync_poll_peers():
    """Poll sync_peers for newer config. Newest mtime wins."""
    try:
        cfg = load_config()
    except Exception:
        return
    peers = cfg.get("sync_peers", [])
    if not peers:
        return
    local_mtime = _config_mtime()
    best_mtime = local_mtime
    best_config = None
    best_peer = None
    for peer_url in peers:
        url = peer_url.rstrip("/") + "/config"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            peer_mtime = data.get("mtime", 0)
            peer_cfg = data.get("config")
            with _sync_lock:
                _sync_state["peers"][peer_url] = {
                    "last_seen": time.time(),
                    "mtime": peer_mtime,
                    "error": None,
                }
            if peer_mtime > best_mtime and peer_cfg:
                best_mtime = peer_mtime
                best_config = peer_cfg
                best_peer = peer_url
        except Exception as e:
            with _sync_lock:
                _sync_state["peers"][peer_url] = {
                    "last_seen": _sync_state["peers"]
                    .get(peer_url, {})
                    .get("last_seen", 0),
                    "mtime": 0,
                    "error": str(e)[:120],
                }
    if best_config and best_peer:
        # Preserve local sync_peers (each machine controls its own peer list)
        try:
            local_cfg = load_config()
            local_sync_peers = local_cfg.get("sync_peers", [])
        except Exception:
            local_sync_peers = []
        if local_sync_peers:
            best_config["sync_peers"] = local_sync_peers
        elif "sync_peers" in best_config:
            del best_config["sync_peers"]
        # Atomic write: write to tmp then rename
        tmp_path = CONFIG_PATH.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(best_config, f, indent=2)
            f.write("\n")
        os.replace(str(tmp_path), str(CONFIG_PATH))
        # Restore mtime from source so other peers see the same timestamp
        os.utime(CONFIG_PATH, (best_mtime, best_mtime))
        with _sync_lock:
            _sync_state["last_update_from"] = best_peer
            _sync_state["last_update_at"] = time.time()
        print(f"[config-sync] Updated from {best_peer} (mtime {best_mtime:.0f})")
    with _sync_lock:
        _sync_state["last_poll"] = time.time()


def _sync_loop():
    """Background thread: poll peers every SYNC_INTERVAL seconds."""
    while True:
        try:
            _sync_poll_peers()
        except Exception as e:
            print(f"[config-sync] Error: {e}")
        time.sleep(SYNC_INTERVAL)


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
_in_flight_since = {}  # "host:model" -> [t0, t1, ...]  (one timestamp per request)
_active_conns = {}   # "host:model" -> [conn, ...]  (http.client.HTTPConnection objects)

# ── Session affinity (cache-warmth stickiness) ───────────────────────
# Tracks which backend recently served each "session", so subsequent
# requests from the same conversation prefer the same backend and benefit
# from warm KV cache. Session identity is derived from the request's `user`
# field (OpenAI standard) or a hash of the system prompt + first user msg.
#
# IMPORTANT: the values below are initial heuristics, NOT tuned from data.
# See ~/Syncs/brain-ousted/System/Caproute_Session_Affinity_Tuning.md for
# tuning plan based on /stats data we log.
import hashlib

_session_affinity = {}  # session_id -> {backend_key: last_used_ts}
_affinity_lock = threading.Lock()
_AFFINITY_STRONG_WINDOW_S = 300  # 5 min: strong preference for same backend
_AFFINITY_MILD_WINDOW_S = 900  # 15 min: mild preference
_AFFINITY_STRONG_BONUS = 40000  # score reduction (lower score = better)
_AFFINITY_MILD_BONUS = (
    25000  # tuned 2026-04-05: 15K was too weak vs p50 base scores of 28-57K
)
_AFFINITY_MAX_SESSIONS = 500  # LRU cap to prevent unbounded growth

# Routing decision log (for offline analysis of affinity effectiveness)
_routing_log = collections.deque(maxlen=5000)
_routing_log_lock = threading.Lock()


def _compute_session_id(req):
    """Derive a stable session identifier from the request.

    Priority:
    1. Client-supplied `user` field (OpenAI standard).
    2. Hash of (system prompt + first user message) — stable across turns
       of the same conversation because those don't change.

    Returns a short hex string (16 chars) prefixed with source tag.
    """
    user = req.get("user")
    if user:
        return "u:" + hashlib.sha256(str(user).encode()).hexdigest()[:16]

    msgs = req.get("messages") or []
    sig_parts = []
    # Collect system message(s) + first user message for the prefix hash
    saw_user = False
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content if c)
        if role == "system":
            sig_parts.append(f"sys:{str(content)[:2000]}")
        elif role == "user" and not saw_user:
            sig_parts.append(f"usr:{str(content)[:1000]}")
            saw_user = True
            break
    if not sig_parts:
        return "empty"
    sig = "|".join(sig_parts)
    return "p:" + hashlib.sha256(sig.encode()).hexdigest()[:16]


def _record_session_usage(session_id, backend_key):
    """Mark that this session just used this backend.

    Replaces any previous backend affinity for this session so we stick
    to the new GPU and benefit from its cache.  Also LRU-trim.
    """
    now = time.time()
    with _affinity_lock:
        if session_id not in _session_affinity:
            if len(_session_affinity) >= _AFFINITY_MAX_SESSIONS:
                oldest = min(
                    _session_affinity.items(),
                    key=lambda kv: max(kv[1].values()) if kv[1] else 0,
                )[0]
                del _session_affinity[oldest]
            _session_affinity[session_id] = {}
        # Clear all previous backend affinities — commit fully to the
        # new backend so cache warmth stickiness follows the GPU we
        # actually got a slot on, not the one we used to use.
        _session_affinity[session_id] = {backend_key: now}


def _affinity_bonus(session_id, backend_key):
    """Return score reduction (positive number) if session has recent affinity."""
    if not session_id:
        return 0
    with _affinity_lock:
        sess = _session_affinity.get(session_id)
        if not sess:
            return 0
        ts = sess.get(backend_key)
        if ts is None:
            return 0
        age = time.time() - ts
    if age < _AFFINITY_STRONG_WINDOW_S:
        return _AFFINITY_STRONG_BONUS
    if age < _AFFINITY_MILD_WINDOW_S:
        return _AFFINITY_MILD_BONUS
    return 0


def _log_routing(
    session_id,
    capability,
    chosen_backend,
    score,
    base_score,
    affinity_bonus,
    candidates_count,
    latency_ms=0,
    prompt_tokens=0,
    completion_tokens=0,
):
    """Record a routing decision for offline analysis.

    latency_ms, prompt_tokens, completion_tokens are filled post-response
    to enable cache-warmth measurement (lower prompt_tokens on turn 2+ = cache hit).
    """
    with _routing_log_lock:
        _routing_log.append(
            {
                "ts": time.time(),
                "session_id": session_id,
                "capability": capability,
                "backend": chosen_backend,
                "final_score": score,
                "base_score": base_score,
                "affinity_bonus": affinity_bonus,
                "candidates_count": candidates_count,
                "latency_ms": round(latency_ms),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )


# ── Request history (ring buffer for dashboard charts) ───────────────
_request_history = collections.deque(maxlen=2000)
_history_lock = threading.Lock()


import re

_QUANT_RE = re.compile(
    r"[-_](Q\d+_K(?:_[A-Z]+)?|q\d+_\d+|f16|f32|bf16)$", re.IGNORECASE
)
_SIZE_RE = re.compile(r"^(.+?)[-:]([a-z]?\d+[bB](?:-[a-z0-9]+)?)$")


def normalize_model_name(name):
    """Normalize a model name for display grouping across backends.
    e.g. 'Qwen3.5-27B-Q4_K_M.gguf' -> 'qwen3.5:27b'
         'gemma4-e2b-it-q8_0' -> 'gemma4:e2b'
    """
    n = name.strip()
    # Strip .gguf
    if n.lower().endswith(".gguf"):
        n = n[:-5]
    # Lowercase
    n = n.lower()
    # Strip :latest
    if n.endswith(":latest"):
        n = n[:-7]
    # Strip quantization suffixes (Q4_K_M, q8_0, etc.)
    n = _QUANT_RE.sub("", n)
    # Strip instruction-tuned marker
    if n.endswith("-it"):
        n = n[:-3]
    # Normalize family-size pattern: "qwen3.5-27b" -> "qwen3.5:27b"
    m = _SIZE_RE.match(n)
    if m:
        n = m.group(1) + ":" + m.group(2)
    return n


def _record_request(model, host, capability, latency_ms, success, start_ts=None):
    """Append a request record to the ring buffer and persistent SQLite store."""
    now = time.time()
    entry = {
        "ts": now,
        "start_ts": start_ts if start_ts is not None else now,
        "model": normalize_model_name(model),
        "host": host,
        "capability": capability,
        "latency_ms": round(latency_ms),
        "ok": success,
    }
    with _history_lock:
        _request_history.append(entry)
    _db_record_request(entry)


# ── Persistent history (SQLite, 14-day retention) ─────────────────────
#
# The ring buffer above stays in-memory for fast dashboard access. SQLite
# adds durability across restarts and enables historical queries beyond
# the 2000-entry buffer. Retention window is pruned hourly.

_DB_PATH = os.environ.get(
    "CAPROUTE_DB_PATH",
    os.path.expanduser("~/.local/state/caproute/history.db"),
)
_DB_RETENTION_DAYS = int(os.environ.get("CAPROUTE_DB_RETENTION_DAYS", "14"))
_DB_PRUNE_INTERVAL = 3600  # seconds — hourly
_db_conn = None
_db_lock = threading.Lock()


def _db_init():
    """Open DB, create schema, apply pragmas. Safe to call multiple times."""
    global _db_conn
    if _db_conn is not None:
        return
    try:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        _db_conn = sqlite3.connect(
            _DB_PATH, check_same_thread=False, isolation_level=None
        )
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _db_conn.execute("PRAGMA synchronous=NORMAL")
        _db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                ts REAL NOT NULL,
                start_ts REAL,
                model TEXT NOT NULL,
                host TEXT NOT NULL,
                capability TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                ok INTEGER NOT NULL
            )
            """
        )
        _db_conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")
        try:
            _db_conn.execute("ALTER TABLE requests ADD COLUMN start_ts REAL")
        except Exception:
            pass  # already exists
    except Exception as e:
        print(f"[caproute] db init failed ({_DB_PATH}): {e}")
        _db_conn = None


def _db_record_request(entry):
    """Persist a single request. Fire-and-forget, never raises."""
    if _db_conn is None:
        return
    try:
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO requests(ts,start_ts,model,host,capability,latency_ms,ok) VALUES (?,?,?,?,?,?,?)",
                (
                    entry["ts"],
                    entry.get("start_ts", entry["ts"]),
                    entry["model"],
                    entry["host"],
                    entry["capability"],
                    entry["latency_ms"],
                    1 if entry["ok"] else 0,
                ),
            )
    except Exception as e:
        print(f"[caproute] db write failed: {e}")


def _db_load_recent(limit=2000):
    """Return most recent N entries, oldest-first (for ring buffer repopulation)."""
    if _db_conn is None:
        return []
    try:
        with _db_lock:
            cur = _db_conn.execute(
                "SELECT ts,start_ts,model,host,capability,latency_ms,ok FROM requests ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "ts": r[0],
                "start_ts": r[1] if r[1] is not None else r[0],
                "model": r[2],
                "host": r[3],
                "capability": r[4],
                "latency_ms": r[5],
                "ok": bool(r[6]),
            }
            for r in reversed(rows)
        ]
    except Exception as e:
        print(f"[caproute] db load failed: {e}")
        return []


def _db_query_since(since):
    """Return all entries with ts > since, oldest-first."""
    if _db_conn is None:
        return []
    try:
        with _db_lock:
            cur = _db_conn.execute(
                "SELECT ts,start_ts,model,host,capability,latency_ms,ok FROM requests WHERE ts > ? ORDER BY ts",
                (since,),
            )
            rows = cur.fetchall()
        return [
            {
                "ts": r[0],
                "start_ts": r[1] if r[1] is not None else r[0],
                "model": r[2],
                "host": r[3],
                "capability": r[4],
                "latency_ms": r[5],
                "ok": bool(r[6]),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[caproute] db query failed: {e}")
        return []


def _db_prune_loop():
    """Hourly: delete entries older than retention window."""
    while True:
        time.sleep(_DB_PRUNE_INTERVAL)
        if _db_conn is None:
            continue
        try:
            cutoff = time.time() - (_DB_RETENTION_DAYS * 86400)
            with _db_lock:
                cur = _db_conn.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))
                deleted = cur.rowcount
            if deleted > 0:
                print(
                    f"[caproute] pruned {deleted} history rows older than {_DB_RETENTION_DAYS}d"
                )
        except Exception as e:
            print(f"[caproute] prune failed: {e}")


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
        # Exponential moving average — asymmetric alpha for faster recovery.
        # When latency improves (current < old avg), use higher alpha so
        # backends bounce back quickly after transient slowness.
        old_avg = s.get("avg_latency_ms", 500)
        alpha = 0.5 if latency_ms < old_avg else 0.3
        s["avg_latency_ms"] = alpha * latency_ms + (1 - alpha) * old_avg
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
        _in_flight_since.setdefault(key, []).append(time.time())
    with _backend_lock:
        if key in _backend_state:
            _backend_state[key]["in_flight"] = _in_flight.get(key, 0)


def _dec_in_flight(key):
    with _in_flight_lock:
        _in_flight[key] = max(0, _in_flight.get(key, 1) - 1)
        ts_list = _in_flight_since.get(key, [])
        if ts_list:
            ts_list.pop(0)  # remove oldest timestamp (FIFO)
    with _backend_lock:
        if key in _backend_state:
            _backend_state[key]["in_flight"] = _in_flight.get(key, 0)


def _get_oldest_in_flight_age(key):
    """Returns seconds since oldest in-flight request was dispatched, or None."""
    with _in_flight_lock:
        ts_list = _in_flight_since.get(key, [])
        if not ts_list:
            return None
        return time.time() - ts_list[0]


# Per-capability in-flight penalty (ms per in-flight request).
# Loaded from llm.json inflight_penalties at call time so changes
# take effect without a restart (config is hot-reloaded).
# Fallback hardcoded values used when config key is absent.
_INFLIGHT_PENALTY_DEFAULTS = {
    "fast": 3_000,
    "light": 8_000,
    "adequate": 20_000,
    "thinking": 25_000,
    "powerful": 25_000,
    "reasoning-fast": 15_000,
}
_INFLIGHT_PENALTY_DEFAULT = 5_000


def _get_inflight_penalty(capability):
    """Return per-in-flight penalty for this capability, from config or defaults."""
    try:
        cfg = load_config()
        penalties = cfg.get("inflight_penalties", {})
        default = penalties.get("_default", _INFLIGHT_PENALTY_DEFAULT)
        return penalties.get(capability, _INFLIGHT_PENALTY_DEFAULTS.get(capability, default))
    except Exception:
        return _INFLIGHT_PENALTY_DEFAULTS.get(capability, _INFLIGHT_PENALTY_DEFAULT)


def backend_score(key, session_id=None, capability=None):
    """
    Score for routing. Lower = better.
    Combines: avg latency, consecutive failures, in-flight count, and
    (when session_id given) session affinity bonus that prefers recently-
    used backends for cache warmth.

    capability: when provided, scales the in-flight penalty to match the
    typical latency of that capability so backends are not overloaded.
    Backward compatible: session_id and capability are optional.
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
    # Penalty for consecutive failures, capped at 6 failures (30k max)
    # so transient outages don't permanently exclude backends
    score += min(failures, 6) * 5000
    # Penalty for in-flight requests (prefer less loaded backends).
    # Penalty is scaled by capability to match expected request duration,
    # so a single in-flight thinking request (30-50s) penalises a backend
    # as heavily as ~8 fast requests.  Higher multiplier when slow/down.
    _base_penalty = _get_inflight_penalty(capability)
    if status in ("slow", "down"):
        score += in_flight * max(_base_penalty, 8000)
    else:
        score += in_flight * _base_penalty
    # Penalty for idle (available but not loaded — will need cold-start)
    if status == "idle":
        score += 10000
    # Penalty for down (recently failed — try last but don't skip entirely)
    # Reduced from 50000 to 15000 so recovered backends re-enter rotation faster
    elif status == "down":
        score += 15000
    # Small penalty for unknown status (prefer backends we've seen succeed)
    elif status == "unknown":
        score += 1000

    # Session affinity: reward backends that recently served this session.
    # Subtracted (not added) because lower score = better. Only kicks in when
    # the backend is at least minimally healthy (not permanently down).
    if session_id and status != "down":
        score -= _affinity_bonus(session_id, key)

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
    Queries /v1/models and checks each model's status field.
    Models with status.value == "loaded" are considered loaded.
    If no status fields are present, falls back to assuming all models are loaded.
    Returns (loaded, total) sets."""
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
                # No status field on this model — assume loaded (backwards compat)
                loaded.add(mid)
        if not has_status:
            # No model had a status field — assume all are loaded (generic OpenAI API)
            return total, total
        return loaded, total
    except Exception:
        return set(), set()


def _passive_health_check_openai(backend):
    """Check llama-server health via /health (no inference). Returns latency_ms or None.

    Tries /slots first (llama-server-specific), falls back to /health (universal).
    Much lighter than sending a real "hi" inference probe — uses zero model compute.
    """
    key = _backend_key(backend["host"], backend["name"])
    base_url = backend["base_url"].rstrip("/")
    # Try /slots first, then /health as fallback (some servers disable /slots)
    # During prompt eval, llama-server returns 500 for /slots but /health may
    # still work (especially through the proxy). Don't count a /slots 500 as a
    # backend failure — fall through to /health first.
    for endpoint in ("/slots", "/health"):
        try:
            t0 = time.time()
            req = urllib.request.Request(base_url + endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status >= 400:
                    continue
                resp.read()
            latency_ms = (time.time() - t0) * 1000
            _record_success(key, latency_ms)
            return latency_ms
        except urllib.error.HTTPError:
            continue  # 500 from /slots during prompt eval — try /health
        except Exception:
            continue
    _record_failure(key)
    return None


def _probe_backend(backend):
    """Health check a backend.

    For llama-server (api="openai"): uses /slots passive check (no inference).
    For Ollama: uses passive observation only — real latency data comes from
    actual user requests via _record_success. A lightweight /api/ps check
    confirms the model is reachable.
    """
    key = _backend_key(backend["host"], backend["name"])

    if backend["api"] == "openai":
        return _passive_health_check_openai(backend)

    # Ollama: lightweight reachability check via /api/ps (no inference)
    try:
        t0 = time.time()
        url = backend["base_url"].rstrip("/") + "/api/ps"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read())
        latency_ms = (time.time() - t0) * 1000
        # Only update latency if no actual request latency is already tracked.
        # Passive /api/ps latency is network-only, not inference latency.
        state = _get_backend_state(key)
        if (
            state.get("last_success", 0) == 0
            or (time.time() - state.get("last_success", 0)) > 600
        ):
            # No recent real traffic — record passive latency as baseline
            _record_success(key, latency_ms)
        else:
            # Just refresh last_probe timestamp without touching latency
            with _backend_lock:
                s = _backend_state.get(key, {})
                s["last_probe"] = time.time()
                _backend_state[key] = s
        return latency_ms
    except Exception:
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
#
# Values are (target_capability, overrides_dict) tuples. The overrides are merged
# into the request params when the fallback hop is taken, allowing the source tier
# to constrain the fallback target (e.g. cap reasoning effort when reasoning-fast
# escalates to the heavyweight thinking tier).
#
# Backward compat: string values (old format) are treated as (target, {}).
DEFAULT_FALLBACKS = {
    "light": ("adequate", {}),
    "fast": ("adequate", {}),
    "adequate": ("powerful", {}),
    "powerful": ("thinking", {}),
    "style": ("adequate", {}),
    "reasoning-fast": ("thinking", {"reasoning_effort": "low"}),
}

# Per-capability param overrides injected on EVERY request for that capability
# (unlike fallback overrides which only apply on escalation).
# Client-supplied params always win over these defaults.
CAPABILITY_OVERRIDES = {
    "light": {"reasoning_effort": "none", "think": False},
}


def _normalize_fallback(value):
    """Accept either string (legacy) or (target, overrides) tuple. Returns tuple."""
    if isinstance(value, str):
        return (value, {})
    if isinstance(value, (list, tuple)) and len(value) == 2:
        target, overrides = value
        return (target, dict(overrides) if overrides else {})
    # Malformed — treat as string coercion
    return (str(value), {})


def get_capabilities():
    cfg = load_config()
    if _is_new_format(cfg):
        return sorted(cfg["capabilities"].keys())
    caps = set()
    for info in cfg.get("models", {}).values():
        caps.update(info.get("capabilities", []))
    return sorted(caps)


def _get_fallbacks(cfg):
    """Get fallback chain from config, merged with defaults. All values normalized
    to (target_capability, overrides_dict) tuples."""
    fallbacks = {k: _normalize_fallback(v) for k, v in DEFAULT_FALLBACKS.items()}
    for k, v in cfg.get("fallbacks", {}).items():
        fallbacks[k] = _normalize_fallback(v)
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


def _resolve_capability_backends(capability, session_id=None):
    """Resolve a capability to backends sorted by probe score.

    When session_id is provided, backends recently used by that session
    receive a score bonus (cache warmth stickiness). Returns empty list
    if no backends available.
    """
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
            base = backend_score(key, session_id=None, capability=capability)  # no affinity
            score = backend_score(key, session_id=session_id, capability=capability)
            backends.append(
                {
                    "name": model_name,
                    "base_url": h["url"],
                    "api": h["api"],
                    "host": host_name,
                    "_score": score,
                    "_base_score": base,
                    "_affinity_bonus": max(0, base - score),
                    "_key": key,
                }
            )

    backends.sort(key=lambda b: b["_score"])
    return backends


def _has_healthy_backend(backends):
    """Check if any backend in the list is not down (score < 999999)."""
    return any(b["_score"] < 999999 for b in backends)


def resolve_capability(capability, session_id=None):
    """
    Resolve a capability to backends sorted by probe score.
    If all backends are down, follows the fallback chain upward to higher-quality
    categories until a healthy backend is found. Each hop in the chain may contribute
    override params (e.g. reasoning_effort) that are accumulated and applied when the
    fallback target is reached.
    Returns (backends, actual_capability, overrides) tuple.
    When session_id is given, prefers backends with recent affinity to that session.
    """
    backends = _resolve_capability_backends(capability, session_id=session_id)
    actual_cap = capability
    overrides = {}

    if backends and not _has_healthy_backend(backends):
        # All backends for this capability are down — try fallback chain
        cfg = load_config()
        fallbacks = _get_fallbacks(cfg)
        # Build ordered chain of (target_cap, overrides) pairs, accumulating overrides
        chain = []
        current = capability
        acc_overrides = {}
        visited = set()
        while current in fallbacks and current not in visited:
            visited.add(current)
            next_cap, hop_overrides = fallbacks[current]
            # Accumulate: earlier hops set defaults, later hops override
            acc_overrides = {**acc_overrides, **hop_overrides}
            chain.append((next_cap, dict(acc_overrides)))
            current = next_cap

        for fallback_cap, fallback_overrides in chain:
            fb_backends = _resolve_capability_backends(
                fallback_cap, session_id=session_id
            )
            if fb_backends and _has_healthy_backend(fb_backends):
                backends = fb_backends
                actual_cap = fallback_cap
                overrides = fallback_overrides
                ov_str = f" overrides={overrides}" if overrides else ""
                print(f"[caproute] fallback: {capability} -> {actual_cap}{ov_str}")
                break

    return backends, actual_cap, overrides


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
    """Extract text content and tool_calls from a chat completion response."""
    # DEBUG
    print(f"[DEBUG _extract_content] result keys: {result.keys()}")

    tool_calls = None

    # OpenAI format
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        print(f"[DEBUG _extract_content] msg keys: {msg.keys()}")
        tool_calls = msg.get("tool_calls")
        content = msg.get("content", "")
        if content or tool_calls:
            return content, tool_calls
        # Fallback: reasoning models
        for key in ("reasoning_content", "reasoning", "thinking"):
            reasoning = msg.get(key, "")
            if reasoning:
                print(
                    f"[DEBUG _extract_content] found reasoning via key={key}, len={len(reasoning)}"
                )
                return reasoning, tool_calls

    # Ollama format (wrapped by _proxy_ollama)
    msg = result.get("message", {})
    if msg:
        tool_calls = msg.get("tool_calls")
        content = msg.get("content", "")
        if content or tool_calls:
            return content, tool_calls
        for key in ("reasoning_content", "reasoning", "thinking"):
            reasoning = msg.get(key, "")
            if reasoning:
                return reasoning, tool_calls
    return "", tool_calls


def _has_free_slot(base_url, api_type, model_name=None, timeout=2):
    """Check if a backend has a free inference slot.

    Queries GET /slots on llama-server and compatible proxies.
    For ollama backends without /slots support, assumes available.

    Returns True if at least one slot is free.
    Returns False if slots are busy OR if /slots errors/times out (during
    prompt eval, llama-server returns 500 and can't serve /slots queries).
    """
    try:
        if api_type == "ollama":
            return True
        url = base_url.rstrip("/") + "/slots"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            slots = json.loads(resp.read())
        if not isinstance(slots, list) or len(slots) == 0:
            return True  # empty or unknown = assume available
        free = sum(1 for s in slots if not s.get("is_processing", False))
        return free > 0
    except urllib.error.HTTPError:
        return False  # 500 during prompt eval = treat as busy
    except (TimeoutError, OSError):
        return False  # timeout = backend too busy to respond
    except Exception:
        return True  # connection refused etc = backend may be restarting, let probe handle it


def _http_post(url, body_bytes, connect_timeout, read_timeout, backend_key=None):
    """POST with separate connect and read timeouts.

    connect_timeout: max seconds to establish TCP connection and send request.
                     This is the "get a slot" window — if the backend is down
                     or unreachable, we fail fast.
    read_timeout:    max seconds to wait for the response body once connected.
                     This covers the actual LLM generation time.
    backend_key:     if provided, registers the live conn in _active_conns so the
                     stuck-request watchdog can close it without a full restart.
    """
    import http.client

    parsed = urllib.parse.urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(
            parsed.hostname, port, timeout=connect_timeout
        )
    else:
        conn = http.client.HTTPConnection(
            parsed.hostname, port, timeout=connect_timeout
        )
    try:
        conn.connect()  # uses connect_timeout
        conn.sock.settimeout(read_timeout)  # switch to full timeout for generation
        # Register conn so the stuck-request watchdog can forcibly close it
        if backend_key:
            with _in_flight_lock:
                _active_conns.setdefault(backend_key, []).append(conn)
        conn.request(
            "POST",
            parsed.path,
            body=body_bytes,
            headers={
                "Content-Type": "application/json",
            },
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            raise urllib.error.HTTPError(
                url,
                resp.status,
                http.client.responses.get(resp.status, ""),
                resp.msg,
                None,
            )
        return json.loads(resp.read())
    finally:
        if backend_key:
            with _in_flight_lock:
                lst = _active_conns.get(backend_key, [])
                try:
                    lst.remove(conn)
                except ValueError:
                    pass
        conn.close()


def _proxy_ollama(base_url, model, messages, params, timeout, connect_timeout=None, backend_key=None):
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
    if params.get("tools") is not None:
        data["tools"] = params["tools"]
    # Ollama-specific: think parameter controls reasoning on/off
    if "think" in params:
        data["think"] = params["think"]

    ct = connect_timeout or timeout
    result = _http_post(url, json.dumps(data).encode(), ct, timeout, backend_key=backend_key)

    content, tool_calls = _extract_content(result)
    # Ollama returns eval_count/prompt_eval_count, convert to OpenAI usage format
    usage = {
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "completion_tokens": result.get("eval_count", 0),
        "total_tokens": result.get("prompt_eval_count", 0)
        + result.get("eval_count", 0),
    }
    return _wrap_openai_response(model, content, tool_calls, usage=usage)


def _proxy_openai(base_url, model, messages, params, timeout, connect_timeout=None, backend_key=None):
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
    for key in [
        "tools",
        "tool_choice",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
    ]:
        if params.get(key) is not None:
            data[key] = params[key]

    ct = connect_timeout or timeout
    result = _http_post(url, json.dumps(data).encode(), ct, timeout, backend_key=backend_key)

    content, tool_calls = _extract_content(result)
    usage = result.get("usage", {})
    return _wrap_openai_response(model, content, tool_calls, usage=usage)


def _wrap_openai_response(model, content, tool_calls=None, usage=None):
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    else:
        message["content"] = ""

    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"caproute-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def proxy_to_backend(backend, messages, params, timeout, connect_timeout=None):
    bkey = _backend_key(backend["host"], backend["name"])
    if backend["api"] == "ollama":
        return _proxy_ollama(
            backend["base_url"],
            backend["name"],
            messages,
            params,
            timeout,
            connect_timeout=connect_timeout,
            backend_key=bkey,
        )
    else:
        return _proxy_openai(
            backend["base_url"],
            backend["name"],
            messages,
            params,
            timeout,
            connect_timeout=connect_timeout,
            backend_key=bkey,
        )


# ── Dashboard HTML ───────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>caproute dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e0e0e0; --dim: #888; --accent: #6c8cff;
    --ok: #4ade80; --slow: #fbbf24; --down: #f87171; --idle: #64748b; --unknown: #94a3b8;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; padding: 20px; }
  h1 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }
  .header-left { display: flex; align-items: baseline; gap: 16px; }
  .meta { color: var(--dim); font-size: 11px; }
  .stats { display: flex; gap: 20px; margin-bottom: 20px; }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 120px; }
  .stat-value { font-size: 22px; font-weight: 700; }
  .stat-label { color: var(--dim); font-size: 11px; margin-top: 2px; }
  .section-title { font-size: 13px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 10px; }
  .hosts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
  .host-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .host-name { font-size: 15px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
  .host-indicator { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .model-row { display: flex; align-items: center; justify-content: space-between; padding: 4px 0; border-top: 1px solid var(--border); font-size: 12px; }
  .model-row:first-child { border-top: none; }
  .model-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .model-meta { display: flex; gap: 10px; align-items: center; color: var(--dim); flex-shrink: 0; margin-left: 8px; }
  .badge { padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
  .badge-ok { background: rgba(74,222,128,0.15); color: var(--ok); }
  .badge-slow { background: rgba(251,191,36,0.15); color: var(--slow); }
  .badge-down { background: rgba(248,113,113,0.15); color: var(--down); }
  .badge-idle { background: rgba(100,116,139,0.15); color: var(--idle); }
  .badge-unknown { background: rgba(148,163,184,0.15); color: var(--unknown); }
  .cap-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  .cap-table th { text-align: left; color: var(--dim); font-weight: 500; font-size: 11px; padding: 6px 10px; border-bottom: 1px solid var(--border); }
  .cap-table td { padding: 5px 10px; border-bottom: 1px solid var(--border); font-size: 12px; }
  .cap-name { color: var(--accent); font-weight: 600; }
  .score { font-variant-numeric: tabular-nums; }
  .latency { font-variant-numeric: tabular-nums; }
  .in-flight { color: var(--slow); font-weight: 600; }
  .in-flight-zero { color: var(--dim); }
  .untagged-list { color: var(--dim); font-size: 12px; margin-top: 6px; }
  .pulse { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .error-banner { background: rgba(248,113,113,0.1); border: 1px solid var(--down); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; color: var(--down); display: none; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>caproute</h1>
    <span class="meta" id="mode">probe-routing</span>
  </div>
  <span class="meta" id="updated"></span>
</div>

<div class="error-banner" id="error-banner"></div>

<div class="stats" id="stats"></div>

<div class="section-title">Host Load Heatmap</div>
<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px;">
  <div style="display:flex; gap:10px; margin-bottom:10px; align-items:center;">
    <select id="heatmap-window" style="background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:4px 8px; font-family:inherit; font-size:12px;">
      <option value="15">Last 15 min</option>
      <option value="60" selected>Last 1 hour</option>
      <option value="360">Last 6 hours</option>
      <option value="1440">Last 24 hours</option>
    </select>
    <span class="meta" id="heatmap-info"></span>
    <span class="meta" style="margin-left:auto;">
      <span style="display:inline-flex;align-items:center;gap:3px;">light = few <span style="display:inline-block;width:60px;height:10px;border-radius:2px;background:linear-gradient(to right,#1a1d27,#6c8cff,#4ade80,#fbbf24,#f87171);"></span> heavy</span>
    </span>
  </div>
  <canvas id="heatmap-canvas" width="1200" height="200" style="width:100%; border-radius:4px;"></canvas>
</div>

<div class="section-title">Hosts</div>
<div class="hosts-grid" id="hosts-grid"></div>

<div class="section-title">Capabilities</div>
<table class="cap-table" id="cap-table">
  <thead><tr><th>Capability</th><th>Best Backend</th><th>Status</th><th>Score</th><th>Avg Latency</th><th>In-Flight</th></tr></thead>
  <tbody id="cap-body"></tbody>
</table>

<div id="untagged-section" style="display:none">
  <div class="section-title">Untagged Models</div>
  <div class="untagged-list" id="untagged-list"></div>
</div>

<div class="section-title">Usage Over Time</div>
<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px;">
  <div style="display:flex; gap:10px; margin-bottom:10px; align-items:center;">
    <select id="chart-window" style="background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:4px 8px; font-family:inherit; font-size:12px;">
      <option value="5">Last 5 min</option>
      <option value="15">Last 15 min</option>
      <option value="60" selected>Last 1 hour</option>
      <option value="360">Last 6 hours</option>
      <option value="1440">Last 24 hours</option>
    </select>
    <span class="meta" id="chart-info"></span>
  </div>
  <canvas id="usage-chart" width="1200" height="300" style="width:100%; height:300px;"></canvas>
  <div id="chart-legend" style="display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:10px; font-size:11px;"></div>
</div>

<script>
const POLL_MS = 5000;
let lastData = {};

function ago(ts) {
  if (!ts) return 'never';
  const s = Math.floor(Date.now()/1000 - ts);
  if (s < 5) return 'just now';
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  return Math.floor(s/3600) + 'h ago';
}

function statusBadge(status) {
  const cls = {ok:'badge-ok', slow:'badge-slow', down:'badge-down', idle:'badge-idle', unknown:'badge-unknown'}[status] || 'badge-unknown';
  return `<span class="badge ${cls}">${status}</span>`;
}

function hostStatus(models) {
  // host is ok if any model is ok, slow if any slow, down if all down
  const statuses = models.map(m => m.status);
  if (statuses.includes('ok')) return 'ok';
  if (statuses.includes('slow')) return 'slow';
  if (statuses.includes('idle') && !statuses.includes('down')) return 'idle';
  if (statuses.every(s => s === 'down')) return 'down';
  return 'unknown';
}

function statusColor(s) {
  return {ok:'var(--ok)', slow:'var(--slow)', down:'var(--down)', idle:'var(--idle)'}[s] || 'var(--unknown)';
}

async function fetchAll() {
  try {
    const [backends, health, discovery] = await Promise.all([
      fetch('/backends').then(r => r.json()),
      fetch('/health').then(r => r.json()),
      fetch('/discovery').then(r => r.json()),
    ]);
    document.getElementById('error-banner').style.display = 'none';
    lastData = { backends: backends.backends || {}, health, discovery };
    render();
  } catch (e) {
    const banner = document.getElementById('error-banner');
    banner.textContent = 'Failed to fetch data: ' + e.message;
    banner.style.display = 'block';
  }
}

function render() {
  const { backends, health, discovery } = lastData;
  if (!backends || !health) return;

  // Updated timestamp
  document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();

  // Stats
  const keys = Object.keys(backends);
  const okCount = keys.filter(k => backends[k].status === 'ok').length;
  const slowCount = keys.filter(k => backends[k].status === 'slow').length;
  const downCount = keys.filter(k => backends[k].status === 'down').length;
  const totalInFlight = keys.reduce((s, k) => s + (backends[k].in_flight || 0), 0);

  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="stat-value" style="color:var(--ok)">${okCount}</div><div class="stat-label">OK</div></div>
    <div class="stat"><div class="stat-value" style="color:var(--slow)">${slowCount}</div><div class="stat-label">Slow</div></div>
    <div class="stat"><div class="stat-value" style="color:var(--down)">${downCount}</div><div class="stat-label">Down</div></div>
    <div class="stat"><div class="stat-value">${totalInFlight}</div><div class="stat-label">In-Flight</div></div>
    <div class="stat"><div class="stat-value">${health.discovered_models || 0}</div><div class="stat-label">Models</div></div>
    <div class="stat"><div class="stat-value">${(health.hosts || []).length}</div><div class="stat-label">Hosts</div></div>
  `;

  // Group backends by host
  const byHost = {};
  for (const [key, state] of Object.entries(backends)) {
    const colonIdx = key.indexOf(':');
    const host = key.substring(0, colonIdx);
    const model = key.substring(colonIdx + 1);
    if (!byHost[host]) byHost[host] = [];
    byHost[host].push({ model, ...state });
  }

  // Sort hosts alphabetically, models by score
  const hostNames = Object.keys(byHost).sort();
  let hostsHtml = '';
  for (const host of hostNames) {
    const models = byHost[host].sort((a, b) => a.score - b.score);
    const hs = hostStatus(models);
    let modelsHtml = '';
    for (const m of models) {
      const latStr = m.avg_latency_ms ? m.avg_latency_ms + 'ms' : '-';
      const ifCls = m.in_flight > 0 ? 'in-flight' : 'in-flight-zero';
      const lastSeen = ago(m.last_success);
      modelsHtml += `<div class="model-row">
        <span class="model-name">${m.model}</span>
        <span class="model-meta">
          ${statusBadge(m.status)}
          <span class="latency">${latStr}</span>
          <span class="${ifCls}">${m.in_flight > 0 ? m.in_flight + ' req' : ''}</span>
          <span class="score">${m.score}</span>
        </span>
      </div>`;
    }
    hostsHtml += `<div class="host-card">
      <div class="host-name"><span class="host-indicator" style="background:${statusColor(hs)}"></span>${host}</div>
      ${modelsHtml}
    </div>`;
  }
  document.getElementById('hosts-grid').innerHTML = hostsHtml;

  // Capabilities table
  const capModels = health.capability_models || {};
  let capHtml = '';
  for (const cap of Object.keys(capModels).sort()) {
    const models = capModels[cap] || [];
    // Find best backend for this capability
    let best = null;
    let bestScore = Infinity;
    for (const model of models) {
      for (const [key, state] of Object.entries(backends)) {
        const colonIdx = key.indexOf(':');
        const bModel = key.substring(colonIdx + 1);
        if (bModel === model && state.score < bestScore) {
          bestScore = state.score;
          best = { key, ...state, model };
        }
      }
    }
    if (best) {
      const ifCls = best.in_flight > 0 ? 'in-flight' : 'in-flight-zero';
      capHtml += `<tr>
        <td class="cap-name">${cap}</td>
        <td>${best.key}</td>
        <td>${statusBadge(best.status)}</td>
        <td class="score">${best.score}</td>
        <td class="latency">${best.avg_latency_ms}ms</td>
        <td class="${ifCls}">${best.in_flight || 0}</td>
      </tr>`;
    } else {
      capHtml += `<tr>
        <td class="cap-name">${cap}</td>
        <td colspan="5" style="color:var(--dim)">no backends discovered</td>
      </tr>`;
    }
  }
  document.getElementById('cap-body').innerHTML = capHtml;

  // Untagged models
  const untagged = health.untagged_models || [];
  const us = document.getElementById('untagged-section');
  if (untagged.length > 0) {
    us.style.display = 'block';
    document.getElementById('untagged-list').textContent = untagged.join(', ');
  } else {
    us.style.display = 'none';
  }
}

// ── Host load heatmap ────────────────────────────────────────────

// Polyfill for roundRect (Chrome <99, Firefox <112, some WebViews)
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    if (typeof r === 'number') r = [r,r,r,r];
    this.moveTo(x+r[0], y);
    this.lineTo(x+w-r[1], y); this.quadraticCurveTo(x+w, y, x+w, y+r[1]);
    this.lineTo(x+w, y+h-r[2]); this.quadraticCurveTo(x+w, y+h, x+w-r[2], y+h);
    this.lineTo(x+r[3], y+h); this.quadraticCurveTo(x, y+h, x, y+h-r[3]);
    this.lineTo(x, y+r[0]); this.quadraticCurveTo(x, y, x+r[0], y);
    this.closePath();
  };
}

function heatColor(value, max) {
  if (max === 0 || value === 0) return 'rgba(42,45,58,0.6)';
  const t = Math.min(value / max, 1);
  // 4-stop gradient: surface -> blue -> green -> yellow -> red
  const stops = [
    [26, 29, 42],    // var(--surface)
    [108, 140, 255], // blue
    [74, 222, 128],  // green
    [251, 191, 36],  // yellow
    [248, 113, 113], // red
  ];
  const pos = t * (stops.length - 1);
  const i = Math.floor(pos);
  const f = pos - i;
  const a = stops[Math.min(i, stops.length - 1)];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

function renderHeatmap() {
  const canvas = document.getElementById('heatmap-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  const windowMin = parseInt(document.getElementById('heatmap-window').value);
  const now = Date.now() / 1000;
  const tMin = now - windowMin * 60;

  // Bucket into time columns
  const numCols = Math.min(48, windowMin);
  const bucketSec = (windowMin * 60) / numCols;

  // Group by host
  const hostBuckets = {};
  let totalReqs = 0;
  for (const r of historyData) {
    const rts = r.start_ts !== undefined ? r.start_ts : r.ts;
    if (rts < tMin) continue;
    totalReqs++;
    const host = r.host;
    if (!hostBuckets[host]) hostBuckets[host] = new Array(numCols).fill(0);
    const biStart = Math.max(0, Math.floor((rts - tMin) / bucketSec));
    const biEnd = Math.min(numCols - 1, Math.floor(((r.ts !== undefined ? r.ts : rts) - tMin) / bucketSec));
    for (let bi = biStart; bi <= biEnd; bi++) hostBuckets[host][bi]++;
  }

  const hosts = Object.keys(hostBuckets).sort();

  // Also include hosts from lastData that have no history yet
  if (lastData.health) {
    for (const h of (lastData.health.hosts || [])) {
      if (!hosts.includes(h)) hosts.push(h);
      if (!hostBuckets[h]) hostBuckets[h] = new Array(numCols).fill(0);
    }
    hosts.sort();
  }

  if (hosts.length === 0) {
    canvas.height = 60 * dpr;
    canvas.style.height = '60px';
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#888';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No data yet', rect.width / 2, 30);
    document.getElementById('heatmap-info').textContent = '';
    return;
  }

  // Find global max for color scaling
  let globalMax = 1;
  for (const h of hosts) {
    for (const v of hostBuckets[h]) {
      if (v > globalMax) globalMax = v;
    }
  }

  const labelW = 110;
  const cellGap = 1;
  const rowH = 28;
  const headerH = 20;
  const totalH = headerH + hosts.length * (rowH + cellGap);
  canvas.width = rect.width * dpr;
  canvas.height = totalH * dpr;
  canvas.style.height = totalH + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const plotW = rect.width - labelW;
  const cellW = (plotW - (numCols - 1) * cellGap) / numCols;

  // Time axis labels
  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  const xTicks = Math.min(6, numCols);
  for (let i = 0; i <= xTicks; i++) {
    const t = tMin + (i / xTicks) * (windowMin * 60);
    const x = labelW + (i / xTicks) * plotW;
    const d = new Date(t * 1000);
    ctx.fillText(d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0'), x, headerH - 4);
  }

  // Rows
  for (let ri = 0; ri < hosts.length; ri++) {
    const host = hosts[ri];
    const y = headerH + ri * (rowH + cellGap);
    const buckets = hostBuckets[host];
    const hostTotal = buckets.reduce((a,b) => a+b, 0);

    // Label
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(host, labelW - 10, y + rowH / 2 + 4);

    // Cells
    for (let ci = 0; ci < numCols; ci++) {
      const x = labelW + ci * (cellW + cellGap);
      ctx.fillStyle = heatColor(buckets[ci], globalMax);
      ctx.beginPath();
      ctx.roundRect(x, y, cellW, rowH, 2);
      ctx.fill();

      // Show count in cell if value > 0 and cells are wide enough
      if (buckets[ci] > 0 && cellW > 18) {
        ctx.fillStyle = buckets[ci] / globalMax > 0.6 ? '#000' : '#ccc';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(buckets[ci], x + cellW / 2, y + rowH / 2 + 3);
      }
    }

    // Total count at end
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
  }

  document.getElementById('heatmap-info').textContent = totalReqs + ' requests across ' + hosts.length + ' hosts';
}

document.getElementById('heatmap-window').addEventListener('change', fetchHistory);

// ── Usage chart ──────────────────────────────────────────────────
const MODEL_COLORS = [
  '#6c8cff','#4ade80','#fbbf24','#f87171','#a78bfa','#fb923c',
  '#38bdf8','#e879f9','#34d399','#f472b6','#facc15','#22d3ee',
  '#818cf8','#fb7185','#a3e635','#c084fc',
];
let historyData = [];
let colorMap = {};
let colorIdx = 0;

function getColor(model) {
  if (!colorMap[model]) {
    colorMap[model] = MODEL_COLORS[colorIdx % MODEL_COLORS.length];
    colorIdx++;
  }
  return colorMap[model];
}

async function fetchHistory() {
  try {
    const chartMin = parseInt(document.getElementById('chart-window').value);
    const heatMin = parseInt(document.getElementById('heatmap-window').value);
    const windowMin = Math.max(chartMin, heatMin);
    const since = Date.now()/1000 - windowMin * 60 - 3600;  // extra buffer for long-running requests
    const resp = await fetch('/history?since=' + since);
    const data = await resp.json();
    historyData = data.history || [];
    renderChart();
    renderHeatmap();
  } catch(e) { console.error('fetchHistory error:', e); }
}

function renderChart() {
  const canvas = document.getElementById('usage-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { top: 10, right: 20, bottom: 30, left: 45 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const windowMin = parseInt(document.getElementById('chart-window').value);
  const now = Date.now() / 1000;
  const tMin = now - windowMin * 60;
  const tMax = now;

  // Bucket requests into time bins — adaptive bucket size
  const numBuckets = Math.min(60, windowMin);
  const bucketSec = (windowMin * 60) / numBuckets;

  // Group by model
  const models = {};
  for (const r of historyData) {
    if (r.ts < tMin) continue;
    const key = r.model;
    if (!models[key]) models[key] = new Array(numBuckets).fill(0);
    const bi = Math.min(numBuckets - 1, Math.floor((r.ts - tMin) / bucketSec));
    models[key][bi]++;
  }

  const modelNames = Object.keys(models).sort();
  if (modelNames.length === 0) {
    ctx.fillStyle = 'var(--dim)';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#888';
    ctx.fillText('No request history yet', W/2, H/2);
    document.getElementById('chart-info').textContent = '0 requests';
    document.getElementById('chart-legend').innerHTML = '';
    return;
  }

  // Find max value for y-axis
  let yMax = 1;
  for (const name of modelNames) {
    for (const v of models[name]) {
      if (v > yMax) yMax = v;
    }
  }
  yMax = Math.ceil(yMax * 1.15);

  // Grid lines
  ctx.strokeStyle = '#2a2d3a';
  ctx.lineWidth = 0.5;
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const y = pad.top + plotH - (i / yTicks) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(yMax * i / yTicks), pad.left - 6, y + 3);
  }

  // Time labels
  ctx.textAlign = 'center';
  const xTicks = Math.min(6, numBuckets);
  for (let i = 0; i <= xTicks; i++) {
    const t = tMin + (i / xTicks) * (tMax - tMin);
    const x = pad.left + (i / xTicks) * plotW;
    const d = new Date(t * 1000);
    const label = d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
    ctx.fillStyle = '#888';
    ctx.fillText(label, x, pad.top + plotH + 18);
  }

  // Y-axis label
  ctx.save();
  ctx.translate(12, pad.top + plotH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('req / bucket', 0, 0);
  ctx.restore();

  // Plot lines
  for (const name of modelNames) {
    const color = getColor(name);
    const buckets = models[name];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < numBuckets; i++) {
      const x = pad.left + (i + 0.5) / numBuckets * plotW;
      const y = pad.top + plotH - (buckets[i] / yMax) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots at non-zero points
    for (let i = 0; i < numBuckets; i++) {
      if (buckets[i] > 0) {
        const x = pad.left + (i + 0.5) / numBuckets * plotW;
        const y = pad.top + plotH - (buckets[i] / yMax) * plotH;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Legend
  let legendHtml = '';
  for (const name of modelNames) {
    const color = getColor(name);
    const total = models[name].reduce((a,b) => a+b, 0);
    legendHtml += '<span style="display:inline-flex;align-items:center;gap:4px;">' +
      '<span style="width:10px;height:10px;border-radius:2px;background:'+color+';display:inline-block;"></span>' +
      name + ' <span style="color:#888">('+total+')</span></span>';
  }
  document.getElementById('chart-legend').innerHTML = legendHtml;
  document.getElementById('chart-info').textContent = historyData.length + ' requests in window';
}

document.getElementById('chart-window').addEventListener('change', fetchHistory);

fetchAll();
fetchHistory();
setInterval(fetchAll, POLL_MS);
setInterval(fetchHistory, POLL_MS);
</script>

<div style="margin-top:32px; border-top:1px solid var(--border); padding-top:16px; color:var(--dim); font-size:11px; line-height:1.7;">
  <div class="section-title" style="margin-top:0;">Glossary</div>
  <dl style="display:grid; grid-template-columns:120px 1fr; gap:4px 12px;">
    <dt style="color:var(--text);">Request</dt>
    <dd>A single chat completion call (<code>POST /v1/chat/completions</code>) routed through caproute to a backend. Each attempt to a backend counts as one request, including retries to different backends for the same client call.</dd>

    <dt style="color:var(--text);">Capability</dt>
    <dd>A logical label (e.g. <em>fast</em>, <em>powerful</em>, <em>thinking</em>) that maps to one or more models. Clients request a capability, and caproute picks the best available backend that serves a matching model.</dd>

    <dt style="color:var(--text);">Score</dt>
    <dd>A routing score computed per backend (lower is better). Combines average latency, consecutive failure count, in-flight request count, and status penalties (idle, down, unknown). Caproute always routes to the lowest-score backend first.</dd>

    <dt style="color:var(--text);">Latency</dt>
    <dd>Wall-clock time from when caproute sends the request to a backend until it receives the full response, in milliseconds. <em>Avg latency</em> is an exponential moving average &mdash; it recovers quickly when latency improves (alpha=0.5) and moves conservatively when it worsens (alpha=0.3).</dd>

    <dt style="color:var(--text);">In-Flight</dt>
    <dd>Number of requests currently being processed by a backend. Backends with higher in-flight counts get a score penalty so new requests are routed to less loaded machines.</dd>

    <dt style="color:var(--text);">Bucket</dt>
    <dd>A time interval used to aggregate requests in the charts. The time window is divided into equal-sized buckets (e.g. a 1-hour window uses ~60 one-minute buckets). Each bucket shows how many requests landed in that interval.</dd>

    <dt style="color:var(--text);">Status</dt>
    <dd>
      <span style="color:var(--ok);">ok</span> &mdash; avg latency &lt; 2s, no recent failures<br>
      <span style="color:var(--slow);">slow</span> &mdash; avg latency 2&ndash;10s, or 1&ndash;2 consecutive failures<br>
      <span style="color:var(--down);">down</span> &mdash; 3+ consecutive failures; retried every 30s<br>
      <span style="color:var(--idle);">idle</span> &mdash; model available on host but not loaded in memory (cold start needed)<br>
      <span style="color:var(--unknown);">unknown</span> &mdash; never probed or never succeeded
    </dd>

    <dt style="color:var(--text);">Heatmap</dt>
    <dd>Rows are hosts, columns are time buckets. Cell color intensity shows request volume relative to the busiest cell in the window &mdash; from dark (zero) through blue/green/yellow to red (peak load).</dd>

    <dt style="color:var(--text);">Fallback</dt>
    <dd>When all backends for a capability are down, caproute follows a fallback chain (e.g. fast &rarr; adequate &rarr; powerful &rarr; thinking) to find a healthy backend in a higher-quality tier.</dd>

    <dt style="color:var(--text);">Probe</dt>
    <dd>A tiny background request (5 tokens max) sent periodically to each loaded model to measure latency and detect failures without waiting for real traffic. Only models currently loaded in memory are probed.</dd>
  </dl>
</div>

</body>
</html>
"""

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

    def _send_sse(self, result):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            msg = result.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            finish_reason = result.get("choices", [{}])[0].get("finish_reason", "stop")

            chunk_base = {
                "id": result.get("id", "caproute-stream"),
                "object": "chat.completion.chunk",
                "created": result.get("created", int(time.time())),
                "model": result.get("model", "unknown"),
            }

            # Send content chunk if present
            if content:
                content_chunk = {
                    **chunk_base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(content_chunk)}\n\n".encode())

            # Send tool_calls as streaming deltas
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    # OpenAI streaming spec requires arguments as a JSON string
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    tc_delta = {
                        "index": i,
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": args,
                        },
                    }
                    tc_chunk = {
                        **chunk_base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [tc_delta]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(tc_chunk)}\n\n".encode())

            final_chunk = {
                **chunk_base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

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
        elif self.path.startswith("/history"):
            self._handle_history()
        elif self.path == "/dashboard":
            self._handle_dashboard()
        elif self.path == "/stats":
            self._handle_stats()
        elif self.path == "/stats/routing":
            self._handle_routing_log()
        elif self.path == "/config":
            self._handle_config()
        elif self.path == "/config/status":
            self._handle_config_status()
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
        elif self.path == "/config/sync":
            _sync_poll_peers()
            with _sync_lock:
                self._send_json({"status": "ok", "sync_state": dict(_sync_state)})
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
            in_flight_since = {k: list(v) for k, v in _in_flight_since.items()}

        now = time.time()
        result = {}
        for key, s in sorted(state.items()):
            ts_list = in_flight_since.get(key, [])
            oldest_age = round(now - ts_list[0], 1) if ts_list else None
            result[key] = {
                "status": s.get("status", "unknown"),
                "score": round(backend_score(key)),
                "avg_latency_ms": round(s.get("avg_latency_ms", 0)),
                "latency_ms": round(s.get("latency_ms", 0)),
                "failures": s.get("failures", 0),
                "in_flight": in_flight.get(key, s.get("in_flight", 0)),
                "oldest_in_flight_s": oldest_age,
                "last_success": s.get("last_success", 0),
                "last_probe": s.get("last_probe", 0),
            }
        self._send_json({"backends": result})

    def _handle_history(self):
        """Return request history, optionally filtered by ?since=<epoch>."""
        since = 0
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if part.startswith("since="):
                    try:
                        since = float(part.split("=", 1)[1])
                    except ValueError:
                        pass
        if since:
            # Historical queries hit SQLite for durability + beyond-ring-buffer range.
            records = _db_query_since(since)
        else:
            with _history_lock:
                records = list(_request_history)
        self._send_json({"history": records, "count": len(records)})

    def _handle_stats(self):
        """Return session affinity state + aggregate routing stats."""
        now = time.time()
        with _affinity_lock:
            sessions = {
                sid: {
                    "backends": [
                        {
                            "key": k,
                            "last_used_age_s": round(now - ts, 1),
                            "stickiness": (
                                "strong"
                                if now - ts < _AFFINITY_STRONG_WINDOW_S
                                else "mild"
                                if now - ts < _AFFINITY_MILD_WINDOW_S
                                else "expired"
                            ),
                        }
                        for k, ts in sorted(usage.items(), key=lambda kv: -kv[1])
                    ],
                }
                for sid, usage in _session_affinity.items()
            }

        # Compute per-session routing hit/miss summaries from the log.
        with _routing_log_lock:
            routing_log = list(_routing_log)

        per_session = {}
        per_backend = {}
        for entry in routing_log:
            sid = entry["session_id"]
            ps = per_session.setdefault(
                sid,
                {
                    "requests": 0,
                    "with_affinity": 0,
                    "total_bonus": 0,
                },
            )
            ps["requests"] += 1
            if entry["affinity_bonus"] > 0:
                ps["with_affinity"] += 1
                ps["total_bonus"] += entry["affinity_bonus"]
            pb = per_backend.setdefault(entry["backend"], {"requests": 0})
            pb["requests"] += 1

        self._send_json(
            {
                "config": {
                    "strong_window_s": _AFFINITY_STRONG_WINDOW_S,
                    "mild_window_s": _AFFINITY_MILD_WINDOW_S,
                    "strong_bonus": _AFFINITY_STRONG_BONUS,
                    "mild_bonus": _AFFINITY_MILD_BONUS,
                    "max_sessions": _AFFINITY_MAX_SESSIONS,
                },
                "active_sessions": len(sessions),
                "sessions": sessions,
                "per_session_routing": per_session,
                "per_backend_routing": per_backend,
                "total_routing_decisions": len(routing_log),
            }
        )

    def _handle_routing_log(self):
        """Raw routing log (last N decisions)."""
        with _routing_log_lock:
            records = list(_routing_log)
        self._send_json({"routing_log": records, "count": len(records)})

    def _handle_config(self):
        """Serve local config with mtime for peer sync."""
        try:
            cfg = load_config()
            # Strip sync_peers from response (private to each machine)
            cfg_out = {k: v for k, v in cfg.items() if k != "sync_peers"}
            self._send_json({"mtime": _config_mtime(), "config": cfg_out})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_config_status(self):
        """Show config sync status."""
        with _sync_lock:
            state = dict(_sync_state)
            state["peers"] = dict(state["peers"])
        try:
            cfg = load_config()
            state["sync_peers"] = cfg.get("sync_peers", [])
        except Exception:
            state["sync_peers"] = []
        state["local_mtime"] = _config_mtime()
        state["config_path"] = str(CONFIG_PATH)
        self._send_json(state)

    def _handle_dashboard(self):
        html = DASHBOARD_HTML
        body = html.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
        for key in [
            "temperature",
            "max_tokens",
            "tools",
            "tool_choice",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "reasoning_effort",
        ]:
            if key in req:
                params[key] = req[key]

        # Inject a sensible max_tokens default when client omits it.
        # (Disabled by user request to let backend decide)
        # if "max_tokens" not in params:
        #     params["max_tokens"] = 8192

        # Compute session identifier for affinity tracking (cache warmth).
        session_id = _compute_session_id(req)

        cfg = load_config()
        default_timeout = _get_timeout(cfg, model_field)
        timeout = req.get("timeout", default_timeout)

        current_cap = model_field
        errors = []
        visited_caps = set()
        accumulated_overrides = {}

        # ── Retry-loop routing ──────────────────────────────────────────
        # Instead of burning the full timeout on each backend sequentially,
        # use a short per-attempt timeout and loop back through all
        # capabilities until the total budget is exhausted.  This latches
        # onto whichever backend recovers first.
        _PER_ATTEMPT_BASE = int(os.getenv("CAPROUTE_PER_ATTEMPT_TIMEOUT", "2"))
        _total_deadline = time.time() + timeout
        _round = 0

        while True:
            _round += 1
            remaining = _total_deadline - time.time()
            if remaining <= 0:
                break

            # Per-attempt timeout is the CONNECT timeout only (time to get a slot).
            # Once connected, the full capability timeout governs generation time.
            # No need to grow — dead backends fail fast, busy ones accept quickly.
            _PER_ATTEMPT_TIMEOUT = _PER_ATTEMPT_BASE

            current_cap = model_field
            visited_caps_this_round = set()
            # Track backends that returned hard errors this round.
            # HTTP 400 = bad request (context too large, etc.) — permanent for this request.
            # HTTP 500 = server error — transient but skip for this round.
            # Timeouts are NOT tracked here — the backend was working, just slow.
            _hard_failed_keys = set()  # 400s — won't work for this request at all
            _soft_failed_keys = set()  # 500s — skip this round only

            while True:
                visited_caps_this_round.add(current_cap)

                # Resolve: capability name or direct model name
                backends, actual_cap, hop_overrides = resolve_capability(
                    current_cap, session_id=session_id
                )
                if hop_overrides:
                    accumulated_overrides = {**accumulated_overrides, **hop_overrides}

                # If the user explicitly requested "thinking" and no backends are healthy,
                # keep the actual backends so proxy_to_backend can wait for them.
                if (
                    actual_cap == "thinking"
                    and not _has_healthy_backend(backends)
                    and current_cap == model_field
                ):
                    backends = _resolve_capability_backends("thinking")

                if not backends and current_cap == model_field:
                    backends, _ = resolve_model_direct(model_field)

                if not backends:
                    if current_cap == model_field:
                        self._send_json(
                            {
                                "error": f"unknown capability or model: '{model_field}'",
                                "available": get_capabilities(),
                            },
                            400,
                        )
                        return
                    else:
                        break

                # Use the shorter of per-attempt timeout or remaining budget.
                attempt_timeout = min(
                    _PER_ATTEMPT_TIMEOUT, _total_deadline - time.time()
                )
                if attempt_timeout <= 0:
                    break

                # Separate backends into slot-available vs slot-busy.
                # Try slot-available first (including affinity-preferred).
                # If ALL are busy, skip to next round instead of queuing.
                # Filter out backends with hard errors (won't work for this request)
                # and soft errors (server error, skip this round). Timeouts are
                # NOT filtered — the backend was working, just slow.
                fresh_backends = [
                    b
                    for b in backends
                    if _backend_key(b["host"], b["name"]) not in _hard_failed_keys
                    and _backend_key(b["host"], b["name"]) not in _soft_failed_keys
                ]
                if not fresh_backends:
                    break  # all backends exhausted this round

                _slot_available = []
                _slot_busy = []
                for backend in fresh_backends:
                    if _has_free_slot(
                        backend["base_url"], backend["api"], backend["name"]
                    ):
                        _slot_available.append(backend)
                    else:
                        _slot_busy.append(backend)
                if _slot_busy:
                    _busy_names = ", ".join(
                        f"{b['name']}@{b['host']}" for b in _slot_busy
                    )
                    print(f"[caproute] slots full, deferring: {_busy_names}")
                if not _slot_available:
                    # All slots busy. Escalate immediately if a fallback exists,
                    # so we don't spin-wait for 210s on a fully-loaded tier.
                    cfg2 = load_config()
                    fbs = _get_fallbacks(cfg2)
                    if (
                        actual_cap in fbs
                        and fbs[actual_cap][0] not in visited_caps_this_round
                    ):
                        next_cap, hop_ov = fbs[actual_cap]
                        if hop_ov:
                            accumulated_overrides = {**accumulated_overrides, **hop_ov}
                        visited_caps_this_round.add(actual_cap)
                        current_cap = next_cap
                        print(
                            f"[caproute] slots full for {actual_cap}, "
                            f"escalating -> {current_cap}"
                        )
                        continue  # inner while — try next capability immediately
                    break  # no fallback available, break to outer retry loop

                for backend in _slot_available:
                    if time.time() >= _total_deadline:
                        break
                    key = _backend_key(backend["host"], backend["name"])
                    _inc_in_flight(key)
                    try:
                        t0 = time.time()
                        aff_bonus = backend.get("_affinity_bonus", 0)
                        aff_tag = f" aff=-{aff_bonus:.0f}" if aff_bonus else ""
                        round_tag = f" r={_round}" if _round > 1 else ""
                        print(
                            f"[caproute] {model_field} -> {backend['name']}@{backend['host']} (score={backend['_score']:.0f}{aff_tag}{round_tag} t={attempt_timeout:.0f}s sess={session_id})"
                        )
                        # Layer: capability defaults < fallback overrides < client params
                        cap_overrides = CAPABILITY_OVERRIDES.get(actual_cap, {})
                        call_params = {**cap_overrides, **accumulated_overrides, **params}
                        result = proxy_to_backend(
                            backend,
                            messages,
                            call_params,
                            timeout,
                            connect_timeout=attempt_timeout,
                        )
                        latency_ms = (time.time() - t0) * 1000
                        _record_success(key, latency_ms)
                        _record_request(
                            backend["name"],
                            backend["host"],
                            actual_cap,
                            latency_ms,
                            True,
                            start_ts=t0,
                        )
                        _record_session_usage(session_id, key)
                        _usage = result.get("usage", {})
                        _log_routing(
                            session_id,
                            actual_cap,
                            key,
                            backend["_score"],
                            backend.get("_base_score", backend["_score"]),
                            aff_bonus,
                            len(backends),
                            latency_ms=latency_ms,
                            prompt_tokens=_usage.get("prompt_tokens", 0),
                            completion_tokens=_usage.get("completion_tokens", 0),
                        )
                        result["_caproute"] = {
                            "capability": actual_cap,
                            "requested": model_field,
                            "model": backend["name"],
                            "host": backend["host"],
                            "latency_ms": round(latency_ms),
                            "score": round(backend["_score"]),
                            "round": _round,
                        }
                        if req.get("stream", False):
                            self._send_sse(result)
                        else:
                            self._send_json(result)
                        return
                    except Exception as e:
                        latency_ms = (time.time() - t0) * 1000
                        _record_failure(key)
                        _record_request(
                            backend["name"],
                            backend["host"],
                            actual_cap,
                            latency_ms,
                            False,
                            start_ts=t0,
                        )
                        # Categorize error to decide retry strategy:
                        # 400 = bad request (context too large) → never retry this request
                        # 500 = server error → skip this round, retry next round
                        # timeout = backend was working but slow → don't skip
                        err_str = str(e)
                        if "400" in err_str:
                            _hard_failed_keys.add(key)
                        elif "500" in err_str or "502" in err_str or "503" in err_str:
                            _soft_failed_keys.add(key)
                        # Timeouts and connection errors: don't add to failed sets
                        # (backend may recover, or other slot may be free)
                        err = f"{backend['name']}@{backend['host']}: {e}"
                        print(f"[caproute] FAIL {err}")
                        errors.append(err)
                    finally:
                        _dec_in_flight(key)

                # All backends failed for current capability. Escalate.
                cfg = load_config()
                fallbacks = _get_fallbacks(cfg)
                if (
                    actual_cap in fallbacks
                    and fallbacks[actual_cap][0] not in visited_caps_this_round
                ):
                    next_cap, hop_overrides = fallbacks[actual_cap]
                    if hop_overrides:
                        accumulated_overrides = {
                            **accumulated_overrides,
                            **hop_overrides,
                        }
                    current_cap = next_cap
                    ov_str = (
                        f" overrides={accumulated_overrides}"
                        if accumulated_overrides
                        else ""
                    )
                    print(
                        f"[caproute] all backends failed for {actual_cap}, mid-flight escalation -> {current_cap}{ov_str}..."
                    )
                    continue
                else:
                    break

            # End of one full sweep through capability chain.
            # Loop back for another round with fresh scores.
            if time.time() < _total_deadline:
                wait = min(2.0, _total_deadline - time.time())
                print(
                    f"[caproute] round {_round} exhausted, retrying in {wait:.0f}s ({_total_deadline - time.time():.0f}s left)..."
                )
                time.sleep(wait)

        self._send_json(
            {
                "error": f"all backends failed for '{model_field}'",
                "tried": errors[-10:],  # last 10 errors to avoid huge payloads
                "rounds": _round,
                "elapsed": round(time.time() - (_total_deadline - timeout)),
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

    # Initialize persistent history store + rehydrate ring buffer
    _db_init()
    if _db_conn is not None:
        recent = _db_load_recent(limit=_request_history.maxlen)
        with _history_lock:
            _request_history.extend(recent)
        print(
            f"[caproute] DB: {_DB_PATH} (retention {_DB_RETENTION_DAYS}d, loaded {len(recent)} recent entries)"
        )
        threading.Thread(target=_db_prune_loop, daemon=True).start()
    else:
        print(f"[caproute] DB: disabled (init failed)")

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

    # Internal stuck-request watchdog — no network traffic, pure in-process state
    def _stuck_watchdog_loop():
        """Restart caproute if any backend request has been in-flight for >STUCK_THRESHOLD seconds.

        This catches cases where a backend connection hangs beyond its timeout
        (e.g. TCP socket not timing out as expected, or a thread pool exhaustion).
        No inference requests are made — uses only internal _in_flight_since state.
        """
        STUCK_THRESHOLD = int(os.environ.get("CAPROUTE_STUCK_THRESHOLD", "300"))
        CHECK_INTERVAL = 30
        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                now = time.time()
                with _in_flight_lock:
                    snapshot = {k: list(v) for k, v in _in_flight_since.items() if v}
                stuck = {
                    k: round(now - ts_list[0], 0)
                    for k, ts_list in snapshot.items()
                    if (now - ts_list[0]) > STUCK_THRESHOLD
                }
                if stuck:
                    details = ", ".join(f"{k}={int(age)}s" for k, age in stuck.items())
                    print(
                        f"[caproute] STUCK requests detected ({details}), "
                        f"threshold={STUCK_THRESHOLD}s — force-closing backend connections"
                    )
                    # Close the stuck socket(s) for each affected backend.
                    # This unblocks the waiting thread (raises socket.error),
                    # which causes CapRoute to mark that backend as failed and
                    # retry the request on the next available backend automatically.
                    with _in_flight_lock:
                        conns_to_close = {
                            k: list(v) for k, v in _active_conns.items() if k in stuck
                        }
                    for bkey, conns in conns_to_close.items():
                        for conn in conns:
                            try:
                                if conn.sock:
                                    conn.sock.close()
                                    print(f"[caproute] force-closed socket for {bkey}")
                            except Exception as ce:
                                print(f"[caproute] close error for {bkey}: {ce}")
            except Exception as e:
                print(f"[caproute] stuck-watchdog error: {e}")

    w = threading.Thread(target=_stuck_watchdog_loop, daemon=True)
    w.start()
    stuck_threshold = int(os.environ.get("CAPROUTE_STUCK_THRESHOLD", "300"))
    print(f"[caproute] Stuck-request watchdog: threshold={stuck_threshold}s, check every 30s")

    # Background config sync thread
    sync_peers = cfg.get("sync_peers", [])
    if sync_peers:
        s = threading.Thread(target=_sync_loop, daemon=True)
        s.start()
        print(
            f"[caproute] Config sync: {len(sync_peers)} peers, interval {SYNC_INTERVAL}s"
        )
    else:
        print(f"[caproute] Config sync: disabled (no sync_peers in config)")

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
    print(
        f"[caproute] Listening on http://0.0.0.0:{args.port} (probe-routing, 32 workers)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[caproute] Shutting down.")


if __name__ == "__main__":
    main()
