# caproute

Capability-routing LLM gateway. Ask for what you need (`"powerful"`, `"fast"`, `"thinking"`, `"vision"`) — caproute finds the right model and host.

Speaks OpenAI API. Zero dependencies (stdlib Python only). Designed for self-hosted LLM setups across multiple machines (e.g. Tailscale).

## How it works

caproute uses **probe-routing**: a background thread continuously pings every backend every 5 seconds to know their real-time health. Routing goes directly to the best available backend — no wasted timeouts on dead or overloaded ones.

```
Client                          caproute (:8800)                    Backends
  │                                  │                                  │
  │ POST /v1/chat/completions        │                                  │
  │ {"model": "powerful", ...}       │                                  │
  │─────────────────────────────────>│                                  │
  │                                  │  check probe scores              │
  │                                  │  route to best backend directly  │
  │                                  │─────────────────────────────────>│
  │                                  │<─────────────────────────────────│
  │<─────────────────────────────────│  OpenAI-format response          │
```

### Probe-routing vs blind fallback

**Old approach (blind):** Try backend 1 → wait 22s timeout → fail → try backend 2 → wait 22s → fail → ... → 90s wasted per request.

**New approach (probe-routing):** Background probes already know which backends are alive. Route directly to the best one. If it fails, fail fast and try the next. Typical latency: one backend round-trip, not N timeouts.

### How probes work

- Every 5 seconds, caproute sends a tiny "hi" request to each backend
- **Ollama hosts**: probes only models already loaded in memory (checked via `/api/ps`). This avoids loading models which evicts others — probes must not destroy availability.
- **llama.cpp/OpenAI hosts**: all configured models are pre-loaded, so all are probed
- Each backend gets a **score** combining:
  - Average latency (exponential moving average, alpha=0.3)
  - Consecutive failure penalty (+5000 per failure)
  - In-flight concurrency penalty (+3000 per active request)
- Status: `ok` (<2s avg), `slow` (2-10s), `down` (3+ consecutive failures)
- Down backends get score 999999 — tried only as last resort

### Concurrency tracking

caproute tracks in-flight requests per backend. When a request starts, the counter goes up. When it finishes (success or failure), it goes down. The score includes this count, so busy backends naturally get deprioritized in favor of idle ones.

```
Backend A: avg_latency=1700ms, in_flight=4  → score = 1700 + 0 + 12000 = 13700
Backend B: avg_latency=2300ms, in_flight=0  → score = 2300 + 0 + 0     =  2300  ← picked
```

## Config

Create `~/.config/llm.json` with two sections:

1. **hosts** — your machines (caproute auto-discovers their models)
2. **capabilities** — your classification of which models serve which purpose

```json
{
  "hosts": {
    "server-a": { "url": "http://server-a:11434", "api": "ollama" },
    "server-b": { "url": "http://server-b:8080",  "api": "openai" }
  },
  "capabilities": {
    "powerful":  ["qwen2.5:32b", "qwen2.5:7b"],
    "fast":      ["qwen2.5:32b", "gemma3:4b"],
    "thinking":  ["qwen3.5:27b"],
    "vision":    ["gemma3:4b"]
  }
}
```

- **hosts**: list your machines. `api` is `"ollama"` or `"openai"` depending on what they run.
- **capabilities**: arbitrary strings you define. Map each to the models that qualify. Order matters — first model found wins.
- **Models are auto-discovered** from hosts every 60s. You don't need to say which model is on which host — caproute finds out by querying `/api/tags` (Ollama) or `/v1/models` (OpenAI).
- Config is re-read on every request, so you can edit it live.

See `llm.json.example` for a full example.

## Usage

### Start

```bash
python3 caproute.py              # default port 8800
python3 caproute.py --port 9000  # custom port
```

### Send a request

```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "powerful", "messages": [{"role": "user", "content": "Hello"}]}'
```

### With any OpenAI client library

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8800/v1", api_key="none")
r = client.chat.completions.create(model="powerful", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

```javascript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8800/v1", apiKey: "none" });
const r = await client.chat.completions.create({ model: "fast", messages: [{ role: "user", "content": "Hi" }] });
```

### Direct model names still work

```bash
curl ... -d '{"model": "qwen2.5:7b", "messages": [...]}'
```

If the `model` field matches a real model name instead of a capability, caproute routes to it directly via discovery.

### Check what caproute sees

```bash
# List capabilities (OpenAI /v1/models format)
curl http://localhost:8800/v1/models

# Health: config status, untagged models
curl http://localhost:8800/health

# Full discovery: every model, which hosts serve it, what capabilities it has
curl http://localhost:8800/discovery

# Force re-discovery now
curl -X POST http://localhost:8800/discovery/refresh

# Real-time backend state: scores, status, in-flight, latency
curl http://localhost:8800/backends
```

The `/health` endpoint shows **untagged models** — models discovered on your hosts that aren't classified under any capability yet. Useful for spotting new models you forgot to tag.

The `/backends` endpoint shows the probe-routing state for every backend:

```json
{
  "backends": {
    "peacewalker:qwen2.5:32b": {
      "status": "ok",
      "score": 1000,
      "avg_latency_ms": 1705,
      "failures": 0,
      "in_flight": 2,
      "last_success": 1775102204
    }
  }
}
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion — put capability in `model` field |
| `/v1/models` | GET | List available capabilities |
| `/health` | GET | Config status, untagged models |
| `/discovery` | GET | Full model-to-host-to-capability map |
| `/discovery/refresh` | POST | Force immediate re-discovery |
| `/backends` | GET | Real-time backend state (probe-routing scores, latency, in-flight) |

## Install as systemd service

```bash
# Copy unit file
mkdir -p ~/.config/systemd/user
cp caproute.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now caproute

# Check status
systemctl --user status caproute
journalctl --user -u caproute -f
```

## Session affinity (cache warmth)

caproute tracks which backend recently served each conversation session. Subsequent requests from the same session prefer the same backend, keeping KV cache warm across turns — measured **4.4x speedup** on turn-2+ vs cold starts.

Session identity is derived from the OpenAI `user` field or a hash of the system prompt + first user message. Affinity decays over time:

| Window | Bonus | Effect |
|--------|-------|--------|
| < 5 min (strong) | -40000 score | Firmly pins to same backend |
| 5-15 min (mild) | -25000 score | Moderate preference, allows rebalancing |
| > 15 min | 0 | No preference, fully load-balanced |

Backends that are `down` never receive affinity bonus, ensuring failover still works.

### Introspection

```bash
# Session affinity map + routing stats
curl http://localhost:8800/stats

# Raw routing decision log (last 5000 decisions)
curl http://localhost:8800/stats/routing
```

### Default max_tokens

When clients don't specify `max_tokens`, caproute injects a default of **8192** tokens. This prevents llama-server's OpenAI-compat default of 4096, which truncates thinking models (reasoning can consume 4K+ tokens alone, leaving nothing for the actual answer).

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion — put capability in `model` field |
| `/v1/models` | GET | List available capabilities |
| `/health` | GET | Config status, untagged models |
| `/discovery` | GET | Full model-to-host-to-capability map |
| `/discovery/refresh` | POST | Force immediate re-discovery |
| `/backends` | GET | Real-time backend state (probe-routing scores, latency, in-flight) |
| `/stats` | GET | Session affinity state + aggregate routing counters |
| `/stats/routing` | GET | Raw routing decision log (last 5000 entries) |
| `/history` | GET | Request history (ring buffer or SQLite since=epoch) |
| `/dashboard` | GET | Built-in web dashboard |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPROUTE_PORT` | `8800` | Listen port |
| `CAPROUTE_CONFIG` | `~/.config/llm.json` | Config file path |
| `CAPROUTE_TIMEOUT` | `60` | Backend request timeout (seconds) |
| `CAPROUTE_DISCOVERY_INTERVAL` | `60` | Seconds between host discovery scans |
| `CAPROUTE_PROBE_INTERVAL` | `5` | Seconds between backend health probes |
| `CAPROUTE_PROBE_TIMEOUT` | `8` | Timeout for individual probe requests (seconds) |
