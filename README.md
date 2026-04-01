# caproute

Capability-routing LLM gateway. Ask for what you need (`"powerful"`, `"fast"`, `"thinking"`, `"vision"`) — caproute finds the right model and host.

Speaks OpenAI API. Zero dependencies (stdlib Python only). Designed for self-hosted LLM setups across multiple machines (e.g. Tailscale).

## How it works

```
Client                          caproute (:8800)                    Backends
  │                                  │                                  │
  │ POST /v1/chat/completions        │                                  │
  │ {"model": "powerful", ...}       │                                  │
  │─────────────────────────────────>│                                  │
  │                                  │  resolve "powerful" via config   │
  │                                  │  discover qwen2.5:32b @ host-a  │
  │                                  │─────────────────────────────────>│
  │                                  │<─────────────────────────────────│
  │<─────────────────────────────────│  OpenAI-format response          │
```

If host-a is down, it falls back to the next model/host that has a "powerful" model.

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
const r = await client.chat.completions.create({ model: "fast", messages: [{ role: "user", content: "Hi" }] });
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
```

The `/health` endpoint shows **untagged models** — models discovered on your hosts that aren't classified under any capability yet. Useful for spotting new models you forgot to tag.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion — put capability in `model` field |
| `/v1/models` | GET | List available capabilities |
| `/health` | GET | Config status, untagged models |
| `/discovery` | GET | Full model-to-host-to-capability map |
| `/discovery/refresh` | POST | Force immediate re-discovery |

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

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPROUTE_PORT` | `8800` | Listen port |
| `CAPROUTE_CONFIG` | `~/.config/llm.json` | Config file path |
| `CAPROUTE_TIMEOUT` | `300` | Backend request timeout (seconds) |
| `CAPROUTE_DISCOVERY_INTERVAL` | `60` | Seconds between host discovery scans |
