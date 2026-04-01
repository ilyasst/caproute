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
  │                                  │  resolve "powerful" via llm.json │
  │                                  │  try qwen2.5:32b @ server-a     │
  │                                  │─────────────────────────────────>│
  │                                  │<─────────────────────────────────│
  │<─────────────────────────────────│  OpenAI-format response          │
```

If server-a is down, it falls back to the next model/host in the list.

## Config

Create `~/.config/llm.json`:

```json
{
  "models": {
    "qwen2.5:32b":  { "capabilities": ["fast", "powerful"],  "hosts": ["server-a"] },
    "gemma3:4b":    { "capabilities": ["fast", "vision"],    "hosts": ["server-b"] }
  },
  "hosts": {
    "server-a": { "base_url": "http://server-a:11434", "api": "ollama" },
    "server-b": { "base_url": "http://server-b:8080",  "api": "openai" }
  }
}
```

- **capabilities**: arbitrary strings — you define what they mean
- **hosts**: ordered list per model — first is preferred, rest are fallbacks
- **api**: `"ollama"` or `"openai"` — caproute translates between them
- Config is re-read on every request, so you can edit it live

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

If the `model` field matches a real model name instead of a capability, caproute routes to it directly.

### List capabilities

```bash
curl http://localhost:8800/v1/models
```

### Health check

```bash
curl http://localhost:8800/health
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion — put capability in `model` field |
| `/v1/models` | GET | List available capabilities |
| `/health` | GET | Config status and host list |

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
