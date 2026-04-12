#!/bin/bash
# Ollama runner watchdog — detects stuck runners and kills them.
# Ollama automatically respawns a fresh runner on the next request.
#
# Install on each Ollama fleet machine via cron:
#   crontab -e
#   */2 * * * * /path/to/ollama-watchdog.sh >> ~/logs/ollama-watchdog.log 2>&1
#
# How it works:
#   1. For each model currently loaded (via /api/ps), send a tiny inference
#   2. If inference hangs (15s timeout), find and kill the runner process
#   3. Ollama detects the dead runner and spawns a new one on next request
#
# Why runners get stuck:
#   llama.cpp runners occasionally hang with established TCP connections
#   that never complete. Health endpoint still returns OK. Ollama doesn't
#   detect this because its liveness check doesn't test actual inference.

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
TIMEOUT=15
LOG_PREFIX="[ollama-watchdog] $(date '+%Y-%m-%d %H:%M:%S')"

# Get loaded models
MODELS=$(curl -sf --max-time 5 "$OLLAMA_HOST/api/ps" 2>/dev/null \
  | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    for m in d.get('models', []):
        print(m['name'])
except: pass
" 2>/dev/null)

if [ -z "$MODELS" ]; then
  # No models loaded or ollama not responding — nothing to watch
  exit 0
fi

for MODEL in $MODELS; do
  # Try a tiny inference
  RESULT=$(curl -sf --max-time "$TIMEOUT" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    "$OLLAMA_HOST/v1/chat/completions" 2>/dev/null)

  if [ $? -ne 0 ]; then
    echo "$LOG_PREFIX $MODEL: inference hung (${TIMEOUT}s timeout). Killing runner."

    # Find the runner process for this model.
    # Ollama runners have "--alias <model>" or "--model <path>" in their args.
    # Match on the alias which is the model name.
    RUNNER_PID=$(ps aux | grep "ollama runner" | grep -v grep \
      | grep -- "--alias $MODEL\|--model.*$(echo "$MODEL" | cut -d: -f1)" \
      | awk '{print $2}' | head -1)

    if [ -n "$RUNNER_PID" ]; then
      echo "$LOG_PREFIX killing runner PID $RUNNER_PID for $MODEL"
      kill "$RUNNER_PID" 2>/dev/null
      sleep 1
      # Force kill if still alive
      kill -0 "$RUNNER_PID" 2>/dev/null && kill -9 "$RUNNER_PID" 2>/dev/null
    else
      echo "$LOG_PREFIX no runner PID found for $MODEL — ollama may need full restart"
    fi
  fi
done
