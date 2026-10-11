#!/bin/bash
# Atlas service launcher. UI binds 127.0.0.1 unless ATLAS_HOST=0.0.0.0 is set (LAN access); models always loopback.
# Usage: scripts/svc.sh start|stop <llm|speech|kev|ui>
# Every process is tracked by a pidfile in run/ so we only ever stop what we started.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p run logs
NS=$(ls -d "$ROOT"/vendor/nemo-speech/nemo-speech-*-linux-aarch64-cuda13 | head -1)
LL="$ROOT/vendor/llama/llama-b11490"
LLC=$(ls -d "$ROOT"/vendor/llama/cudart-llama-* | head -1)
GGUF=$(ls ~/.cache/huggingface/hub/models--ggml-org--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/snapshots/*/*.gguf | head -1)
# Speech models by explicit local path (pulled once with `nemo-speech pull`), so startup never resolves names online.
NSM=~/.cache/nemo-speech/models/nvidia
ASR_GGUF=$(ls $NSM/nemotron-3.5-asr-streaming-0.6b/*/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf | head -1)
CODEC_GGUF=$(ls $NSM/nemo-nano-codec-22khz-1.89kbps-21.5fps/*/*.decoder.f16.gguf | head -1)
TOK_DIR=$(ls -d $NSM/magpie_tts_multilingual_357m/*/tokenizer | head -1)

ours() { kill -0 "$1" 2>/dev/null && tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q "$ROOT"; }
start() {
  local name=$1; shift
  if [ -f $ROOT/run/$name.pid ] && ours "$(cat $ROOT/run/$name.pid)"; then echo "$name already running"; return; fi
  nohup "$@" > $ROOT/logs/$name.log 2>&1 &
  echo $! > $ROOT/run/$name.pid; echo "started $name pid $!"
}
# True only if the pid is alive AND its command line lives in this work dir (guards against stale pidfiles).
ours() { kill -0 "$1" 2>/dev/null && tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q "$ROOT"; }
stop() {
  local name=$1
  if [ -f $ROOT/run/$name.pid ]; then
    local pid=$(cat $ROOT/run/$name.pid)
    if ! ours "$pid"; then rm -f $ROOT/run/$name.pid; echo "$name not running"; return; fi
    kill "$pid" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -9 "$pid" 2>/dev/null; rm -f $ROOT/run/$name.pid; echo "stopped $name"
  fi
}

case "$1:$2" in
  start:llm)
    LD_LIBRARY_PATH="$LL:$LLC" start llm "$LL/llama-server" -m "$GGUF" --host 127.0.0.1 --port 8101 \
      --ctx-size 16384 --parallel 2 --cache-ram 0 -ngl 999 --flash-attn on --threads 8 --jinja \
      --chat-template-kwargs '{"enable_thinking":false}' --alias nemotron-3.5-lightning --metrics ;;
  start:speech)
    LD_PRELOAD="$ROOT/vendor/shim/libigpu_shim.so" LD_LIBRARY_PATH="$NS/lib" start speech "$NS/bin/nemo-speech" serve \
      --port 8110 --asr-model "$ASR_GGUF" --tts-model "$ROOT/vendor/magpie.v2607.q8_0.gguf" \
      --codec-model "$CODEC_GGUF" --tokenizer-dir "$TOK_DIR" --tn-model-dir "$ROOT/vendor/tn/tn_configs" \
      --no-ui --device cuda ;;
  start:kev)
    cd "$ROOT/vendor/kev" && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 start kev "$ROOT/vendor/kev/.venv/bin/python" -m kev.serve \
      --run jaredpalmer/kev-4b --port 8120; cd "$ROOT" ;;
  start:vision)
    cd "$ROOT/vendor/vision" && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 start vision "$ROOT/vendor/vision/.venv/bin/python" "$ROOT/atlas/vision_server.py" --port 8130; cd "$ROOT" ;;
  start:embed)
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 start embed "$ROOT/vendor/kev/.venv/bin/python" "$ROOT/atlas/embed_server.py" --port 8140 ;;
  start:ui)
    start ui "$ROOT/.venv/bin/python" -m uvicorn atlas.server:app --host "${ATLAS_HOST:-127.0.0.1}" --port 8100 --log-level warning ;;
  stop:*) stop "$2" ;;
  *) echo "usage: $0 start|stop llm|speech|kev|ui"; exit 1 ;;
esac
