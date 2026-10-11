#!/bin/bash
# Start the whole Atlas stack from cold. UI: http://localhost:8100
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
mkdir -p run logs out
[ -f data/atlas.db ] || .venv/bin/python -m atlas.data.generate --seed 42
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable: ${avail} GB"
if [ "$avail" -lt 24 ]; then echo "WARNING: only ${avail} GB available; a cold Atlas start needs ~46 GB of which some may already be loaded."; fi
scripts/svc.sh start speech
scripts/svc.sh start llm
scripts/svc.sh start kev
[ -x vendor/vision/.venv/bin/python ] && scripts/svc.sh start vision
[ "${ATLAS_DOCS:-1}" != "0" ] && scripts/svc.sh start embed
# Second brain: build the synthetic corpus, vector index and knowledge graphs if missing (embedding service needed).
if [ "${ATLAS_DOCS:-1}" != "0" ]; then
  for i in $(seq 1 90); do curl -sf localhost:8140/health | grep -q true && break; sleep 1; done
  .venv/bin/python -c "from atlas.knowledge import docs; [docs.ensure(p) for p in ('enterprise','university')]" || echo "WARNING: document index not built (search_knowledge falls back to keywords)"
fi
[ "${ATLAS_BRAIN:-1}" != "0" ] && .venv/bin/python -c "from atlas.knowledge import graph as G; [G.load(p) for p in ('enterprise','university')]"
scripts/svc.sh start ui
echo -n "Waiting for services"
for i in $(seq 1 180); do
  ok=1
  curl -sf localhost:8110/health >/dev/null || ok=0
  curl -sf localhost:8101/health >/dev/null || ok=0
  curl -sf localhost:8100/api/config >/dev/null || ok=0
  [ $ok = 1 ] && break
  echo -n "."; sleep 1
done
echo
echo -n "Waiting for System One (Kev-4B) and vision"
for i in $(seq 1 150); do
  curl -sf localhost:8120/v1/models >/dev/null && { [ ! -x vendor/vision/.venv/bin/python ] || curl -sf localhost:8130/health | grep -q true; } && break
  echo -n "."; sleep 1
done
echo
curl -sf localhost:8120/v1/models >/dev/null && echo "System One (Kev-4B): up" || echo "System One (Kev-4B): still loading (UI falls back to LLM option probabilities)"
curl -sf localhost:8130/health 2>/dev/null | grep -q true && echo "Vision (RF-DETR Nano): up" || echo "Vision: not running (camera tile shows the feed without boxes)"
curl -sf localhost:8140/health 2>/dev/null | grep -q true && echo "Embeddings (Nemotron-3-Embed-1B): up" || echo "Embeddings: not running (document search uses keyword fallback)"
curl -sf localhost:8100/api/config >/dev/null && echo "Atlas is up: http://127.0.0.1:8100 (bound to ${ATLAS_HOST:-127.0.0.1})" || { echo "UI failed; see logs/ui.log"; exit 1; }
