#!/bin/bash
# Optional Kev-4B fine-tune on Meridian deal history (Commit-stage close won?). Runs only under the shared heavy lock
# and only if MemAvailable >= 30 GB once the lock is held. Writes data/brain/finetune.json with measured results.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/vendor/kev"
OUT="$ROOT/data/brain/finetune.json"; FT="$ROOT/data/finetune"; RUN="$ROOT/vendor/kev/runs/atlas-deals"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
exec 9>/tmp/atlas-heavy.lock
flock -w 900 9 || { echo "could not take /tmp/atlas-heavy.lock"; exit 1; }
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "lock held; MemAvailable ${avail} GB"
if [ "$avail" -lt 30 ]; then
  echo "{\"status\": \"skipped\", \"reason\": \"MemAvailable ${avail} GB < 30 GB with the heavy lock held\", \"ui_text\": \"Not fine-tuned: general calibration\"}" > "$OUT"; exit 0
fi
.venv/bin/python -m kev.benchmark --remote http://127.0.0.1:8120 --data "$FT/heldout.jsonl" --out runs/atlas-ft-before || exit 1
.venv/bin/python -m kev.train --data "$FT/train.jsonl" --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --seed 42 --out "$RUN" || exit 1
.venv/bin/python -m kev.benchmark --run "$RUN" --data "$FT/heldout.jsonl" --out runs/atlas-ft-after --device cuda || exit 1
echo "done; reports in runs/atlas-ft-before and runs/atlas-ft-after"
