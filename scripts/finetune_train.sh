#!/bin/bash
# Step 2 of the optional fine-tune (after scripts/finetune_run.sh scored the released model): under the heavy lock,
# pause the Kev server so Atlas stays under its 48 GB footprint, train the LoRA delta, score it on the same held-out
# 20%, then restart the Kev server (released checkpoint; swapping is a separate, measured decision).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FT="$ROOT/data/finetune"; RUN="$ROOT/vendor/kev/runs/atlas-deals"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
exec 9>/tmp/atlas-heavy.lock
flock -w 900 9 || { echo "could not take /tmp/atlas-heavy.lock"; exit 1; }
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "$(date +%T) lock held; MemAvailable ${avail} GB"
[ "$avail" -lt 30 ] && { echo "MemAvailable < 30 GB: skipping"; exit 0; }
"$ROOT/scripts/svc.sh" stop kev
cd "$ROOT/vendor/kev"
echo "$(date +%T) training"
.venv/bin/python -m kev.train --data "$FT/train.jsonl" --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --seed 42 --out "$RUN"
echo "$(date +%T) train exit $?"
.venv/bin/python -m kev.benchmark --run "$RUN" --data "$FT/heldout.jsonl" --out runs/atlas-ft-after --device cuda
echo "$(date +%T) benchmark exit $?"
"$ROOT/scripts/svc.sh" start kev
echo "$(date +%T) done"
