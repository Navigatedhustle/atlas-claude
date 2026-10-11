#!/bin/bash
# Stop only the processes Atlas started (tracked by pidfiles in run/).
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
for s in ui embed vision kev llm speech; do scripts/svc.sh stop $s; done
