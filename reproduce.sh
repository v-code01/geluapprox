#!/usr/bin/env bash
# Regenerate the geluapprox study. Self-contained (deterministic input; no model/data). Usage: ./reproduce.sh [REPS]
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
make >/dev/null
bin/bench "${1:-400}" > results/bench.jsonl
python tools/analyze.py
python tools/verify.py
echo "regenerated; see bench_results/frontier.md"
