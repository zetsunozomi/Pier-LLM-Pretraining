#!/usr/bin/env bash
# Submit only unfinished arms. Re-running this command resumes the same campaign.
set -euo pipefail
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PIER_ROOT="${PIER_ROOT:-$script_root}"
cd "$PIER_ROOT"
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PIER_CAPACITY_DIR="${PIER_CAPACITY_DIR:-$PIER_ROOT/out/capacity40-depth-v1}"
minutes="${PIER_CAPACITY_MINUTES:-60}"
case "$minutes" in
    30|60) ;;
    *) echo 'PIER_CAPACITY_MINUTES must be 30 or 60.' >&2; exit 2 ;;
esac
export PIER_CAPACITY_BUDGET_SECONDS="$((minutes * 60 - 300))"
concurrency="${PIER_CAPACITY_CONCURRENCY:-1}"
[[ "$concurrency" =~ ^[1-7]$ ]] || { echo 'PIER_CAPACITY_CONCURRENCY must be 1..7.' >&2; exit 2; }
mkdir -p "$PIER_CAPACITY_DIR"
export PIER_CAPACITY_DIR="$(cd "$PIER_CAPACITY_DIR" && pwd)"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
indices=$("$PIER_PYTHON" - "$@" <<'PY'
import json, os, sys
from pathlib import Path
from experiments.qwen.capacity_config import ARMS, DEFAULT_ARMS
requested = sys.argv[1:] or DEFAULT_ARMS
unknown = set(requested) - set(ARMS)
if unknown:
    raise SystemExit(f'Unknown arms: {sorted(unknown)}. Choose {list(ARMS)}')
indices = []
for arm in dict.fromkeys(requested):
    path = Path(os.environ['PIER_CAPACITY_DIR']) / arm / 'summary.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        continue
    indices.append(str(list(ARMS).index(arm)))
print(','.join(indices))
PY
)
if [[ -z "$indices" ]]; then
    exec "$PIER_PYTHON" experiments/qwen/capacity.py --output-dir "$PIER_CAPACITY_DIR" --summary
fi
if [[ "$minutes" == 60 ]]; then limit=01:00:00; else limit=00:30:00; fi
printf 'Campaign: %s\nArms: array %s; %s per allocation\n' "$PIER_CAPACITY_DIR" "$indices" "$limit"
sbatch --export=ALL --time="$limit" --array="$indices%$concurrency" \
    --output="$PIER_CAPACITY_DIR/slurm-%A_%a.out.txt" experiments/qwen/capacity.sbatch
