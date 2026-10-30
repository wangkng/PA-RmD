#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-}"
case "$EXPERIMENT" in
  synthetic|youtube) shift ;;
  *) echo "Usage: bash scripts/siar.bash {synthetic|youtube} [options]" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
exec "${PYTHON:-python}" "$ROOT/main.py" "$EXPERIMENT" "$@" --method siar
