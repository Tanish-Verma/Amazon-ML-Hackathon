#!/usr/bin/env bash
# Reproducible environment setup for the entity-resolution pipeline.
#
# Targets the machine this was developed on: Python 3.11, CUDA driver 535
# (CUDA 12.2), no system-wide installs. Creates an isolated venv under the
# project root and installs pinned dependencies into it.
#
# Usage:  bash src/bootstrap.sh [PROJECT_ROOT]
set -euo pipefail

ROOT="${1:-$HOME/amlc}"
VENV="$ROOT/venv"
PY="${PYTHON:-/usr/local/bin/python3}"

echo "==> project root : $ROOT"
echo "==> python       : $PY ($($PY -V 2>&1))"

mkdir -p "$ROOT"
if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel -q

# Core pipeline: columnar I/O, numerics, the gradient-boosted matcher, and a
# fast C++ string-similarity library (rapidfuzz) for the pairwise features.
python -m pip install -q \
    "numpy==2.1.3" "pandas==2.2.3" "pyarrow==18.1.0" "scipy==1.14.1" \
    "scikit-learn==1.5.2" "lightgbm==4.5.0" "rapidfuzz==3.10.1" \
    "regex==2024.11.6" "tqdm==4.67.1" "psutil==6.1.0"

echo "==> core install complete"
python - <<'PY'
import numpy, pandas, pyarrow, sklearn, lightgbm, rapidfuzz
print(f"  numpy {numpy.__version__} | pandas {pandas.__version__} | pyarrow {pyarrow.__version__}")
print(f"  sklearn {sklearn.__version__} | lightgbm {lightgbm.__version__} | rapidfuzz {rapidfuzz.__version__}")
PY
echo "==> done. activate with: source $VENV/bin/activate"
