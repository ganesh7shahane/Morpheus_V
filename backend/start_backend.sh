#!/usr/bin/env bash
# ===========================================================================
# start_backend.sh – auto-setup Python venv and launch the FastAPI backend.
#
# Called by "npm run dev" via package.json.
# Safe to run repeatedly: deps are only re-installed when requirements.txt
# changes (detected via mtime of a sentinel file).
# ===========================================================================
set -euo pipefail

# Directory of this script (works even when called from the repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SENTINEL="$VENV_DIR/.deps_ok"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 1. Find a suitable Python (≥3.10)
# ---------------------------------------------------------------------------
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ok=$("$cmd" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null || echo "False")
            if [ "$ok" = "True" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    # Fall back: check common conda/pyenv install roots on macOS
    for root in \
        "$HOME/miniconda3" \
        "$HOME/anaconda3" \
        "$HOME/opt/miniconda3" \
        "$HOME/opt/anaconda3" \
        "/opt/homebrew/anaconda3" \
        "/opt/homebrew/miniconda3" \
        "$HOME/.pyenv/versions/anaconda3-2023.09-0"; do
        cand="$root/bin/python3"
        if [ -x "$cand" ]; then
            ok=$("$cand" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null || echo "False")
            if [ "$ok" = "True" ]; then
                echo "$cand"
                return 0
            fi
        fi
    done
    echo ""
}

BASE_PYTHON=$(find_python)
if [ -z "$BASE_PYTHON" ]; then
    echo ""
    echo "❌  ERROR: Could not find Python 3.10+."
    echo "   Install it with one of:"
    echo "     brew install python"
    echo "     conda install python=3.12"
    echo ""
    exit 1
fi

echo "🐍  Python: $BASE_PYTHON  ($("$BASE_PYTHON" --version))"

# ---------------------------------------------------------------------------
# 2. Create virtual environment if it doesn't exist yet
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "📦  Creating virtual environment at backend/.venv …"
    "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ---------------------------------------------------------------------------
# 3. Install / update dependencies when requirements.txt is newer than sentinel
# ---------------------------------------------------------------------------
needs_install=false
if [ ! -f "$SENTINEL" ]; then
    needs_install=true
elif [ "$REQUIREMENTS" -nt "$SENTINEL" ]; then
    needs_install=true
fi

if $needs_install; then
    echo "⬇️   Installing Python dependencies (this only happens once) …"
    "$PIP" install --quiet --upgrade pip

    # rdkit: try the modern wheel name; fall back to the legacy rdkit-pypi name
    if ! "$PIP" install --quiet rdkit 2>/dev/null; then
        echo "    ↳ 'rdkit' wheel not found, trying 'rdkit-pypi' …"
        "$PIP" install --quiet rdkit-pypi
    fi

    # Install the rest of the requirements (rdkit line already handled by the
    # explicit install above; skip comments and blank lines too)
    TMP_REQ=$(mktemp /tmp/morpheus_req_XXXXXX.txt)
    grep -v -i "^rdkit" "$REQUIREMENTS" \
        | grep -v "^#" \
        | grep -v "^[[:space:]]*$" \
        > "$TMP_REQ" || true   # grep non-zero exit just means no lines matched
    if [ -s "$TMP_REQ" ]; then
        "$PIP" install --quiet -r "$TMP_REQ"
    fi
    rm -f "$TMP_REQ"

    touch "$SENTINEL"
    echo "✅  All dependencies installed."
fi

# ---------------------------------------------------------------------------
# 4. Verify critical imports before starting (friendly error messages)
# ---------------------------------------------------------------------------
"$PY" - <<'PYCHECK'
import sys, importlib

missing = []
for pkg, import_name in [
    ("fastapi",    "fastapi"),
    ("uvicorn",    "uvicorn"),
    ("rdkit",      "rdkit"),
    ("Pillow",     "PIL"),
    ("requests",   "requests"),
    ("espsim",     "espsim"),
    ("scipy",      "scipy"),
    ("sklearn",    "sklearn"),
]:
    try:
        importlib.import_module(import_name)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"\n❌  Missing packages: {', '.join(missing)}")
    print("   Delete backend/.venv and re-run npm run dev to reinstall.\n")
    sys.exit(1)

print("✅  All imports OK")
PYCHECK

# ---------------------------------------------------------------------------
# 5. Kill any stale process on port 8000 before starting
# ---------------------------------------------------------------------------
STALE_PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    echo "⚠️   Port 8000 is in use (PID $STALE_PID). Killing stale process …"
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 0.5
fi

# ---------------------------------------------------------------------------
# 6. Launch uvicorn with the venv's interpreter
# ---------------------------------------------------------------------------
echo "🚀  Starting FastAPI backend → http://localhost:8000"
echo ""
cd "$SCRIPT_DIR"
exec "$VENV_DIR/bin/uvicorn" main:app --reload --port 8000
