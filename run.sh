#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate
CORE_DIR="$SCRIPT_DIR/voicevox_core"
export LD_LIBRARY_PATH="$CORE_DIR/onnxruntime/lib:$CORE_DIR/additional_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uvicorn main:app --host 0.0.0.0 --port 8080
