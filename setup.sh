#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VOICEVOX_CORE_VERSION="0.16.3"
DOWNLOADER_URL="https://github.com/VOICEVOX/voicevox_core/releases/download/${VOICEVOX_CORE_VERSION}/download-linux-x64"
WHL_URL="https://github.com/VOICEVOX/voicevox_core/releases/download/${VOICEVOX_CORE_VERSION}/voicevox_core-${VOICEVOX_CORE_VERSION}-cp310-abi3-manylinux_2_34_x86_64.whl"

# Create and activate venv
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install voicevox_core from whl
echo "Installing voicevox_core..."
pip install "$WHL_URL"

# Download ONNX Runtime, Open JTalk dict, and VVM models
if [ ! -f download-linux-x64 ]; then
    echo "Downloading voicevox_core downloader..."
    curl -sSLf "$DOWNLOADER_URL" -o download-linux-x64
    chmod +x download-linux-x64
fi

if [ ! -d voicevox_core ]; then
    echo "Downloading ONNX Runtime, Open JTalk dict, and voice models..."
    ./download-linux-x64 --devices cuda --output voicevox_core
fi

echo ""
echo "Setup complete. To start the server:"
echo "  source .venv/bin/activate"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
