#!/usr/bin/env bash
# One-shot setup for Ubuntu 22.04/24.04 (native or WSL2).
set -euo pipefail

sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential \
    git curl wget ffmpeg tesseract-ocr tesseract-ocr-tel tesseract-ocr-hin espeak-ng

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
mkdir -p input/images input/videos output models

echo "Core setup finished. Optional next steps:"
echo "  pip install -r requirements-detect.txt   # YOLO detection (large download)"
echo "  curl -fsSL https://ollama.com/install.sh | sh   # inspect the script first"
echo "  ollama pull llama3.1:8b"
