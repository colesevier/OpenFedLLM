#!/bin/bash
set -e

PYTHON_VERSION="3.12.6"
VENV_DIR="venv"

echo "[INFO] Checking pyenv..."
if ! command -v pyenv &> /dev/null; then
  echo "[ERROR] pyenv not found. Please install pyenv first."
  exit 1
fi

echo "[INFO] Installing Python $PYTHON_VERSION with pyenv (if not already)..."
pyenv install -s $PYTHON_VERSION

echo "[INFO] Setting local Python version to $PYTHON_VERSION"
pyenv local $PYTHON_VERSION

# Always wipe and recreate venv
if [ -d "$VENV_DIR" ]; then
  echo "[INFO] Removing old virtual environment..."
  rm -rf $VENV_DIR
fi

echo "[INFO] Creating new virtual environment with Python $PYTHON_VERSION..."
pyenv exec python -m venv $VENV_DIR

echo "[INFO] Activating virtual environment..."
source $VENV_DIR/bin/activate

# --- NEW: Check Python version ---
ACTUAL_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
if [[ $ACTUAL_VERSION != 3.12.* ]]; then
  echo "[ERROR] Virtualenv is using Python $ACTUAL_VERSION but $PYTHON_VERSION is required."
  echo "        Make sure pyenv is properly initialized in your shell."
  deactivate || true
  exit 1
fi
echo "[INFO] Verified Python version $ACTUAL_VERSION"

echo "[INFO] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-mac.txt

echo "[SUCCESS] Setup complete. Run 'source venv/bin/activate' to use this environment."
