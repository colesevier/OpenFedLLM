#!/bin/bash
set -e

# Detect OS
OS="$(uname -s)"

# Check for Python 3.12
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "Python 3.12 is recommended for this project."
    echo "If you use pyenv, run: brew install pyenv pyenv-virtualenv"
    echo "Then: pyenv install 3.12.3 && pyenv virtualenv 3.12.3 openfedllm-3.12"
    echo "And: pyenv activate openfedllm-3.12"
    echo "Or: python3.12 -m venv venv && source venv/bin/activate"
    exit 1
fi

# Create and activate venv if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Creating Python 3.12 virtual environment..."
    python3.12 -m venv venv
    source venv/bin/activate
fi

if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS, installing requirements..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
elif [[ "$OS" == "Linux" ]]; then
    echo "Detected Linux, installing requirements..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo "Installation complete. Activate your venv with: source venv/bin/activate"