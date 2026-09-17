#!/usr/bin/env bash
set -e

echo "=========================================="
echo " Lumina Cognitive Organism - Linux Setup"
echo "=========================================="
echo

PYTHON_VERSION="3.11.9"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

###########################################
# Detect OS
###########################################

OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
    DISTRO="$(. /etc/os-release 2>/dev/null && echo "$ID" || echo "unknown")"
else
    DISTRO="unknown"
fi

###########################################
# System dependencies
###########################################

echo "[*] Checking system dependencies..."

install_system_deps() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "[*] Installing system deps via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            build-essential \
            cmake \
            libopenblas-dev \
            liblapack-dev \
            libx11-dev \
            libgtk-3-dev \
            ffmpeg \
            portaudio19-dev \
            libasound2-dev
    elif command -v dnf >/dev/null 2>&1; then
        echo "[*] Installing system deps via dnf..."
        sudo dnf install -y \
            gcc gcc-c++ cmake \
            openblas-devel lapack-devel \
            libX11-devel gtk3-devel \
            ffmpeg portaudio-devel
    elif command -v pacman >/dev/null 2>&1; then
        echo "[*] Installing system deps via pacman..."
        sudo pacman -S --noconfirm \
            base-devel cmake openblas lapack \
            libx11 gtk3 ffmpeg portaudio
    else
        echo "[WARN] Unknown package manager. Install manually:"
        echo "  build-essential cmake libopenblas-dev ffmpeg portaudio19-dev"
    fi
}

# Check for cmake (dlib build requirement)
if ! command -v cmake >/dev/null 2>&1; then
    echo "[!] cmake not found (required for dlib/face-recognition)"
    install_system_deps
else
    echo "[OK] cmake found: $(cmake --version | head -1)"
fi

# Check for ffmpeg (openai-whisper requirement)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[!] ffmpeg not found (required for openai-whisper)"
    install_system_deps
else
    echo "[OK] ffmpeg found"
fi

###########################################
# pyenv
###########################################

if ! command -v pyenv >/dev/null 2>&1; then
    echo "[*] Installing pyenv..."
    curl https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"'    >> ~/.bashrc
    echo 'eval "$(pyenv init -)"'                 >> ~/.bashrc
    eval "$(pyenv init -)"
else
    eval "$(pyenv init -)"
fi

###########################################
# Python 3.11.9
###########################################

if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "[*] Installing Python $PYTHON_VERSION via pyenv..."
    pyenv install "$PYTHON_VERSION"
fi

pyenv local "$PYTHON_VERSION"
echo "[OK] Python: $(python --version)"

###########################################
# Virtual environment
###########################################

if [ ! -d "$PROJECT_DIR/venv" ] || ! "$PROJECT_DIR/venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    if [ -d "$PROJECT_DIR/venv" ]; then
        echo "[!] Existing venv is broken — recreating it..."
        rm -rf "$PROJECT_DIR/venv"
    fi
    echo "[*] Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
echo "[OK] venv activated"

###########################################
# pip upgrade
###########################################

pip install --upgrade pip -q

###########################################
# PyTorch (CPU default — edit for CUDA/ROCm)
###########################################

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[*] Installing a matched PyTorch and torchaudio pair for this Python (CPU)..."
    echo "    For CUDA 12.1, append: --index-url https://download.pytorch.org/whl/cu121"
    echo "    For ROCm 6.2 on Linux, append: --index-url https://download.pytorch.org/whl/rocm6.2"
    pip install torch torchaudio -q
else
    echo "[OK] PyTorch already installed"
fi

###########################################
# Pinned deps (must precede coqui-tts)
###########################################

echo "[*] Pinning numpy, pandas, transformers..."
pip install "numpy<2.0" "pandas<2.0" -q
pip install "transformers>=4.43.0,<=4.46.2" -q

###########################################
# Coqui TTS
###########################################

if ! python -c "import TTS" >/dev/null 2>&1; then
    echo "[*] Installing coqui-tts==0.25.3..."
    pip install "coqui-tts==0.25.3" librosa pydub -q
    # Keep the matched torch/torchaudio pair selected above. A historical
    # torchaudio pin breaks newer Python environments.
    echo "[OK] coqui-tts installed"
else
    echo "[OK] coqui-tts already installed"
fi

###########################################
# Remaining requirements
###########################################

echo "[*] Installing project dependencies..."
pip install -r requirements.txt -q
echo "[OK] Dependencies installed"

###########################################
# Optional VoiceMem
###########################################

if [ "${LUMINA_INSTALL_VOICEMEM:-0}" = "1" ]; then
    echo "[*] Installing optional VoiceMem voice-memory extension..."
    pip install voicemem -q || echo "[WARN] VoiceMem install failed — native memory remains available"
else
    echo "[i] VoiceMem is optional. Set LUMINA_INSTALL_VOICEMEM=1 before setup to install it."
fi

if ! python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
    echo "[*] Installing spaCy English POS model for insight validation..."
    python -m spacy download en_core_web_sm -q
fi
echo "[OK] spaCy POS model ready"

###########################################
# New React interface
###########################################

echo "[*] Preparing the new Lumina interface..."
if ! command -v npm >/dev/null 2>&1; then
    echo "[*] npm not found — installing Node.js and npm..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq nodejs npm
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y nodejs npm
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm nodejs npm
    else
        echo "[ERROR] npm is required to build the new interface."
        echo "        Install Node.js 18+ and npm, then run setup.sh again."
        exit 1
    fi
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "[ERROR] npm installation failed. The new interface cannot be built."
    exit 1
fi

echo "[OK] Node: $(node --version) | npm: $(npm --version)"
npm --prefix "$PROJECT_DIR/frontend" ci --no-audit --no-fund
npm --prefix "$PROJECT_DIR/frontend" run build
echo "[OK] New interface built at frontend/dist"

###########################################
# face-recognition (optional)
###########################################

echo ""
echo "[*] Attempting to install face recognition..."
echo "    (requires cmake and build-essential — already checked above)"

if python -c "import face_recognition" >/dev/null 2>&1; then
    echo "[OK] face_recognition already installed"
else
    if python -c "import dlib" >/dev/null 2>&1; then
        echo "[OK] dlib already installed"
    else
        echo "[*] Building dlib from source (this takes 2-5 minutes)..."
        pip install dlib -q && echo "[OK] dlib built and installed" || {
            echo "[WARN] dlib build failed."
            echo "       Face recognition will be disabled."
            echo "       Try manually: sudo apt install cmake libopenblas-dev && pip install dlib"
        }
    fi

    if python -c "import dlib" >/dev/null 2>&1; then
        pip install "face-recognition>=1.3.0" -q && \
            echo "[OK] face_recognition installed" || \
            echo "[WARN] face-recognition install failed — vision runs without face ID"
    fi
fi

# face_recognition is a wrapper; its model data is a separate package and can
# be missing even when the wrapper import succeeds. Check it independently.
if ! python -c "import face_recognition_models" >/dev/null 2>&1; then
    echo "[*] Installing missing face_recognition_models..."
    pip install --upgrade "setuptools<81" -q
    pip install --upgrade --force-reinstall --no-deps --only-binary=:all: --index-url https://pypi.org/simple "face-recognition-models" -q || \
        pip install --upgrade --no-deps "git+https://github.com/ageitgey/face_recognition_models" -q
fi
if python -c "import face_recognition_models" >/dev/null 2>&1; then
    echo "[OK] face_recognition_models ready"
else
    echo "[WARN] Face model data unavailable — vision runs without face ID"
fi

###########################################
# Playwright
###########################################

if ! python -c "from playwright.sync_api import sync_playwright" >/dev/null 2>&1; then
    echo "[*] Installing Playwright..."
    pip install playwright -q
fi
playwright install chromium
echo "[OK] Playwright + Chromium ready"

###########################################
# Project directories
###########################################

mkdir -p data/voices data/faces data/persona logs
echo "[OK] Data directories created"

###########################################
# Port check
###########################################

PORT=8080
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "[!] Port $PORT in use — switching to 8081"
        PORT=8081
    fi
fi

###########################################
# Launch
###########################################

echo ""
echo "=========================================="
echo " Starting Lumina Cognitive Organism"
echo "=========================================="
echo ""
export ROBOT_PORT="$PORT"
echo "[*] New interface: http://localhost:$PORT/next/"
echo ""

if command -v xdg-open >/dev/null 2>&1; then
    (sleep 2; xdg-open "http://localhost:$PORT/next/" >/dev/null 2>&1) &
fi

python app.py
