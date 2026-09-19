@echo off
echo ==========================================
echo  PandoraBOX Cognitive Organism - Windows
echo ==========================================

REM Force UTF-8 so emoji in log messages do not crash the console
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM --- Check for venv ---
if not exist venv (
    echo [!] No venv found. Running first-time setup...
    goto :setup
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate venv. Re-running setup...
    goto :setup
)

REM Quick dep check
python -c "import nicegui" >nul 2>&1
if errorlevel 1 (
    echo [!] Missing core dependencies. Installing...
    goto :install_deps
)

goto :face_recognition_check

REM --- First-time setup ---
:setup
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found.
    echo         Download: https://www.python.org/downloads/release/python-3119/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [*] Creating virtual environment with Python 3.11...
py -3.11 -m venv venv
call venv\Scripts\activate.bat

:install_deps
echo [*] Upgrading pip...
python -m pip install --upgrade pip -q

echo [*] Installing PyTorch 2.5.1 (CPU)...
echo     For CUDA 12.1, edit start.bat and add: --index-url https://download.pytorch.org/whl/cu121
pip install "torch==2.5.1" "torchaudio==2.5.1" -q

echo [*] Pinning numpy and pandas before coqui-tts...
pip install "numpy<2.0" "pandas<2.0" -q

echo [*] Pinning transformers (coqui-tts requirement)...
pip install "transformers>=4.43.0,<=4.46.2" -q

echo [*] Installing coqui-tts...
pip install "coqui-tts==0.25.3" librosa pydub -q

echo [*] Force-pinning torchaudio after coqui (coqui may upgrade it)...
pip install "torchaudio==2.5.1" --force-reinstall -q

echo [*] Installing remaining dependencies...
pip install -r requirements.txt -q
echo [OK] Core dependencies installed.

REM --- Create data dirs ---
if not exist data\voices    mkdir data\voices
if not exist data\faces     mkdir data\faces
if not exist data\persona   mkdir data\persona
if not exist logs           mkdir logs

REM --- Face Recognition (optional - requires Visual C++ Build Tools) ---
:face_recognition_check
python -c "import face_recognition" >nul 2>&1
if not errorlevel 1 (
    echo [OK] face_recognition already installed.
    goto :playwright_check
)

echo.
echo [*] Attempting to install face recognition (requires dlib)...
echo     If this fails, PandoraBOX will run without face recognition.
echo     To fix manually: see FACE_RECOGNITION_INSTALL.txt

REM Try prebuilt dlib wheel for Python 3.11 x64 (no compiler needed)
python -c "import dlib" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing prebuilt dlib for Python 3.11...
    pip install "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl" -q
    if errorlevel 1 (
        echo [WARN] Prebuilt dlib wheel failed.
        echo        Face recognition will be disabled.
        echo        See FACE_RECOGNITION_INSTALL.txt for manual install steps.
        goto :playwright_check
    )
)

pip install "face-recognition>=1.3.0" -q
if errorlevel 1 (
    echo [WARN] face-recognition install failed. Vision will run without face ID.
) else (
    echo [OK] face_recognition installed successfully.
)

REM --- Playwright ---
:playwright_check
python -c "from playwright.sync_api import sync_playwright" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing Playwright for Research Cortex web search...
    pip install playwright -q
    playwright install chromium
    echo [OK] Playwright + Chromium installed.
) else (
    playwright install chromium >nul 2>&1
)

REM --- Run the app ---
:run_app
if not exist config.json (
    echo [*] No config.json found - will be created on first launch.
)

echo.
echo [*] Starting PandoraBOX Brain Only
echo [*] Press Ctrl+C to stop.
echo.
python brain.py --host 0.0.0.0 --port 8765
pause
