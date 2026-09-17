@echo off
echo ==========================================
echo  Lumina Cognitive Organism - Windows
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

REM A venv directory can survive after its base Python was removed. Validate
REM the interpreter itself, not just the directory name.
venv\Scripts\python.exe -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [!] Existing venv is broken. Recreating it...
    goto :setup
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate venv. Re-running setup...
    goto :setup
)
set "LUMINA_PYTHON=%~dp0venv\Scripts\python.exe"

REM Quick dep check
"%LUMINA_PYTHON%" -c "import nicegui" >nul 2>&1
if errorlevel 1 (
    echo [!] Missing core dependencies. Installing...
    goto :install_deps
)

goto voicemem_check

REM --- First-time setup ---
:setup
REM Prefer Python 3.11 for the legacy dlib wheel, but do not accept a
REM Microsoft Store alias that exists yet cannot execute in this shell.
py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Python 3.11 is unavailable or cannot execute.
    py -3.14 -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No usable Python 3.11 or 3.14 installation found.
        echo         Install Python from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    set "PYTHON_CMD=py -3.14"
    set "PYTHON_VERSION=3.14"
    echo [*] Using Python 3.14. Optional dlib face recognition will be skipped.
) else (
    set "PYTHON_CMD=py -3.11"
    set "PYTHON_VERSION=3.11"
    echo [*] Using Python 3.11.
)
echo [*] Creating virtual environment with Python %PYTHON_VERSION%...
%PYTHON_CMD% -m venv --clear venv
if errorlevel 1 (
    echo [ERROR] Could not create a working virtual environment.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
set "LUMINA_PYTHON=%~dp0venv\Scripts\python.exe"
"%LUMINA_PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The created virtual environment cannot execute.
    pause
    exit /b 1
)

:install_deps
echo [*] Upgrading pip...
"%LUMINA_PYTHON%" -m pip install --upgrade pip -q

echo [*] Installing a matched PyTorch and torchaudio pair for this Python...
echo     For NVIDIA CUDA, add the matching PyTorch index URL to this command.
echo     AMD ROCm wheels are generally Linux-only; Windows uses the CPU wheel.
"%LUMINA_PYTHON%" -m pip install torch torchaudio -q

echo [*] Pinning numpy and pandas before coqui-tts...
"%LUMINA_PYTHON%" -m pip install "numpy<2.0" "pandas<2.0" -q

echo [*] Pinning transformers (coqui-tts requirement)...
"%LUMINA_PYTHON%" -m pip install "transformers>=4.43.0,<=4.46.2" -q

echo [*] Installing coqui-tts...
"%LUMINA_PYTHON%" -m pip install "coqui-tts==0.25.3" librosa pydub -q

REM Keep the matched torch/torchaudio pair installed above. Reinstalling a
REM historical torchaudio pin here breaks newer Python environments.

echo [*] Installing remaining dependencies...
"%LUMINA_PYTHON%" -m pip install -r requirements.txt -q
echo [OK] Core dependencies installed.

REM --- Optional VoiceMem voice-memory extension ---
REM Installation is requested by config.json or the explicit environment flag.
:voicemem_check
"%LUMINA_PYTHON%" -c "import json,sys; from pathlib import Path; p=Path('config.json'); d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; raise SystemExit(0 if d.get('VOICEMEM_ENABLED', False) else 1)" >nul 2>&1
if /I "%LUMINA_INSTALL_VOICEMEM%"=="1" goto voicemem_install
if errorlevel 1 goto voicemem_done

:voicemem_install
"%LUMINA_PYTHON%" -c "import voicemem" >nul 2>&1
if not errorlevel 1 goto voicemem_done
echo [*] Installing optional VoiceMem requested by config.json...
"%LUMINA_PYTHON%" -m pip install voicemem -q
if errorlevel 1 echo [WARN] VoiceMem install failed. Native memory remains available.

:voicemem_done
if /I "%LUMINA_INSTALL_VOICEMEM%"=="1" echo [i] VoiceMem installation check completed.

REM --- Create data dirs ---
if not exist data\voices    mkdir data\voices
if not exist data\faces     mkdir data\faces
if not exist data\persona   mkdir data\persona
if not exist logs           mkdir logs

REM --- Face Recognition (optional - requires Visual C++ Build Tools) ---
:facecheck
"%LUMINA_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    for /f "delims=" %%V in ('"%LUMINA_PYTHON%" -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))"') do set "PYTHON_VERSION=%%V"
    echo [WARN] Skipping the Python 3.11 dlib wheel on Python %PYTHON_VERSION%.
    echo        Checking the model package separately before startup.
    goto :face_models_check
)
"%LUMINA_PYTHON%" -c "import face_recognition" >nul 2>&1
if not errorlevel 1 (
    echo [OK] face_recognition already installed.
    goto :face_models_check
)

echo.
echo [*] Attempting to install face recognition (requires dlib)...
echo     If this fails, Lumina will run without face recognition.
echo     To fix manually: see FACE_RECOGNITION_INSTALL.txt

REM Try prebuilt dlib wheel for Python 3.11 x64 (no compiler needed)
"%LUMINA_PYTHON%" -c "import dlib" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing prebuilt dlib for Python 3.11...
    "%LUMINA_PYTHON%" -m pip install "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl" -q
    if errorlevel 1 (
        echo [WARN] Prebuilt dlib wheel failed.
        echo        Face recognition will be disabled.
        echo        See FACE_RECOGNITION_INSTALL.txt for manual install steps.
        goto :playwright_check
    )
)

"%LUMINA_PYTHON%" -m pip install "face-recognition>=1.3.0" -q
if errorlevel 1 (
    echo [WARN] face-recognition install failed. Vision will run without face ID.
) else (
    echo [OK] face_recognition installed successfully.
)

REM face_recognition can be importable while its model data is absent. Check
REM the model package separately so an existing partial install is repaired.
:face_models_check
"%LUMINA_PYTHON%" -c "import face_recognition_models" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing face recognition model data...
    REM Prefer the published wheel: the Git source can stop after metadata
    REM preparation without leaving an importable package on Windows.
    REM face_recognition_models uses pkg_resources, which was removed from
    REM newer setuptools releases. Keep the compatibility module installed.
    "%LUMINA_PYTHON%" -m pip install --upgrade "setuptools<81"
    "%LUMINA_PYTHON%" -m pip install --upgrade --force-reinstall --no-deps --only-binary=:all: --index-url https://pypi.org/simple "face-recognition-models"
    if errorlevel 1 (
        echo [WARN] Published model package failed; trying GitHub source...
        "%LUMINA_PYTHON%" -m pip install --upgrade --no-deps "git+https://github.com/ageitgey/face_recognition_models"
    )
)
"%LUMINA_PYTHON%" -c "import face_recognition_models" >nul 2>&1
if errorlevel 1 (
    echo [WARN] face_recognition_models is still unavailable. Vision will run without face ID.
) else (
    for /f "delims=" %%P in ('"%LUMINA_PYTHON%" -c "import face_recognition_models; print(face_recognition_models.__file__)"') do echo [OK] face_recognition_models ready: %%P
)

REM --- Playwright ---
:playwright_check
"%LUMINA_PYTHON%" -c "from playwright.sync_api import sync_playwright" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing Playwright for Research Cortex web search...
    "%LUMINA_PYTHON%" -m pip install playwright -q
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

REM --- Build the new React interface ---
REM The Python server serves frontend\dist at /next/ when this folder exists.
REM Set LUMINA_SKIP_FRONTEND_BUILD=1 to skip this step during development.
if /I not "%LUMINA_SKIP_FRONTEND_BUILD%"=="1" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [WARN] npm was not found. The new interface was not built.
        echo        Install Node.js, then run: cd frontend ^&^& npm install ^&^& npm run build
    ) else (
        if not exist frontend\node_modules (
            echo [*] Installing new interface dependencies...
            pushd frontend
            call npm install --no-audit --no-fund
            if errorlevel 1 (
                echo [WARN] Frontend dependency installation failed. Continuing with the existing UI.
                popd
                goto :frontend_done
            )
            popd
        )
        echo [*] Building new Lumina interface...
        pushd frontend
        call npm run build
        if errorlevel 1 echo [WARN] Frontend build failed. Continuing with the existing UI.
        popd
    )
)
:frontend_done

echo.
echo [*] Starting Lumina on http://localhost:8080
echo [*] New interface: http://localhost:8080/next/
echo [*] Existing interface: http://localhost:8080/
echo [*] Press Ctrl+C to stop.
echo.
"%LUMINA_PYTHON%" app.py
pause
