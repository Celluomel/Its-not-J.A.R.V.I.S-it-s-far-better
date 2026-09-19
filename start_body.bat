@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
if not exist body_venv\Scripts\python.exe (
  echo [*] Creating the independent Body virtual environment...
  py -3.11 -m venv body_venv
  if errorlevel 1 py -3 -m venv body_venv
  if errorlevel 1 (
    echo [ERROR] Could not create body_venv. Install Python 3.11 or newer.
    exit /b 1
  )
)
set "BODY_PYTHON=body_venv\Scripts\python.exe"
"%BODY_PYTHON%" -m pip install -q --upgrade pip
"%BODY_PYTHON%" -m pip install -q -r body_requirements.txt
if not defined BODY_PYTHON (
  echo [ERROR] No Python environment found. Run start.bat once first.
  exit /b 1
)
echo Starting standalone PandoraBOX Body Runtime...
"%BODY_PYTHON%" -m body_runtime_host
endlocal
