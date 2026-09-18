@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BODY_PYTHON=venv\Scripts\python.exe"
)
if not defined BODY_PYTHON if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BODY_PYTHON=.venv\Scripts\python.exe"
)
if not defined BODY_PYTHON (
  echo [ERROR] No Python environment found. Run start.bat once first.
  exit /b 1
)
echo Starting standalone Lumina Body Runtime...
"%BODY_PYTHON%" -m cognition.body_runtime
endlocal
