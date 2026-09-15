@echo off
setlocal

REM -----------------------------------------
REM Activate pyenv local Python 3.11
REM -----------------------------------------
pyenv local 3.11.8

REM -----------------------------------------
REM Create venv if missing
REM -----------------------------------------
if not exist venv (
    echo Creating Python 3.11 virtual environment...
    python -m venv venv
)

REM -----------------------------------------
REM Activate venv
REM -----------------------------------------
call venv\Scripts\activate

REM -----------------------------------------
REM Install requirements
REM -----------------------------------------
echo Installing dependencies...
pip install --upgrade pip
pip install -r ..\requirements.txt

REM -----------------------------------------
REM Run the brain visualizer
REM -----------------------------------------
python example.py

endlocal
pause