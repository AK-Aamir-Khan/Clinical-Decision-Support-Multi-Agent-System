@echo off
REM ===========================================================================
REM  stop_medagent.bat - stop MedAgent-CDSS started by start_medagent.bat
REM
REM  Stops ONLY the processes the launcher started (recorded in
REM  .medagent_launcher.json): the Streamlit app, and Ollama only if the
REM  launcher itself started it. Other Python or Ollama processes are untouched.
REM ===========================================================================
setlocal
title Stop MedAgent-CDSS

REM --- Move to the project folder ---
cd /d "%~dp0"

REM --- Activate the virtual environment ---
if not exist ".venv\Scripts\activate.bat" (
    echo [MedAgent] ERROR: .venv not found in "%CD%".
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

REM --- Stop the launcher's processes ---
set PYTHONIOENCODING=utf-8
python scripts\launcher.py --stop
pause
