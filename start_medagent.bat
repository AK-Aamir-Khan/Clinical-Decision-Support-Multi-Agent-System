@echo off
REM ===========================================================================
REM  start_medagent.bat - one-click start for MedAgent-CDSS (Windows)
REM
REM  Double-click this file. It will:
REM    1. go to the project folder (wherever this file is located)
REM    2. activate the .venv virtual environment
REM    3. detect Ollama (start it only if it is not already running)
REM    4. make sure the gemma3:4b model is available (pull it if missing)
REM    5. start the Streamlit app:  streamlit run app/main.py
REM    6. open http://localhost:8501 in your browser once the app is ready
REM
REM  Steps 3-6 are done by scripts\launcher.py, which reads the Ollama URL and
REM  model name from your .env (nothing is hard-coded here).
REM  Keep this window open while using the app; press Ctrl+C or run
REM  stop_medagent.bat to stop. Optional: start_medagent.bat --port 8502
REM ===========================================================================
setlocal
title MedAgent-CDSS

REM --- 1. Move to the project folder (the folder containing this .bat file) ---
cd /d "%~dp0"

REM --- 2. Check and activate the virtual environment ---
if not exist ".venv\Scripts\activate.bat" (
    echo [MedAgent] ERROR: .venv not found in "%CD%".
    echo [MedAgent] Create it once with:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    goto :failed
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [MedAgent] ERROR: could not activate .venv
    goto :failed
)

REM --- Check that the project's packages are installed in .venv ---
python -c "import streamlit, langgraph, langchain_openai" 1>nul 2>nul
if errorlevel 1 (
    echo [MedAgent] ERROR: required packages are missing in .venv.
    echo [MedAgent] Install them with:  pip install -r requirements.txt
    goto :failed
)

REM --- 3-6. Ollama + gemma3:4b check, start the app, open the browser ---
REM  Make Python print UTF-8 so log messages display correctly in this window.
set PYTHONIOENCODING=utf-8
python scripts\launcher.py %*
if errorlevel 1 goto :failed

echo.
echo [MedAgent] MedAgent-CDSS has stopped.
pause
exit /b 0

:failed
echo.
echo [MedAgent] Startup failed - read the messages above.
echo [MedAgent] Manual start:  .venv\Scripts\activate  then  streamlit run app/main.py
pause
exit /b 1
