@echo off
setlocal
echo ================================================
echo     Stock-AIML Local One-Click Runner           
echo ================================================

echo [1] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10+
    pause
    exit /b
)

echo [2] Setting up Virtual Environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
)

echo [3] Activating Environment and Installing Dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo [4] Starting FastAPI Backend (opens in new window)...
start "Stock-AIML FastAPI Backend" cmd /c "call .venv\Scripts\activate.bat && uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

:: Give backend 3 seconds to start
timeout /t 3 /nobreak >nul

echo [5] Starting Streamlit Dashboard...
streamlit run dashboard\app.py
