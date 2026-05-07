#!/bin/bash
echo "================================================"
echo "    Stock-AIML Local One-Click Runner           "
echo "================================================"

echo "[1] Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    exit 1
fi

echo "[2] Setting up Virtual Environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "[3] Activating Environment and Installing Dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

echo "[4] Starting FastAPI Backend in the background..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

sleep 3

echo "[5] Starting Streamlit Dashboard..."
streamlit run dashboard/app.py

# When Streamlit is stopped (Ctrl+C), kill the backend too
kill $BACKEND_PID
