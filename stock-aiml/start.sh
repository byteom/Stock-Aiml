#!/bin/bash

# Start FastAPI backend in the background on port 8000
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for backend to be ready
sleep 3

# Start Streamlit dashboard in the foreground on the dynamically assigned PORT
echo "Starting Streamlit dashboard..."
streamlit run dashboard/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
