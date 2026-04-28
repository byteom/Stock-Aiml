"""FastAPI main application — wires all routes."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import backtest, stress, optimize, explain, data
from backend.api.schemas import HealthResponse

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Startup
    print("Stock-AIML API starting up...")
    yield
    # Shutdown
    print("Stock-AIML API shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock-AIML API",
    description="Deep Learning Backtesting & Strategy Optimization Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(backtest.router)
app.include_router(stress.router)
app.include_router(optimize.router)
app.include_router(explain.router)
app.include_router(data.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "name": "Stock-AIML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
