"""POST /api/v1/data/upload — dataset upload."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/api/v1", tags=["data"])

DATA_DIR = Path(__file__).parents[3] / "data" / "raw"


@router.post("/data/upload")
async def upload_data(file: UploadFile = File(...)) -> dict:
    """
    Upload a CSV dataset.

    Saves to data/raw/<uuid>_<filename>.csv
    Returns the saved path for use in backtest calls.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    out_path  = DATA_DIR / safe_name

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status": "uploaded",
        "path": str(out_path),
        "filename": file.filename,
        "size_bytes": out_path.stat().st_size,
    }


@router.get("/data/datasets")
async def list_datasets() -> list[dict]:
    """List all available datasets in data/raw/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in DATA_DIR.glob("*.csv"):
        files.append({
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
        })
    return files
