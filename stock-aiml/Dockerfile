FROM python:3.11-slim

WORKDIR /app

# ── System deps for weasyprint (PDF generation) ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
    libffi8 shared-mime-info libxml2 libxslt1.1 && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project ─────────────────────────────────────────────────────────────────
COPY . .

# ── Expose ports ──────────────────────────────────────────────────────────────
# Streamlit dashboard: 8501
# FastAPI: 8000
EXPOSE 8501 8000

# ── Entry point ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
