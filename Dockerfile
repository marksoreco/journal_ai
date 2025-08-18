# --- Build stage: build wheels for all deps ---
    FROM python:3.11-slim AS builder
    WORKDIR /build
    RUN pip install --no-cache-dir --upgrade pip setuptools wheel
    COPY requirements.txt .
    RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt
    
    # --- Runtime stage: minimal image + your app ---
    FROM python:3.11-slim
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PORT=8080
    WORKDIR /app
    
    # (Often needed by scikit-learn/numpy OpenMP at runtime)
    RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
      && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements (so pip can resolve from wheels)
    COPY requirements.txt .
    # Bring in prebuilt wheels from builder
    COPY --from=builder /wheels /wheels
    
    # ✅ Install using the wheel repository (no PyPI access during this step)
    RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
      && rm -rf /wheels
    
    # Copy app code
    COPY . .
    
    # Start FastAPI (Cloud Run sets $PORT)
    CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8080"]
    