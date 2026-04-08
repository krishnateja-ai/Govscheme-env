FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (layer cache)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/app.py .
COPY server/govscheme_environment.py .
COPY server/eligibility.py .
COPY server/graders.py .
COPY server/schemes.json .
COPY server/citizens.json .
COPY server/__init__.py .

# Copy models.py (shared between server and client)
COPY models.py .

# HF Spaces non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
