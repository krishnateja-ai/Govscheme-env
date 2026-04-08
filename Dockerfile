FROM python:3.11-slim
WORKDIR /app
# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
# Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy all app files
COPY app.py .
COPY govscheme_environment.py .
COPY eligibility.py .
COPY graders.py .
COPY models.py .
COPY schemes.json .
COPY citizens.json .
COPY __init__.py .
# HF Spaces non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]