FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .
COPY inference.py .

# Copy optional files
COPY openenv.yaml . 2>/dev/null || true

# Expose port
COPY app.py .
COPY openenv.yaml .
COPY pyproject.toml .

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run on 7860 (HF Spaces requirement)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
