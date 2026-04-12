FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python deps directly
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    requests==2.32.3 \
    python-dotenv==1.0.1 \
    "pydantic==2.8.2"

# Try openenv-core (graceful fallback)
RUN pip install openenv-core --quiet || pip install openenv --quiet || true

# Copy source files (no server/ — created inline below)
COPY main.py .
COPY app.py .
COPY inference.py .
COPY pyproject.toml .
COPY uv.lock .
COPY openenv.yaml .

# Create server/ package inline — avoids COPY server/ failures if folder missing
RUN mkdir -p server && \
    echo '# server package' > server/__init__.py && \
    printf 'import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom main import app\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))\n' > server/app.py

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
