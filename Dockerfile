# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    requests \
    python-dotenv \
    pydantic \
    openenv-core

# Copy entire project into container
COPY . .

# Expose Hugging Face required port
EXPOSE 7860

# Health check (required for HF stability)
HEALTHCHECK CMD curl --fail http://localhost:7860/ || exit 1

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
