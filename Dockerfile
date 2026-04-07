# Use Python 3.11 slim image
FROM python:3.11-slim

# Hugging Face Spaces uses port 7860
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
