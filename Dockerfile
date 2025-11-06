# Multi-stage Dockerfile for WJP ANALYSER
# =======================================

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    bandit \
    safety

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs output uploads cache

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose Streamlit port only for dev
EXPOSE 8501

# Default command for development: unified Streamlit UI
CMD ["python", "wjp_analyser_unified.py", "web-ui", "--interface", "streamlit", "--host", "0.0.0.0", "--port", "8501"]

# Stage 3: Production environment
FROM base as production

# Install production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard]

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs output uploads cache

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose Streamlit in production as well (Streamlit-only build)
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command for production: unified Streamlit UI
CMD ["python", "wjp_analyser_unified.py", "web-ui", "--interface", "streamlit", "--host", "0.0.0.0", "--port", "8501"]
