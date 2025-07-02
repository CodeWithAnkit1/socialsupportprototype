# Use official Python 3.12 base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install Ollama ---
# Add Ollama binaries
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# --- Download llama3 model before startup ---
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama3

# Expose ports
EXPOSE 8501
EXPOSE 11434

# Set PYTHONPATH so Python can find your packages
ENV PYTHONPATH=/app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Copy your startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Final command
CMD ["/app/start.sh"]