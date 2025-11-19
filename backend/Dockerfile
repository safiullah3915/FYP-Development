FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Copy the backend source code
COPY backend /app

# Create media directory if it doesn't exist
RUN mkdir -p /app/media/uploads

# Default Django port
EXPOSE 8000

# Environment defaults (override via docker run -e ...)
ENV DJANGO_SETTINGS_MODULE=startup_platform.settings

# Start the Django development server (for production, use gunicorn)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

