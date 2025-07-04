# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.9
FROM python:${PYTHON_VERSION}-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Switch to non-root user (best practice)
USER appuser

# Expose port (if the app uses it)
EXPOSE 8000

# Default command to run the app
CMD ["python", "main.py"]
