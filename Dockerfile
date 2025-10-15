# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies for OpenCV (updated for newer Debian)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs models

# Expose port (Render will use the PORT environment variable)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}