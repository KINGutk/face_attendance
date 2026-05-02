FROM python:3.11-slim

# Install system dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

# Copy application code
COPY . .

# Set working directory to the Flask app
WORKDIR /app/face_attendance_web

# Create faces and logs directories
RUN mkdir -p faces logs

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Run with gunicorn on port 7860 (HF Spaces requirement)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
