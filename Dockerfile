FROM python:3.11-slim

# Install system dependencies for dlib/face_recognition
# Install system dependencies for dlib/face_recognition
# Install system dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libglib2.0-0 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Speed up dlib C++ compilation with parallel build
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

# Install dlib FIRST as separate layer (longest step, gets cached)
RUN pip install --no-cache-dir dlib==20.0.1

# Install face_recognition_models from GitHub (avoids pkg_resources issue)
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set working directory to the Flask app
WORKDIR /app/face_attendance_web

# Create faces and logs directories
RUN mkdir -p faces logs

# Environment memory optimizations for 512MB RAM instances
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2
ENV WEB_CONCURRENCY=1

# Run with gunicorn on port 7860 with 1 worker to fit within 512MB RAM
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
