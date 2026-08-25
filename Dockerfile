# Production Dockerfile for Smart Traffic Route Planner backend
# Builds a small container to run the FastAPI app using Gunicorn+Uvicorn

FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install system deps required by some Python packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the repository into the image
COPY . /app

# Expose the port used by Render and other hosts
ENV PORT 8000
EXPOSE 8000

# Use Gunicorn with Uvicorn workers for production readiness
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--bind", "0.0.0.0:8000"]
