# Use Gunicorn with Uvicorn worker for production on Render
FROM python:3.11-slim

# Install system dependencies required by osmnx/geopandas/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin libgdal-dev \
    libgeos-dev \
    libproj-dev proj-bin proj-data \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL (optional but sometimes required)
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

# Create working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python requirements
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose application port
EXPOSE 8000

# Default environment variables (can be overridden in Render)
ENV OSM_PLACE="Pune, India"
ENV PORT=8000

# Run the app with Gunicorn + Uvicorn worker
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--bind", "0.0.0.0:8000"]
