# Use Runpod PyTorch base image
FROM runpod/pytorch:1.0.3-cu1300-torch280-ubuntu2404

# Set environment variables
# This ensures Python output is immediately visible in logs, and configures UV for production use.
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install .
# ENV PATH="/app/.venv/bin:$PATH"

# Switch to the persistent RunPod workspace so all relative output paths
# (checkpoints/, logs/, mlruns/, generated/) land in persistent storage.
WORKDIR /workspace