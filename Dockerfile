FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO- cli.runpod.net | sudo bash

# Create venv
RUN python3.12 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools

COPY . /app

# Install nnUNet and dependencies
RUN pip install --no-cache-dir .

# Switch to persistent workspace for outputs
WORKDIR /workspace

CMD ["/bin/bash"]
