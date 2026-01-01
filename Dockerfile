# FLUX.1-Schnell RunPod Serverless Worker
# Model BAKED-IN + Network Volume support
# Optimized for 24GB VRAM (RTX 4090, A10, L40S)

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# HuggingFace Token (required for gated models)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    safetensors \
    sentencepiece \
    protobuf \
    Pillow \
    runpod \
    huggingface_hub

# Login to HuggingFace and download model
RUN python3 -c "\
import os; \
from huggingface_hub import login; \
token = os.environ.get('HF_TOKEN', ''); \
if token: \
    login(token=token); \
    print('Logged in to HuggingFace'); \
else: \
    print('WARNING: No HF_TOKEN provided'); \
"

# ============================================
# BAKE MODEL INTO IMAGE
# ============================================
RUN python3 -c "\
import torch; \
from diffusers import FluxPipeline; \
print('='*50); \
print('Downloading FLUX.1-Schnell model...'); \
print('='*50); \
pipe = FluxPipeline.from_pretrained( \
    'black-forest-labs/FLUX.1-schnell', \
    torch_dtype=torch.bfloat16 \
); \
print('='*50); \
print('Model cached successfully!'); \
print('='*50); \
"

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python3", "-u", "handler.py"]
