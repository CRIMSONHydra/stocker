# Training Space — Gemma 4 E4B GRPO on L4 GPU (24 GB)
# Deploy this directory as a HF Space with hardware tier "Nvidia L4".
# Set secrets: HF_TOKEN, API_BASE_URL, MODEL_NAME in the Space settings.
#
# The main stocker repo root is the build context, so push like this:
#   huggingface-cli upload Hydr473/stocker-train . --repo-type=space
# OR push from the project root:
#   git subtree push ...  (see README for full instructions)
#
# This Dockerfile is placed in spaces/train/ but the build context is the
# repo root — so COPY paths are relative to the repo root.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency specification and install all extras
COPY pyproject.toml ./
RUN uv pip install --system -e ".[data,eval,train]"

# Reinstall gradio (not in pyproject.toml — only needed in this Space)
RUN pip install --no-cache-dir gradio>=4.44.0 huggingface_hub>=1.10.0

# Copy the full project (data/, app/, training/, scripts/, etc.)
COPY . .

# Pre-build the bundled dataset (parquet + charts). Idempotent.
RUN python scripts/build_dataset.py && python scripts/build_ideal_profit.py

EXPOSE 7860

CMD ["python", "spaces/train/app.py"]
