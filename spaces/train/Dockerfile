# Training Space — Gemma 4 E4B GRPO on L4 GPU (24 GB)
# This file is uploaded as Dockerfile (root) of the Hydr473/stocker-train Space.
# It clones the stocker repo at build time so it has full access to all code.
#
# Required Space secrets (Settings → Variables and secrets):
#   HF_TOKEN      write-access token for uploading results
#   API_BASE_URL  https://at0e6z2u64774tc7.us-east-1.aws.endpoints.huggingface.cloud/v1
#   MODEL_NAME    ggml-org/gemma-4-26B-A4B-it-GGUF
#   RESULTS_REPO  Hydr473/stocker-results

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Bust cache when main branch advances on GitHub. The ADD layer fetches
# the latest commit metadata; its body changes on every commit, which
# invalidates all subsequent RUN layers (including the git clone below).
ADD https://api.github.com/repos/CRIMSONHydra/stocker/commits/main /tmp/.commit-sha

# Clone the stocker repo (latest main, no stale Docker cache)
RUN git clone --depth=1 https://github.com/CRIMSONHydra/stocker.git /app

# Install uv + all required extras + gradio for the UI
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[data,eval,train]" && \
    pip install --no-cache-dir "gradio>=4.44.0" "huggingface_hub>=1.10.0"

# Pre-build the bundled dataset (parquet + charts). Idempotent.
RUN python scripts/build_dataset.py && python scripts/build_ideal_profit.py

EXPOSE 7860

CMD ["python", "spaces/train/app.py"]
