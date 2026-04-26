# Training Space — Gemma 4 E4B GRPO on L4 GPU (24 GB)
# Uploaded as Dockerfile (root) of the Hydr473/stocker-train Space.
#
# Required Space secrets (Settings → Variables and secrets):
#   HF_TOKEN      write-access token for uploading results
#   API_BASE_URL  https://<endpoint>.endpoints.huggingface.cloud/v1
#   MODEL_NAME    ggml-org/gemma-4-26B-A4B-it-GGUF
#   RESULTS_REPO  Hydr473/stocker-results
#
# Layer order is deliberate:
#   1. OS packages         (rare changes — cached forever)
#   2. Heavy pip install   (cached unless THIS Dockerfile changes)
#   3. Smoke test          (fails build immediately on dep mismatch)
#   4. ADD commits/main    (cache-busts everything below)
#   5. git clone           (~10s)
#   6. editable install    (~5s)
#   7. dataset build       (~30s)
# Normal git push triggers only steps 5-7 (~45s total).

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ─── Layer 1: OS packages ───────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ─── Layer 2: heavy pip install (cached) ────────────────────────────────
# Newer coherent stack: torch 2.6 (has torch.int1) → torchao>=0.16 →
# peft>=0.16 (LoRA-injection probe satisfied) → transformers>=4.55
# (correct Gemma-4 tokenizer behavior) → trl>=0.22 → huggingface_hub>=1.0.
# Gradio is NOT installed — it conflicts with openenv-core's websockets>=15.
# The training Space UI uses minimal FastAPI + HTML (spaces/train/app.py).
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.0.0" "pydantic-settings>=2.6.0" \
        "openai>=1.0.0" "openenv-core>=0.2.0" \
        "pandas>=2.0.0" "pyarrow>=14.0.0" "pillow>=10.0.0" "markdown-it-py>=3.0.0" \
        "yfinance>=0.2.0" "mplfinance>=0.12.0" "matplotlib>=3.8.0" \
        "trl>=0.22.0" "transformers>=4.55.0" "peft>=0.16.0" \
        "accelerate>=1.6.0" "bitsandbytes>=0.45.0" "datasets>=3.0.0" \
        "tensorboard>=2.18.0" "torchao>=0.16.0" "huggingface_hub>=1.0.0"

# ─── Layer 3: build-time smoke test ────────────────────────────────────
# If any dep combo is broken (import error, version mismatch, missing
# attribute, tokenizer regression), this fails at BUILD time in 30 sec
# instead of crashing 25 minutes into phase 3 of training.
RUN python -c "\
import torch, transformers, peft, trl, bitsandbytes, accelerate; \
import huggingface_hub as hub; \
from peft import LoraConfig; \
from trl import GRPOConfig, GRPOTrainer; \
from transformers import AutoTokenizer; \
print(f'torch={torch.__version__} hub={hub.__version__}'); \
print(f'transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__}'); \
assert hasattr(torch, 'int1'), 'torch.int1 missing — torchao>=0.16 will crash'; \
print('smoke test passed')\
"

# ─── Layer 4: cache-bust on every commit to main ────────────────────────
ADD https://api.github.com/repos/CRIMSONHydra/stocker/commits/main /tmp/.commit-sha

# ─── Layer 5: fresh clone of the repo ───────────────────────────────────
RUN git clone --depth=1 https://github.com/CRIMSONHydra/stocker.git /app

# ─── Layer 6: register the local package (no dep resolution) ────────────
RUN cd /app && pip install --no-deps -e .

# ─── Layer 7: build the bundled dataset ─────────────────────────────────
RUN python scripts/build_dataset.py && python scripts/build_ideal_profit.py

EXPOSE 7860
CMD ["python", "spaces/train/app.py"]
