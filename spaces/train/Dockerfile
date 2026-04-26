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
#   3. ADD commits/main    (cache-busts everything below)
#   4. git clone           (~10s)
#   5. editable install    (~5s)
#   6. dataset build       (~30s)
# So a normal git push only triggers steps 4-6 (~45s total).

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
# Pinned for torch 2.5.1 compatibility:
#   transformers<4.50 → avoids torch.int1 import path in newer torchao
#   peft<0.13         → avoids the torchao>=0.16 LoRA-injection probe
#   trl<0.20          → compatible with peft<0.13
#   gradio<5          → keeps huggingface_hub<1.0 (required by transformers<4.50)
# huggingface_hub is intentionally NOT pinned — pip resolves to <1.0 to
# satisfy transformers' constraint, which lets older gradio + transformers
# coexist without conflicts.
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.0.0" "pydantic-settings>=2.6.0" \
        "openai>=1.0.0" "openenv-core>=0.2.0" \
        "pandas>=2.0.0" "pyarrow>=14.0.0" "pillow>=10.0.0" "markdown-it-py>=3.0.0" \
        "yfinance>=0.2.0" "mplfinance>=0.12.0" "matplotlib>=3.8.0" \
        "trl>=0.11.0,<0.20.0" "transformers>=4.45.0,<4.50.0" \
        "peft>=0.11.0,<0.13.0" "accelerate>=1.0.0" "bitsandbytes>=0.43.0" \
        "datasets>=3.0.0" "tensorboard>=2.18.0" \
        "gradio>=4.44.0,<5.0"

# Fail-fast smoke test: if dep resolution produced an incompatible set,
# this layer fails the build immediately instead of crashing at phase 3.
RUN python -c "import torch, transformers, peft, trl, bitsandbytes, accelerate; \
from peft import LoraConfig; from trl import GRPOConfig, GRPOTrainer; \
import huggingface_hub as hub; \
print(f'OK torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__} hub={hub.__version__}')"

# ─── Layer 3: cache-bust on every commit to main ────────────────────────
ADD https://api.github.com/repos/CRIMSONHydra/stocker/commits/main /tmp/.commit-sha

# ─── Layer 4: fresh clone of the repo ───────────────────────────────────
RUN git clone --depth=1 https://github.com/CRIMSONHydra/stocker.git /app

# ─── Layer 5: register the local package (no dep resolution) ────────────
RUN cd /app && pip install --no-deps -e .

# ─── Layer 6: build the bundled dataset ─────────────────────────────────
RUN python scripts/build_dataset.py && python scripts/build_ideal_profit.py

EXPOSE 7860
CMD ["python", "spaces/train/app.py"]
