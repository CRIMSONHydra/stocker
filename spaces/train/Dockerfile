# Training Space — Gemma 4 GRPO on L4 GPU (24 GB)
# Uploaded as Dockerfile (root) of the Hydr473/stocker-train Space.
#
# Decoupled from the env Space: this Dockerfile pins ALL ML deps inline
# with exact versions. The env Space (root Dockerfile) installs only core
# FastAPI deps from pyproject.toml and never touches torch/transformers.
# Conflicts in either Space cannot affect the other.
#
# Required Space secrets (Settings → Variables and secrets):
#   HF_TOKEN, API_BASE_URL, MODEL_NAME, RESULTS_REPO

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

# ─── Layer 2: fresh venv (avoids PEP 668 in newer pytorch base images) ──
# Not using --system-site-packages — we install our own pinned torch so
# the dep set is fully reproducible regardless of base image's torch.
# pytorch:2.10+ images are Ubuntu 24.04 based, Python at /usr/bin/python3
# (via python-is-python3). Older images had Python under /opt/conda.
ENV VIRTUAL_ENV=/opt/stocker-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir uv

# ─── Layer 3: heavy pip install (cached unless THIS Dockerfile changes) ─
# All exact-pinned. Torch/torchvision/torchaudio come from the base image.
# These versions were verified locally to import cleanly together and load
# the google/gemma-4-E4B-it tokenizer without the 'list has no .keys()'
# regression.
# --index-strategy unsafe-best-match needed because the PyTorch wheel
# index ships an old `requests` (2.28.1) that breaks datasets>=4.0
# (needs requests>=2.32.2). With this flag, uv picks the right version
# across both indexes. (Verified via uv pip install --dry-run locally.)
RUN uv pip install --no-cache --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "fastapi" "uvicorn[standard]" "pydantic>=2.0" "pydantic-settings" \
    "openai" "openenv-core" "pandas" "pyarrow" "pillow" "markdown-it-py" \
    "yfinance" "mplfinance" "matplotlib" \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
    "transformers==5.6.2" "peft==0.19.1" "trl==1.2.0" \
    "accelerate==1.13.0" "bitsandbytes==0.49.2" "datasets==4.8.4" \
    "tensorboard==2.20.0" "torchao==0.17.0" "huggingface_hub==1.12.0"

# ─── Layer 4: build-time smoke test ────────────────────────────────────
# Fails the build immediately if any pin combination is broken.
RUN python -c "\
import torch, transformers, peft, trl, bitsandbytes, accelerate; \
import huggingface_hub as hub; \
from peft import LoraConfig; \
from trl import GRPOConfig, GRPOTrainer; \
from transformers import AutoTokenizer; \
print(f'torch={torch.__version__} hub={hub.__version__}'); \
print(f'transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__}'); \
assert hasattr(torch, 'int1'), 'torch.int1 missing'; \
assert hasattr(torch.utils._pytree, 'register_constant'), 'register_constant missing'; \
print('smoke test (imports + torch features): passed')\
"

# ─── Layer 5: cache-bust on every commit to main ────────────────────────
ADD https://api.github.com/repos/CRIMSONHydra/stocker/commits/main /tmp/.commit-sha

# ─── Layer 6: fresh clone of the repo (gets latest code) ────────────────
RUN git clone --depth=1 https://github.com/CRIMSONHydra/stocker.git /app

# ─── Layer 7: register the local package WITHOUT pulling its extras ─────
# Crucial: --no-deps. Otherwise pip would try to re-resolve everything
# from pyproject.toml and our pinned versions could drift.
RUN cd /app && pip install --no-deps -e .

# ─── Layer 8: build the bundled dataset ─────────────────────────────────
RUN python scripts/build_dataset.py && python scripts/build_ideal_profit.py

EXPOSE 7860
CMD ["python", "spaces/train/app.py"]
