#!/usr/bin/env bash
# Boot vLLM with Gemma 4 E4B IT, OpenAI-compatible at http://localhost:8000/v1
#
# Usage:
#   ./scripts/serve_vllm.sh                    # base only
#   ./scripts/serve_vllm.sh --lora <path>      # base + a LoRA adapter named "moderator"
#
# After it's up:
#   export API_BASE_URL=http://localhost:8000/v1
#   export MODEL_NAME=google/gemma-4-E4B-it
#   export HF_TOKEN=any
#   python inference.py --task all
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
QUANT="${QUANT:-bitsandbytes}"          # bitsandbytes | fp8 | none
DTYPE="${DTYPE:-bfloat16}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

LORA_PATH=""
LORA_NAME="moderator"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora)        LORA_PATH="$2"; shift 2 ;;
    --lora-name)   LORA_NAME="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --no-quant)    QUANT="none"; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)             echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ARGS=(
  --dtype "$DTYPE"
  --port "$PORT"
  --max-model-len "${MAX_MODEL_LEN:-4096}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.70}"
)
if [[ "$QUANT" != "none" ]]; then
  ARGS+=(--quantization "$QUANT")
fi
if [[ -n "$LORA_PATH" ]]; then
  ARGS+=(--enable-lora --max-loras 1 --max-lora-rank "$MAX_LORA_RANK"
         --lora-modules "${LORA_NAME}=${LORA_PATH}")
fi

echo "Booting vLLM:  vllm serve $MODEL ${ARGS[*]}"
exec vllm serve "$MODEL" "${ARGS[@]}"
