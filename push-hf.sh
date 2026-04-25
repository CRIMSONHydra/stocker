#!/usr/bin/env bash
# Push the project to a HuggingFace Space.
#
# Expects the following in .env (see .env.example):
#   HF_USERNAME    your HF username/org
#   HF_TOKEN       a write-scope token
#   HF_SPACE       the space name (e.g. Stocker)
set -euo pipefail

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

: "${HF_USERNAME:?Set HF_USERNAME in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_SPACE:?Set HF_SPACE in .env}"

git push "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE}" main
