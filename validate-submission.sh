#!/usr/bin/env bash
# Validate that the HF Space is live, the Docker image builds, and openenv validate passes.
#
# Usage: ./validate-submission.sh <ping_url> [repo_dir]

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

REPO_DIR="$(cd "$REPO_DIR" && pwd)"
PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS+1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() { printf "\n${RED}${BOLD}Stopped at %s.${NC}\n" "$1"; exit 1; }

printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  Stocker Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"

log "${BOLD}Step 1/3:${NC} Pinging HF Space ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /tmp/stocker_resp -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live"
else
  fail "HF Space returned $HTTP_CODE"
  hint "Check the Space is running at $PING_URL"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3:${NC} docker build ..."
if ! command -v docker &>/dev/null; then
  fail "docker not found"; stop_at "Step 2"
fi
if ! docker build "$REPO_DIR" >/tmp/stocker_build.log 2>&1; then
  fail "docker build failed"
  tail -20 /tmp/stocker_build.log
  stop_at "Step 2"
fi
pass "docker build succeeded"

log "${BOLD}Step 3/3:${NC} openenv validate ..."
if ! command -v openenv &>/dev/null; then
  fail "openenv not found"; hint "pip install openenv-core"; stop_at "Step 3"
fi
if ! ( cd "$REPO_DIR" && openenv validate ); then
  fail "openenv validate failed"; stop_at "Step 3"
fi
pass "openenv validate"

printf "\n${GREEN}${BOLD}All 3/3 checks passed.${NC}\n"
