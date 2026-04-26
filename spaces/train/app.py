"""Training Space UI — minimal FastAPI + HTML.

Avoids Gradio because gradio<5 pulls gradio-client==1.3.0 which constrains
websockets<13, conflicting with openenv-core's websockets>=15.0.1.

Required Space secrets (Settings → Variables and secrets):
  HF_TOKEN      — write-access token for uploading results
  API_BASE_URL  — https://<endpoint>.endpoints.huggingface.cloud/v1
  MODEL_NAME    — ggml-org/gemma-4-26B-A4B-it-GGUF
  RESULTS_REPO  — Hydr473/stocker-results  (defaults to this if unset)
  USE_MOCK_SPECIALISTS=1  — fall back to MockLLMClient (skip endpoint calls)
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from huggingface_hub import HfApi

WORKDIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = Path("/tmp/stocker_train.log")
RESULTS_REPO = os.getenv("RESULTS_REPO", "Hydr473/stocker-results")

_state: dict = {"phase": "idle"}


def _stream_proc(cmd: list[str], env_extra: dict | None = None) -> None:
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(WORKDIR),
        env=env,
    )
    with open(LOG_PATH, "a") as lf:
        assert proc.stdout
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
    proc.wait()


def _run_pipeline() -> None:
    LOG_PATH.write_text("")
    endpoint_env = {
        "API_BASE_URL": os.getenv("API_BASE_URL", ""),
        "MODEL_NAME":   os.getenv("MODEL_NAME", ""),
        "HF_TOKEN":     os.getenv("HF_TOKEN", ""),
    }
    use_mock = os.getenv("USE_MOCK_SPECIALISTS", "0") == "1"
    mock_flag = ["--mock"] if use_mock else []
    train_mock_flag = ["--mock-specialists"] if use_mock else []

    _state["phase"] = "precaching"
    _stream_proc(
        [sys.executable, "scripts/precache_endpoint.py",
         "--tasks", "task_easy", *mock_flag],
        env_extra=endpoint_env,
    )

    _state["phase"] = "eval_pre"
    _stream_proc(
        [sys.executable, "-m", "training.eval_rollout",
         "--tasks", "task_easy", *mock_flag,
         "--out", "training/runs/eval_pre"],
        env_extra=endpoint_env,
    )

    _state["phase"] = "training"
    _stream_proc(
        [sys.executable, "-m", "training.train_grpo",
         "--tasks", "task_easy", *train_mock_flag,
         "--model", "google/gemma-4-E4B-it",
         "--epochs", "3",
         "--num-generations", "4",
         "--batch-size", "4",
         "--grad-accum", "1",
         "--lora-rank", "16",
         "--lr", "5e-6"],
        env_extra=endpoint_env,
    )

    _state["phase"] = "eval_post"
    run_dirs = sorted(glob.glob(str(WORKDIR / "training/runs/grpo_*")))
    if run_dirs:
        lora_dir = Path(run_dirs[-1]) / "moderator-lora"
        _stream_proc(
            [sys.executable, "-m", "training.eval_rollout",
             "--tasks", "task_easy", *mock_flag,
             "--moderator-lora", str(lora_dir),
             "--out", "training/runs/eval_post"],
            env_extra=endpoint_env,
        )

    _state["phase"] = "compiling"
    _stream_proc([sys.executable, "scripts/compile_results.py"])

    _state["phase"] = "uploading"
    _upload_results(run_dirs[-1] if run_dirs else None)

    _state["phase"] = "done"


def _upload_results(run_dir: str | None) -> None:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        return
    api = HfApi(token=token)
    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True)
    except Exception:
        pass

    to_upload = []
    for pattern in [
        "training/runs/*.md",
        "training/runs/eval_pre/*.png",
        "training/runs/eval_post/*.png",
    ]:
        to_upload.extend(glob.glob(str(WORKDIR / pattern)))
    if run_dir:
        for pattern in ["*.png", "args.json"]:
            to_upload.extend(glob.glob(str(Path(run_dir) / pattern)))

    for local_path in to_upload:
        rel = Path(local_path).relative_to(WORKDIR)
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=str(rel),
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                token=token,
            )
            with open(LOG_PATH, "a") as lf:
                lf.write(f"Uploaded: {rel}\n")
        except Exception as e:
            with open(LOG_PATH, "a") as lf:
                lf.write(f"Upload failed for {rel}: {e}\n")


PHASE_LABELS = {
    "idle":       "⭕ Ready — click Launch to start",
    "precaching": "⏳ Phase 1/5 — pre-caching specialist votes ...",
    "eval_pre":   "⏳ Phase 2/5 — baseline eval (no LoRA) ...",
    "training":   "🚀 Phase 3/5 — GRPO training Gemma 4 E4B ...",
    "eval_post":  "⏳ Phase 4/5 — post-training eval ...",
    "compiling":  "⏳ Phase 5/5 — compiling results ...",
    "uploading":  "⬆️  Uploading artifacts to HF Hub ...",
    "done":       "✅ Done! See plots below.",
}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stocker — GRPO Training</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 24px; max-width: 1100px; margin: 0 auto; }
  h1 { color: #38bdf8; }
  h2 { color: #38bdf8; border-top: 1px solid #334155; padding-top: 12px; margin-top: 24px; }
  .status { background: #1e293b; padding: 14px 18px; border-radius: 8px; margin: 12px 0; font-size: 1.1em; border: 1px solid #334155; }
  .btn { background: #2563eb; color: white; padding: 12px 28px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; }
  .btn:hover { background: #1d4ed8; }
  .btn:disabled { background: #475569; cursor: not-allowed; }
  .logs { background: #020617; padding: 14px; border-radius: 8px; font-family: 'SF Mono', Menlo, monospace; font-size: 12px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #1e293b; }
  .plots img { max-width: 48%; border-radius: 8px; margin: 8px 1%; border: 1px solid #334155; }
  .meta { color: #94a3b8; font-size: 0.9em; margin: 8px 0; }
</style>
</head>
<body>
  <h1>⚡ Stocker — GRPO Training</h1>
  <p class="meta">L4 GPU · pre-cache 26B specialist votes → baseline eval → GRPO E4B → post eval → compile → upload</p>

  <div class="status" id="status">Loading status…</div>
  <button class="btn" id="launchBtn" onclick="launch()">🚀 Launch Pipeline</button>

  <h2>Live logs</h2>
  <div class="logs" id="logs">No logs yet.</div>

  <h2>Plots</h2>
  <div class="plots" id="plots">Plots appear here once training completes.</div>

<script>
async function launch() {
  const r = await fetch('/launch', { method: 'POST' });
  const d = await r.json();
  if (d.status === 'started') document.getElementById('launchBtn').disabled = true;
}
async function poll() {
  try {
    const s = await (await fetch('/status')).json();
    document.getElementById('status').textContent = s.label;
    if (s.phase !== 'idle' && s.phase !== 'done') {
      document.getElementById('launchBtn').disabled = true;
    } else {
      document.getElementById('launchBtn').disabled = false;
    }
    const txt = await (await fetch('/logs')).text();
    const logsEl = document.getElementById('logs');
    const wasNearBottom = logsEl.scrollHeight - logsEl.scrollTop - logsEl.clientHeight < 50;
    logsEl.textContent = txt.slice(-12000) || 'No logs yet.';
    if (wasNearBottom) logsEl.scrollTop = logsEl.scrollHeight;
    const ps = await (await fetch('/plots')).json();
    if (ps.length) {
      document.getElementById('plots').innerHTML = ps.map(u => `<img src="${u}" />`).join('');
    }
  } catch (e) { /* ignore poll errors */ }
}
setInterval(poll, 3000);
poll();
</script>
</body>
</html>
"""


app = FastAPI(title="Stocker Training")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.post("/launch")
def launch():
    if _state["phase"] in ("idle", "done"):
        LOG_PATH.write_text("")
        threading.Thread(target=_run_pipeline, daemon=True).start()
        return {"status": "started"}
    return {"status": "already_running", "phase": _state["phase"]}


@app.get("/status")
def status():
    phase = _state["phase"]
    return {"phase": phase, "label": PHASE_LABELS.get(phase, phase)}


@app.get("/logs", response_class=PlainTextResponse)
def logs():
    if LOG_PATH.exists():
        return LOG_PATH.read_text()
    return ""


@app.get("/plots")
def plots():
    patterns = [
        "training/runs/grpo_*/*.png",
        "training/runs/eval_pre/*.png",
        "training/runs/eval_post/*.png",
    ]
    images = []
    for p in patterns:
        images.extend(sorted(glob.glob(str(WORKDIR / p))))
    return [f"/plot?p={i}" for i in images]


@app.get("/plot")
def plot(p: str):
    if not p.startswith(str(WORKDIR)):
        return PlainTextResponse("forbidden", status_code=403)
    return FileResponse(p)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
