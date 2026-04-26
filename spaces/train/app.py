"""Gradio training UI for Stocker GRPO on HF Spaces GPU.

Deploy as a Space with hardware "Nvidia L4" (24 GB).
Required secrets (set in Space Settings → Variables and secrets):
  HF_TOKEN      — read/write access for uploading results
  API_BASE_URL  — https://at0e6z2u64774tc7.us-east-1.aws.endpoints.huggingface.cloud/v1
  MODEL_NAME    — ggml-org/gemma-4-26B-A4B-it-GGUF
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi

WORKDIR = Path(__file__).resolve().parent.parent.parent  # repo root
LOG_PATH = Path("/tmp/stocker_train.log")
RESULTS_REPO = os.getenv("RESULTS_REPO", "Hydr473/stocker-results")

_state: dict = {"phase": "idle", "proc": None}


def _stream_proc(cmd: list[str], env_extra: dict | None = None) -> None:
    env = {**os.environ, **(env_extra or {})}
    _state["proc"] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(WORKDIR),
        env=env,
    )
    with open(LOG_PATH, "a") as lf:
        assert _state["proc"] and _state["proc"].stdout
        for line in _state["proc"].stdout:
            lf.write(line)
            lf.flush()
    _state["proc"].wait()


def _run_pipeline() -> None:
    LOG_PATH.write_text("")  # clear log
    endpoint_env = {
        "API_BASE_URL": os.getenv("API_BASE_URL", ""),
        "MODEL_NAME":   os.getenv("MODEL_NAME", ""),
        "HF_TOKEN":     os.getenv("HF_TOKEN", ""),
    }

    # Phase 1 — pre-cache specialist votes via HF inference endpoint
    _state["phase"] = "precaching"
    _stream_proc(
        [sys.executable, "scripts/precache_endpoint.py", "--tasks", "task_easy"],
        env_extra=endpoint_env,
    )

    # Phase 2 — baseline eval (no LoRA). Hits the 26B endpoint, but
    # specialist votes are already cached from Phase 1 so this is fast.
    _state["phase"] = "eval_pre"
    _stream_proc(
        [sys.executable, "-m", "training.eval_rollout",
         "--tasks", "task_easy",
         "--out", "training/runs/eval_pre"],
        env_extra=endpoint_env,
    )

    # Phase 3 — GRPO training on E4B (fits on L4 24 GB)
    # (4 * 1) % 4 == 0 ✓
    _state["phase"] = "training"
    _stream_proc(
        [sys.executable, "-m", "training.train_grpo",
         "--tasks", "task_easy",
         "--model", "google/gemma-4-E4B-it",
         "--epochs", "3",
         "--num-generations", "4",
         "--batch-size", "4",
         "--grad-accum", "1",
         "--lora-rank", "16",
         "--lr", "5e-6"],
        env_extra=endpoint_env,
    )

    # Phase 4 — post-training eval (with the trained LoRA loaded into the
    # moderator path; specialist votes still come from cache).
    _state["phase"] = "eval_post"
    run_dirs = sorted(glob.glob(str(WORKDIR / "training/runs/grpo_*")))
    if run_dirs:
        lora_dir = Path(run_dirs[-1]) / "moderator-lora"
        _stream_proc(
            [sys.executable, "-m", "training.eval_rollout",
             "--tasks", "task_easy",
             "--moderator-lora", str(lora_dir),
             "--out", "training/runs/eval_post"],
            env_extra=endpoint_env,
        )

    # Phase 5 — compile results
    _state["phase"] = "compiling"
    _stream_proc([sys.executable, "scripts/compile_results.py"])

    # Phase 6 — upload artifacts to HF Hub
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
    "precaching": "⏳ Phase 1/5 — pre-caching 26B specialist votes ...",
    "eval_pre":   "⏳ Phase 2/5 — baseline eval (no LoRA) ...",
    "training":   "🚀 Phase 3/5 — GRPO training Gemma 4 E4B ...",
    "eval_post":  "⏳ Phase 4/5 — post-training eval ...",
    "compiling":  "⏳ Phase 5/5 — compiling results ...",
    "uploading":  "⬆️  Uploading artifacts to HF Hub ...",
    "done":       "✅ Done! See plots below.",
}


def launch_training():
    if _state["phase"] not in ("idle", "done"):
        return PHASE_LABELS.get(_state["phase"], _state["phase"])
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return PHASE_LABELS["precaching"]


def poll_status():
    return PHASE_LABELS.get(_state["phase"], _state["phase"])


def poll_logs():
    if LOG_PATH.exists():
        text = LOG_PATH.read_text()
        return text[-8000:] if len(text) > 8000 else text
    return ""


def poll_plots():
    patterns = [
        "training/runs/grpo_*/*.png",
        "training/runs/eval_pre/*.png",
        "training/runs/eval_post/*.png",
    ]
    images = []
    for p in patterns:
        images.extend(sorted(glob.glob(str(WORKDIR / p))))
    return images or None


with gr.Blocks(title="Stocker — GRPO Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Stocker — GRPO Training\n"
        "Runs the full pipeline: **pre-cache 26B specialist votes → baseline eval → "
        "GRPO train E4B moderator LoRA → post eval → compile & upload**.\n\n"
        "Requires `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` set as Space secrets."
    )

    status_box = gr.Textbox(
        label="Status",
        value=PHASE_LABELS["idle"],
        interactive=False,
        lines=1,
    )

    with gr.Row():
        launch_btn = gr.Button("🚀 Launch Pipeline", variant="primary", scale=2)

    log_box = gr.Textbox(
        label="Live Logs",
        lines=30,
        max_lines=60,
        interactive=False,
        autoscroll=True,
    )

    gallery = gr.Gallery(
        label="Training & Eval Plots",
        show_label=True,
        columns=2,
        height="auto",
    )

    launch_btn.click(fn=launch_training, outputs=status_box)

    # Poll every 3 seconds
    timer = gr.Timer(3.0)
    timer.tick(fn=poll_status, outputs=status_box)
    timer.tick(fn=poll_logs,   outputs=log_box)
    timer.tick(fn=poll_plots,  outputs=gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
