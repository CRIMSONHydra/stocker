"""Training Space UI — Gradio (5.x).

Compatible with openenv-core because gradio>=5 dropped the websockets<13
pin (gradio-client 1.14+ uses websockets>=13 / >=15 family).

Required Space secrets (Settings → Variables and secrets):
  HF_TOKEN      — write-access token for uploading results
  API_BASE_URL  — https://<endpoint>.endpoints.huggingface.cloud/v1
  MODEL_NAME    — ggml-org/gemma-4-26B-A4B-it-GGUF
  RESULTS_REPO  — Hydr473/stocker-results  (defaults if unset)
  USE_MOCK_SPECIALISTS=1  — fall back to MockLLMClient (skip endpoint)
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
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
        bufsize=1,
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
         "--lr", "5e-6",
         "--imitation-warmstart"],
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


def launch_pipeline() -> str:
    if _state["phase"] not in ("idle", "done"):
        return PHASE_LABELS.get(_state["phase"], _state["phase"])
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return PHASE_LABELS["precaching"]


def poll_status() -> str:
    return PHASE_LABELS.get(_state["phase"], _state["phase"])


def poll_logs() -> str:
    if not LOG_PATH.exists():
        return "No logs yet."
    text = LOG_PATH.read_text()
    return text[-12000:] if len(text) > 12000 else text


def poll_plots() -> list[str]:
    patterns = [
        "training/runs/grpo_*/*.png",
        "training/runs/eval_pre/*.png",
        "training/runs/eval_post/*.png",
    ]
    images = []
    for p in patterns:
        images.extend(sorted(glob.glob(str(WORKDIR / p))))
    return images


with gr.Blocks(title="Stocker — GRPO Training") as demo:
    gr.Markdown(
        "# ⚡ Stocker — GRPO Training\n"
        "L4 GPU · pre-cache 26B specialist votes → baseline eval → "
        "GRPO E4B → post eval → compile → upload"
    )

    status_box = gr.Textbox(
        label="Status",
        value=PHASE_LABELS["idle"],
        interactive=False,
        lines=1,
    )

    with gr.Row():
        launch_btn = gr.Button(
            "🚀 Launch Pipeline",
            variant="primary",
            scale=2,
        )

    log_box = gr.Textbox(
        label="Live logs",
        lines=24,
        max_lines=40,
        interactive=False,
        autoscroll=True,
    )

    gallery = gr.Gallery(
        label="Plots (training curves + eval rollouts)",
        show_label=True,
        columns=2,
        height="auto",
        object_fit="contain",
    )

    launch_btn.click(fn=launch_pipeline, outputs=status_box)

    timer = gr.Timer(2.0)
    timer.tick(fn=poll_status, outputs=status_box)
    timer.tick(fn=poll_logs,   outputs=log_box)
    timer.tick(fn=poll_plots,  outputs=gallery)

if __name__ == "__main__":
    # theme= moved to launch() in Gradio 6+; harmless on 5 too.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
