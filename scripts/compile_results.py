#!/usr/bin/env python3
"""Compile GRPO training artifacts into training/runs/RESULTS.md.

Finds the latest grpo_* dir and eval_pre / eval_post dirs, reads their
summary.csv files, and writes a markdown results document that can be
pasted into the HF blog or README.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "training" / "runs"


def _latest(pattern: str) -> Path | None:
    hits = sorted(RUNS.glob(pattern))
    return hits[-1] if hits else None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_args(run_dir: Path) -> dict:
    p = run_dir / "args.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _copy_assets(src: Path | None, label: str, out_dir: Path) -> dict[str, str]:
    """Copy PNGs to out_dir and return {label: relative_path}."""
    mapping = {}
    if src is None:
        return mapping
    for png in src.glob("*.png"):
        dest = out_dir / f"{label}_{png.name}"
        shutil.copy2(png, dest)
        mapping[png.stem] = dest.relative_to(ROOT).as_posix()
    return mapping


def main() -> None:
    grpo_dir = _latest("grpo_*")
    pre_dir = _latest("eval_pre*") or _latest("eval_*pre*")
    post_dir = _latest("eval_post*") or _latest("eval_*post*")

    if grpo_dir is None:
        sys.exit("No grpo_* run directory found in training/runs/. Run training first.")

    out_dir = RUNS
    out_dir.mkdir(parents=True, exist_ok=True)

    assets: dict[str, str] = {}
    assets.update(_copy_assets(grpo_dir, "train", out_dir))
    assets.update(_copy_assets(pre_dir, "pre", out_dir))
    assets.update(_copy_assets(post_dir, "post", out_dir))

    args = _read_args(grpo_dir)
    pre_rows = _read_csv(pre_dir / "summary.csv") if pre_dir else []
    post_rows = _read_csv(post_dir / "summary.csv") if post_dir else []

    # Build comparison table
    pre_map = {r["task_id"]: r for r in pre_rows}
    post_map = {r["task_id"]: r for r in post_rows}
    all_tasks = sorted(set(list(pre_map) + list(post_map)))

    lines: list[str] = []
    lines.append("# Stocker GRPO Training Results\n")
    lines.append(f"\n**Run:** `{grpo_dir.name}`\n")
    if args:
        lines.append("\n## Training config\n\n")
        lines.append("| Parameter | Value |\n")
        lines.append("|-----------|-------|\n")
        for k, v in args.items():
            lines.append(f"| `{k}` | `{v}` |\n")

    lines.append("\n## Pre vs Post comparison\n\n")
    lines.append("| Task | Reward (pre) | Reward (post) | Δ Reward | Final PV (pre) | Final PV (post) | Alpha (post) |\n")
    lines.append("|------|-------------|--------------|----------|----------------|-----------------|-------------|\n")
    for task in all_tasks:
        pre = pre_map.get(task, {})
        post = post_map.get(task, {})
        r_pre = float(pre.get("total_reward", 0) or 0)
        r_post = float(post.get("total_reward", 0) or 0)
        delta = r_post - r_pre
        pv_pre = pre.get("final_portfolio", "—")
        pv_post = post.get("final_portfolio", "—")
        alpha = post.get("alpha_pct", "—")
        lines.append(
            f"| `{task}` | {r_pre:+.4f} | {r_post:+.4f} | "
            f"**{delta:+.4f}** | {pv_pre} | {pv_post} | {alpha}% |\n"
        )

    if pre_rows or post_rows:
        avg_pre = sum(float(r.get("total_reward", 0) or 0) for r in pre_rows) / max(len(pre_rows), 1)
        avg_post = sum(float(r.get("total_reward", 0) or 0) for r in post_rows) / max(len(post_rows), 1)
        lines.append(f"\n**Average reward pre-training:** {avg_pre:+.4f}  \n")
        lines.append(f"**Average reward post-training:** {avg_post:+.4f}  \n")
        lines.append(f"**Average reward delta:** {avg_post - avg_pre:+.4f}\n")

    lines.append("\n## Training curves\n\n")
    for key in ["train_loss", "train_reward"]:
        if key in assets:
            label = key.replace("train_", "").capitalize()
            lines.append(f"### {label}\n\n")
            lines.append(f"![{label}]({assets[key]})\n\n")

    lines.append("\n## Evaluation curves\n\n")
    for label, prefix in [("Pre-training", "pre"), ("Post-training", "post")]:
        reward_key = f"{prefix}_reward_curve"
        pv_key = f"{prefix}_portfolio_curve"
        if reward_key in assets or pv_key in assets:
            lines.append(f"### {label}\n\n")
            if reward_key in assets:
                lines.append(f"![Reward curve]({assets[reward_key]})\n\n")
            if pv_key in assets:
                lines.append(f"![Portfolio curve]({assets[pv_key]})\n\n")

    out_path = RUNS / "RESULTS.md"
    out_path.write_text("".join(lines))
    print(f"Results written to: {out_path}")
    print("".join(lines))


if __name__ == "__main__":
    main()
