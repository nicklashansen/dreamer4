#!/usr/bin/env python3
"""Plot BC eval curves from Phase 2 log file.

Parses the [bc_eval step ...] lines written by run_bc_eval() during training
and produces a summary figure alongside the existing phase2_curves.png.

Usage:
    python dreamer4/plot_bc_eval.py <run_dir>

Looks for <run_dir>/phase2.log and writes <run_dir>/bc_eval_curves.png.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_bc_eval_log(path: Path) -> dict:
    """Parse [bc_eval step ...] lines from a phase2 log file."""
    steps = []
    bc_nll, bc_std, act_mae, rew_mse, rew_corr = [], [], [], [], []

    pattern = re.compile(
        r"\[bc_eval step\s+(\d+)\]\s+"
        r"bc_nll=([-\d.nan]+)\s+"
        r"bc_std=([-\d.nan]+)\s+"
        r"act_mae=([-\d.nan]+)\s+"
        r"rew_mse=([-\d.nan]+)\s+"
        r"rew_corr=([-\d.nan]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                bc_nll.append(float(m.group(2)))
                bc_std.append(float(m.group(3)))
                act_mae.append(float(m.group(4)))
                rew_mse.append(float(m.group(5)))
                rew_corr.append(float(m.group(6)))

    return {
        "steps":    np.array(steps),
        "bc_nll":   np.array(bc_nll),
        "bc_std":   np.array(bc_std),
        "act_mae":  np.array(act_mae),
        "rew_mse":  np.array(rew_mse),
        "rew_corr": np.array(rew_corr),
    }


def smooth(y, window=3):
    if len(y) <= window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return (
        np.concatenate([y[:pad_left], smoothed, y[len(y) - pad_right:]])
        if pad_right > 0
        else np.concatenate([y[:pad_left], smoothed])
    )


def _can_use_log_scale(*arrays: np.ndarray) -> bool:
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        if np.nanmin(finite) <= 0:
            return False
    return True


def _apply_scale(ax, yscale: str, *arrays: np.ndarray) -> None:
    if yscale == "linear":
        return
    if yscale == "log" and not _can_use_log_scale(*arrays):
        return
    ax.set_yscale(yscale)


def _scaled_out_path(base: Path, yscale: str) -> Path:
    if yscale == "linear":
        return base
    return base.with_name(f"{base.stem}_{yscale}{base.suffix}")


def plot_bc_eval(data: dict, out_path: Path, yscale: str = "linear"):
    steps = data["steps"]
    w = max(3, len(steps) // 20)  # adaptive smoothing window

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 2 — BC Eval (clean forward pass)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, smooth(data["bc_nll"], w), "b-", alpha=0.85)
    ax.set_title("BC NLL (↓ better)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    _apply_scale(ax, yscale, data["bc_nll"])

    ax = axes[0, 1]
    ax.plot(steps, smooth(data["bc_std"], w), "m-", alpha=0.85)
    ax.set_title("Policy Std — confidence (↓ = more certain)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    _apply_scale(ax, yscale, data["bc_std"])

    ax = axes[0, 2]
    ax.plot(steps, smooth(data["act_mae"], w), "g-", alpha=0.85)
    ax.set_title("Action MAE — mean |pred − gt| (↓ better)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    _apply_scale(ax, yscale, data["act_mae"])

    ax = axes[1, 0]
    ax.plot(steps, smooth(data["rew_mse"], w), "r-", alpha=0.85)
    ax.set_title("Reward MSE (↓ better)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    _apply_scale(ax, yscale, data["rew_mse"])

    ax = axes[1, 1]
    ax.plot(steps, smooth(data["rew_corr"], w), "c-", alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title("Reward Pearson r (↑ better)")
    ax.set_xlabel("Step")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    _apply_scale(ax, yscale, data["rew_corr"])

    # Summary text panel
    ax = axes[1, 2]
    ax.axis("off")
    if len(steps) > 0:
        last = {k: v[-1] for k, v in data.items() if k != "steps"}
        best_nll = data["bc_nll"].min()
        best_corr = data["rew_corr"].max()
        summary = (
            f"Last step:  {steps[-1]:,}\n"
            f"\n"
            f"BC NLL     (last): {last['bc_nll']:.4f}\n"
            f"BC NLL     (best): {best_nll:.4f}\n"
            f"\n"
            f"Action MAE (last): {last['act_mae']:.4f}\n"
            f"Policy Std (last): {last['bc_std']:.3f}\n"
            f"\n"
            f"Reward MSE (last): {last['rew_mse']:.5f}\n"
            f"Reward r   (last): {last['rew_corr']:.3f}\n"
            f"Reward r   (best): {best_corr:.3f}\n"
        )
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot BC-eval curves from phase2.log.")
    parser.add_argument("run_dir", type=str, help="Run directory containing phase2.log")
    parser.add_argument("--yscale", type=str, default="linear", choices=["linear", "log", "symlog"],
                        help="Y-axis scale for line plots.")
    parser.add_argument("--both_scales", action="store_true",
                        help="Save both linear and log versions (adds *_log.png).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log = run_dir / "phase2.log"

    if not log.exists():
        print(f"No phase2.log found in {run_dir}")
        raise SystemExit(1)

    data = parse_bc_eval_log(log)
    if len(data["steps"]) == 0:
        print(f"No [bc_eval ...] lines found in {log}")
        print("Make sure you're running train_agent.py with --eval_every > 0")
        raise SystemExit(1)

    print(f"Found {len(data['steps'])} bc_eval entries "
          f"(steps {data['steps'][0]}–{data['steps'][-1]})")
    if args.both_scales:
        plot_bc_eval(data, run_dir / "bc_eval_curves.png", yscale="linear")
        plot_bc_eval(data, run_dir / "bc_eval_curves_log.png", yscale="log")
    else:
        out_path = _scaled_out_path(run_dir / "bc_eval_curves.png", args.yscale)
        plot_bc_eval(data, out_path, yscale=args.yscale)


if __name__ == "__main__":
    main()
