#!/usr/bin/env python3
"""Plot training curves from Phase 2 and Phase 3 log files."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_phase2_log(path: Path) -> dict:
    """Parse phase2.log for training metrics."""
    steps, total, dyn, bc, rew = [], [], [], [], []
    flow, boot, std, r_pred, r_tgt = [], [], [], [], []

    pattern = re.compile(
        r"step (\d+) \| "
        r"total=([-\d.]+) dyn=([-\d.]+) bc=([-\d.]+) rew=([-\d.]+) "
        r"\| flow=([-\d.]+) boot=([-\d.]+) "
        r"\| std=([-\d.]+) r_pred=([-\d.]+) r_tgt=([-\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                total.append(float(m.group(2)))
                dyn.append(float(m.group(3)))
                bc.append(float(m.group(4)))
                rew.append(float(m.group(5)))
                flow.append(float(m.group(6)))
                boot.append(float(m.group(7)))
                std.append(float(m.group(8)))
                r_pred.append(float(m.group(9)))
                r_tgt.append(float(m.group(10)))

    return {
        "steps": np.array(steps),
        "total": np.array(total),
        "dynamics": np.array(dyn),
        "bc": np.array(bc),
        "reward": np.array(rew),
        "flow_mse": np.array(flow),
        "bootstrap_mse": np.array(boot),
        "policy_std": np.array(std),
        "reward_pred": np.array(r_pred),
        "reward_target": np.array(r_tgt),
    }


def parse_phase3_log(path: Path) -> dict:
    """Parse phase3.log for training metrics."""
    steps, total, pi, val = [], [], [], []
    adv_pos, R, r, V = [], [], [], []
    pi_std = []

    pattern = re.compile(
        r"step (\d+) \| "
        r"total=([-\d.]+) pi=([-\d.]+) val=([-\d.]+) "
        r"\| adv\+=([\d.]+) R=([-\d.]+) r=([-\d.]+) V=([-\d.]+)"
    )
    std_pattern = re.compile(r"std=([\d.]+) \|")

    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                total.append(float(m.group(2)))
                pi.append(float(m.group(3)))
                val.append(float(m.group(4)))
                adv_pos.append(float(m.group(5)))
                R.append(float(m.group(6)))
                r.append(float(m.group(7)))
                V.append(float(m.group(8)))
                sm = std_pattern.search(line)
                pi_std.append(float(sm.group(1)) if sm else float("nan"))

    return {
        "steps": np.array(steps),
        "total": np.array(total),
        "policy": np.array(pi),
        "value": np.array(val),
        "adv_pos_frac": np.array(adv_pos),
        "lambda_return": np.array(R),
        "mean_reward": np.array(r),
        "mean_value": np.array(V),
        "policy_std": np.array(pi_std),
    }


def smooth(y, window=3):
    if len(y) <= window:
        return y
    kernel = np.ones(window) / window
    # "valid" avoids edge artifacts but shortens the array
    smoothed = np.convolve(y, kernel, mode="valid")
    # pad start/end with original values to keep length
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.concatenate([y[:pad_left], smoothed, y[len(y) - pad_right:]]) if pad_right > 0 else np.concatenate([y[:pad_left], smoothed])


def smooth_x(x, window=3):
    return x


def plot_phase2(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 2: Agent Finetuning", fontsize=14, fontweight="bold")
    w = 5

    ax = axes[0, 0]
    ax.plot(smooth_x(data["steps"], w), smooth(data["total"], w), "k-", alpha=0.8)
    ax.set_title("Total Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(smooth_x(data["steps"], w), smooth(data["bc"], w), "b-", label="BC", alpha=0.8)
    ax.plot(smooth_x(data["steps"], w), smooth(data["reward"], w), "r-", label="Reward", alpha=0.8)
    ax.set_title("Head Losses")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(smooth_x(data["steps"], w), smooth(data["dynamics"], w), "g-", alpha=0.8)
    ax.set_title("Dynamics Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(smooth_x(data["steps"], w), smooth(data["flow_mse"], w), "m-", label="Flow MSE", alpha=0.8)
    ax.plot(smooth_x(data["steps"], w), smooth(data["bootstrap_mse"], w), "c-", label="Bootstrap MSE", alpha=0.8)
    ax.set_title("Flow / Bootstrap MSE")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(smooth_x(data["steps"], w), smooth(data["policy_std"], w), "b-", alpha=0.8)
    ax.set_title("Policy Std (mean)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(smooth_x(data["steps"], w), smooth(data["reward_pred"], w), "r-", label="Predicted", alpha=0.8)
    ax.plot(smooth_x(data["steps"], w), smooth(data["reward_target"], w), "k--", label="Target", alpha=0.5)
    ax.set_title("Reward Pred vs Target")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_phase3(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 3: Imagination RL", fontsize=14, fontweight="bold")
    w = 5

    ax = axes[0, 0]
    ax.plot(smooth_x(data["steps"], w), smooth(data["total"], w), "k-", alpha=0.8)
    ax.set_title("Total Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(smooth_x(data["steps"], w), smooth(data["policy"], w), "b-", label="Policy", alpha=0.8)
    ax.plot(smooth_x(data["steps"], w), smooth(data["value"], w), "r-", label="Value", alpha=0.8)
    ax.set_title("Policy / Value Losses")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(smooth_x(data["steps"], w), smooth(data["adv_pos_frac"], w), "g-", alpha=0.8)
    ax.set_title("Advantage Positive Fraction")
    ax.set_xlabel("Step")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(smooth_x(data["steps"], w), smooth(data["lambda_return"], w), "m-", alpha=0.8)
    ax.set_title("Mean Lambda Return")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(smooth_x(data["steps"], w), smooth(data["mean_reward"], w), "r-", alpha=0.8)
    ax.set_title("Mean Imagined Reward")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if not np.all(np.isnan(data.get("policy_std", []))):
        ax.plot(smooth_x(data["steps"], w), smooth(data["policy_std"], w), "b-", alpha=0.8)
        ax.set_title("Policy Std")
    else:
        ax.plot(smooth_x(data["steps"], w), smooth(data["mean_value"], w), "b-", alpha=0.8)
        ax.set_title("Mean Value Estimate")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    p2_log = run_dir / "phase2.log"
    p3_log = run_dir / "phase3.log"

    if p2_log.exists():
        data = parse_phase2_log(p2_log)
        if len(data["steps"]) > 0:
            plot_phase2(data, run_dir / "phase2_curves.png")
        else:
            print(f"No data points found in {p2_log}")

    if p3_log.exists():
        data = parse_phase3_log(p3_log)
        if len(data["steps"]) > 0:
            plot_phase3(data, run_dir / "phase3_curves.png")
        else:
            print(f"No data points found in {p3_log}")

    if not p2_log.exists() and not p3_log.exists():
        print(f"No log files found in {run_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
