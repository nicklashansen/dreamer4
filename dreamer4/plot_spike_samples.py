#!/usr/bin/env python3
"""Visualize spike sample dumps (hard vs easy) from phase2 diagnostics.

Outputs two figures per spike step:
  1) overview: frame strips + reward/edge-fraction traces
  2) actions: per-action-channel target vs predicted (+/-2std band)
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _short_src(src) -> str:
    s = str(src)
    # source format is "data_dir|frames_dir"; keep only tail names
    if "|" in s:
        dd, fd = s.split("|", 1)
        return f"{Path(dd).name}|{Path(fd).name}"
    return Path(s).name


def _compact_meta(sample: Dict, tag: str, i: int) -> str:
    # Keep subplot titles short and scannable.
    sid = sample.get("source_id")
    ep = sample.get("episode_id")
    idx = sample.get("sample_idx")
    return f"{tag}{i} s{sid} ep{ep} i{idx}"


def _to_numpy_img_strip(obs_u8: torch.Tensor, n_frames: int = 6) -> np.ndarray:
    # obs_u8: (T+1,3,H,W) uint8
    t = obs_u8.shape[0]
    idxs = np.linspace(0, t - 1, num=min(n_frames, t), dtype=int)
    imgs = []
    for i in idxs:
        im = obs_u8[i].permute(1, 2, 0).cpu().numpy()  # HWC
        imgs.append(im)
    return np.concatenate(imgs, axis=1)


def _valid_dims(mask: np.ndarray) -> List[int]:
    valid = mask.mean(axis=0) > 0.5
    return np.where(valid)[0].tolist()


def _plot_group_overview(ax_row, sample: Dict, title_prefix: str) -> None:
    obs = sample["obs_u8"]  # (T+1,3,H,W)
    act = sample["act"].float().cpu().numpy()        # (T,A)
    mask = sample["act_mask"].float().cpu().numpy()  # (T,A)
    rew = sample["rew"].float().cpu().numpy()      # (T,)
    dims = _valid_dims(mask)

    strip = _to_numpy_img_strip(obs, n_frames=6)
    ax_row[0].imshow(strip)
    ax_row[0].set_title(f"{title_prefix} frames (valid dims: {dims})")
    ax_row[0].axis("off")

    # Quick aggregate action activity for overview readability
    act_mag = np.nanmean(np.abs(np.where(mask > 0.5, act, np.nan)), axis=1)
    ax_row[1].plot(act_mag, lw=1.2, color="tab:blue")
    ax_row[1].set_title(f"{title_prefix} mean |action| over valid dims")
    ax_row[1].set_xlabel("t")
    ax_row[1].grid(True, alpha=0.25)

    edge = (np.abs(act) > 0.98) * (mask > 0.5)
    edge_frac_t = edge.sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    ax_row[2].plot(rew, "g-", lw=1.5, label="reward")
    ax_row[2].plot(edge_frac_t, "r--", lw=1.2, label="edge_frac_t")
    ax_row[2].set_title(f"{title_prefix} reward + edge frac")
    ax_row[2].set_xlabel("t")
    ax_row[2].grid(True, alpha=0.25)
    ax_row[2].legend(fontsize=8)


def _plot_action_channel_single(
    ax,
    sample: Dict,
    title_prefix: str,
    d: int,
) -> None:
    act = sample["act"].float().cpu().numpy()        # (T,A)
    mask = sample["act_mask"].float().cpu().numpy()  # (T,A)
    pred_mean = sample.get("pred_mean_l0", None)
    pred_lo = sample.get("pred_lo_l0", None)
    pred_hi = sample.get("pred_hi_l0", None)
    if pred_mean is not None:
        pred_mean = pred_mean.float().cpu().numpy()
    if pred_lo is not None:
        pred_lo = pred_lo.float().cpu().numpy()
    if pred_hi is not None:
        pred_hi = pred_hi.float().cpu().numpy()

    if d >= act.shape[1]:
        ax.axis("off")
        return

    t = np.arange(act.shape[0])
    valid = mask[:, d] > 0.5
    y = act[:, d].copy()
    y[~valid] = np.nan
    ax.plot(t, y, lw=1.6, color="tab:blue", label=f"tgt a{d}")

    if pred_mean is not None and pred_lo is not None and pred_hi is not None:
        m = pred_mean[:, d].copy()
        lo = pred_lo[:, d].copy()
        hi = pred_hi[:, d].copy()
        m[~valid] = np.nan
        lo[~valid] = np.nan
        hi[~valid] = np.nan
        ax.plot(t, m, lw=1.1, color="tab:orange", linestyle="--", alpha=0.95, label=f"pred a{d}")
        ax.fill_between(t, lo, hi, color="tab:orange", alpha=0.12)

    ax.set_title(f"{title_prefix} a{d}")
    ax.set_xlabel("t")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="upper right")


def render_spike_file(path: Path, out_dir: Path, max_pairs: int = 3, max_action_dim: int = 16) -> List[Path]:
    d = torch.load(path, map_location="cpu")
    step = int(d.get("step", -1))
    hard: List[Dict] = d.get("hard_samples", [])
    easy: List[Dict] = d.get("easy_samples", [])
    # Backward compatibility: older snapshots only stored "samples" (hard outliers).
    if len(hard) == 0 and len(easy) == 0 and "samples" in d:
        hard = list(d.get("samples", []))
        easy = list(reversed(hard))
    k = min(max_pairs, len(hard), len(easy))
    if k <= 0:
        raise RuntimeError(f"No hard/easy samples in {path}")

    # Figure 1: overview
    fig, axes = plt.subplots(2 * k, 3, figsize=(15, 4.2 * k))
    if 2 * k == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for i in range(k):
        hs = hard[i]
        es = easy[i]
        h_meta = _compact_meta(hs, "H", i)
        e_meta = _compact_meta(es, "E", i)
        _plot_group_overview(axes[2 * i], hs, h_meta)
        _plot_group_overview(axes[2 * i + 1], es, e_meta)

    fig.suptitle(f"Spike Step {step}: Hard vs Easy Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_overview = out_dir / f"spike_step_{step:07d}_overview.png"
    fig.savefig(out_overview, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: action-channel contrasts (one column per action channel)
    all_samples = []
    for i in range(k):
        all_samples.append(hard[i])
        all_samples.append(easy[i])
    valid_union = set()
    for s in all_samples:
        mask = s["act_mask"].float().cpu().numpy()
        valid_union.update(_valid_dims(mask))
    dims = sorted([d for d in valid_union if d < max_action_dim])
    if len(dims) == 0:
        # Fallback: show first channels if masks are empty
        a = int(all_samples[0]["act"].shape[1])
        dims = list(range(min(a, max_action_dim)))
    ncols = max(1, len(dims))
    nrows = 2 * k
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(max(18, 3.0 * ncols), 2.9 * nrows), squeeze=False)
    for i in range(k):
        hs = hard[i]
        es = easy[i]
        h_meta = _compact_meta(hs, "H", i)
        e_meta = _compact_meta(es, "E", i)
        for j, d in enumerate(dims):
            _plot_action_channel_single(axes2[2 * i, j], hs, h_meta, d)
            _plot_action_channel_single(axes2[2 * i + 1, j], es, e_meta, d)

    fig2.suptitle(
        f"Spike Step {step}: Per-Channel Action Contrast (cols={len(dims)} channels)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out_actions = out_dir / f"spike_step_{step:07d}_actions.png"
    fig2.savefig(out_actions, dpi=170, bbox_inches="tight")
    plt.close(fig2)
    return [out_overview, out_actions]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot hard vs easy spike sample dumps.")
    p.add_argument("run_dir", type=str, help="Run directory with phase2/spike_samples/*.pt")
    p.add_argument("--spike_dir", type=str, default="", help="Explicit spike sample dir")
    p.add_argument("--out_dir", type=str, default="", help="Output directory for plots")
    p.add_argument("--max_files", type=int, default=20, help="Max spike files to render")
    p.add_argument("--max_pairs", type=int, default=3, help="Hard/easy pairs per figure")
    p.add_argument("--max_action_dim", type=int, default=16, help="Max action channels to visualize.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    spike_dir = Path(args.spike_dir) if args.spike_dir else (run_dir / "phase2" / "spike_samples")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "phase2_spike_visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(spike_dir.glob("spike_step_*.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No spike sample files found in {spike_dir}")
    files = files[: max(1, int(args.max_files))]

    for f in files:
        outs = render_spike_file(
            f,
            out_dir=out_dir,
            max_pairs=max(1, int(args.max_pairs)),
            max_action_dim=max(1, int(args.max_action_dim)),
        )
        for out in outs:
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
