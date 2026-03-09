#!/usr/bin/env python3
"""Visualize Phase-2 diagnostics JSONL produced by agent_diagnostics.py."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe(v: Any, default: float = np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if isinstance(ev, dict) and ev.get("kind") == "phase2_diagnostics":
                    events.append(ev)
            except json.JSONDecodeError:
                continue
    return events


def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) <= w or w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / w
    y = np.convolve(x, k, mode="valid")
    pad_left = (w - 1) // 2
    pad_right = w - 1 - pad_left
    if pad_right > 0:
        return np.concatenate([x[:pad_left], y, x[-pad_right:]])
    return np.concatenate([x[:pad_left], y])


def _agg_group_from_events(events: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    # key is "by_task" or "by_source"
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "count_events": 0.0,
        "count_samples": 0.0,
        "bc_mean_sum": 0.0,
        "bc_p95_sum": 0.0,
        "bc_max_sum": 0.0,
        "edge_frac_sum": 0.0,
        "std_mean_sum": 0.0,
    })
    for ev in events:
        rows = ev.get(key, [])
        if not isinstance(rows, list):
            continue
        for r in rows:
            name = str(r.get("name", "unknown"))
            s = stats[name]
            s["count_events"] += 1
            s["count_samples"] += _safe(r.get("count"), 0.0)
            s["bc_mean_sum"] += _safe(r.get("bc_mean"))
            s["bc_p95_sum"] += _safe(r.get("bc_p95"))
            s["bc_max_sum"] += _safe(r.get("bc_max"))
            s["edge_frac_sum"] += _safe(r.get("edge_frac_mean"))
            s["std_mean_sum"] += _safe(r.get("std_mean"))
    out = []
    for name, s in stats.items():
        n = max(1.0, s["count_events"])
        out.append({
            "name": name,
            "events": int(s["count_events"]),
            "samples_total": int(s["count_samples"]),
            "bc_mean_avg": s["bc_mean_sum"] / n,
            "bc_p95_avg": s["bc_p95_sum"] / n,
            "bc_max_avg": s["bc_max_sum"] / n,
            "edge_frac_avg": s["edge_frac_sum"] / n,
            "std_mean_avg": s["std_mean_sum"] / n,
        })
    out.sort(key=lambda x: x["bc_mean_avg"], reverse=True)
    return out


def make_plot(events: List[Dict[str, Any]], out_png: Path, smooth_w: int = 5, yscale: str = "log") -> None:
    steps = np.array([int(ev.get("step", 0)) for ev in events], dtype=np.int64)
    bc = np.array([_safe(ev.get("loss", {}).get("bc")) for ev in events], dtype=np.float64)
    bc_thr = np.array([_safe(ev.get("spike_threshold")) for ev in events], dtype=np.float64)
    total = np.array([_safe(ev.get("loss", {}).get("total")) for ev in events], dtype=np.float64)
    dyn = np.array([_safe(ev.get("loss", {}).get("dyn")) for ev in events], dtype=np.float64)
    rew = np.array([_safe(ev.get("loss", {}).get("rew")) for ev in events], dtype=np.float64)
    flow = np.array([_safe(ev.get("loss", {}).get("flow_mse")) for ev in events], dtype=np.float64)
    boot = np.array([_safe(ev.get("loss", {}).get("boot_mse")) for ev in events], dtype=np.float64)

    bc_mean = np.array([_safe(ev.get("batch", {}).get("bc_mean")) for ev in events], dtype=np.float64)
    bc_p95 = np.array([_safe(ev.get("batch", {}).get("bc_p95")) for ev in events], dtype=np.float64)
    bc_max = np.array([_safe(ev.get("batch", {}).get("bc_max")) for ev in events], dtype=np.float64)
    std_mean = np.array([_safe(ev.get("batch", {}).get("std_mean")) for ev in events], dtype=np.float64)
    std_min = np.array([_safe(ev.get("batch", {}).get("std_min")) for ev in events], dtype=np.float64)
    std_max = np.array([_safe(ev.get("batch", {}).get("std_max")) for ev in events], dtype=np.float64)
    edge_mean = np.array([_safe(ev.get("batch", {}).get("edge_frac_mean")) for ev in events], dtype=np.float64)
    valid_mean = np.array([_safe(ev.get("batch", {}).get("valid_frac_mean")) for ev in events], dtype=np.float64)
    mu_err = np.array([_safe(ev.get("batch", {}).get("mu_err_mean")) for ev in events], dtype=np.float64)
    z_err = np.array([_safe(ev.get("batch", {}).get("z_err_mean")) for ev in events], dtype=np.float64)
    h_norm = np.array([_safe(ev.get("batch", {}).get("h_norm_mean")) for ev in events], dtype=np.float64)
    h_p95 = np.array([_safe(ev.get("batch", {}).get("h_norm_p95")) for ev in events], dtype=np.float64)

    spikes = np.array([1 if bool(ev.get("is_spike", False)) else 0 for ev in events], dtype=np.int64)
    spike_steps = steps[spikes == 1]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("Phase-2 BC Diagnostics Summary", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, moving_avg(bc, smooth_w), "r-", label="bc")
    ax.plot(steps, moving_avg(bc_thr, smooth_w), "k--", label="spike_thr")
    if len(spike_steps) > 0:
        ax.vlines(spike_steps, ymin=np.nanmin(bc), ymax=np.nanmax(bc), colors="orange", alpha=0.15, linewidth=1)
    ax.set_title("BC vs Spike Threshold")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[0, 1]
    ax.plot(steps, moving_avg(bc_mean, smooth_w), "b-", label="bc_mean")
    ax.plot(steps, moving_avg(bc_p95, smooth_w), "m-", label="bc_p95")
    ax.plot(steps, moving_avg(bc_max, smooth_w), "r-", label="bc_max")
    ax.set_title("Batch BC Distribution")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[0, 2]
    ax.plot(steps, moving_avg(total, smooth_w), "k-", label="total")
    ax.plot(steps, moving_avg(dyn, smooth_w), "g-", label="dyn")
    ax.plot(steps, moving_avg(rew, smooth_w), "c-", label="rew")
    ax.set_title("Losses (Logged)")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[1, 0]
    ax.plot(steps, moving_avg(std_mean, smooth_w), "b-", label="std_mean")
    ax.plot(steps, moving_avg(std_min, smooth_w), "k--", label="std_min")
    ax.plot(steps, moving_avg(std_max, smooth_w), "r--", label="std_max")
    ax.set_title("Policy Std Stats")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[1, 1]
    ax.plot(steps, moving_avg(edge_mean, smooth_w), "r-", label="edge_frac")
    ax.plot(steps, moving_avg(valid_mean, smooth_w), "g-", label="valid_frac")
    ax.set_title("Action Pathology")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[1, 2]
    ax.plot(steps, moving_avg(mu_err, smooth_w), "m-", label="|mu-a|")
    ax.plot(steps, moving_avg(z_err, smooth_w), "c-", label="|atanh(a)-mu|/std")
    ax.set_title("Policy-Target Mismatch")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[2, 0]
    ax.plot(steps, moving_avg(h_norm, smooth_w), "g-", label="h_norm_mean")
    ax.plot(steps, moving_avg(h_p95, smooth_w), "k--", label="h_norm_p95")
    ax.set_title("Representation Norms")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[2, 1]
    ax.plot(steps, moving_avg(flow, smooth_w), "m-", label="flow")
    ax.plot(steps, moving_avg(boot, smooth_w), "c-", label="boot")
    ax.set_title("Dynamics Aux")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if yscale != "linear":
        ax.set_yscale(yscale)

    ax = axes[2, 2]
    if len(spike_steps) > 0:
        bins = min(30, max(5, len(spike_steps) // 2))
        ax.hist(spike_steps, bins=bins, color="orange", alpha=0.8)
    ax.set_title(f"Spike Step Histogram (n={int(spikes.sum())})")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_summary(events: List[Dict[str, Any]], out_txt: Path, topn: int = 10) -> None:
    steps = [int(ev.get("step", 0)) for ev in events]
    spikes = [ev for ev in events if bool(ev.get("is_spike", False))]
    bc = [_safe(ev.get("loss", {}).get("bc")) for ev in events]
    bc_thr = [_safe(ev.get("spike_threshold")) for ev in events]
    top_outlier_tasks = Counter()
    top_outlier_sources = Counter()
    top_outlier_samples = Counter()

    for ev in spikes:
        to = ev.get("top_outliers", [])
        if isinstance(to, list) and len(to) > 0:
            for row in to:
                t = str(row.get("task", "unknown"))
                s = str(row.get("source", "unknown"))
                sid = int(_safe(row.get("sample_idx"), -1))
                top_outlier_tasks[t] += 1
                top_outlier_sources[s] += 1
                top_outlier_samples[(t, s, sid)] += 1

    by_task = _agg_group_from_events(events, "by_task")
    by_source = _agg_group_from_events(events, "by_source")

    lines: List[str] = []
    lines.append("# Phase-2 Diagnostics Summary")
    lines.append("")
    lines.append(f"- Events parsed: {len(events)}")
    lines.append(f"- Step range: {min(steps) if steps else 'n/a'} -> {max(steps) if steps else 'n/a'}")
    lines.append(f"- Spike events: {len(spikes)} ({(100.0 * len(spikes) / max(1, len(events))):.2f}%)")
    if len(bc) > 0:
        lines.append(f"- BC mean/max: {np.nanmean(np.array(bc)):.4f} / {np.nanmax(np.array(bc)):.4f}")
    if len(bc_thr) > 0:
        lines.append(f"- Spike threshold mean: {np.nanmean(np.array(bc_thr)):.4f}")
    lines.append("")

    lines.append("## Worst Tasks by Mean BC (event-avg)")
    for r in by_task[:topn]:
        lines.append(
            f"- {r['name']}: bc_mean={r['bc_mean_avg']:.4f} bc_p95={r['bc_p95_avg']:.4f} "
            f"edge={r['edge_frac_avg']:.3f} std={r['std_mean_avg']:.3f} events={r['events']}"
        )
    lines.append("")

    lines.append("## Worst Sources by Mean BC (event-avg)")
    for r in by_source[:topn]:
        lines.append(
            f"- {r['name']}: bc_mean={r['bc_mean_avg']:.4f} bc_p95={r['bc_p95_avg']:.4f} "
            f"edge={r['edge_frac_avg']:.3f} std={r['std_mean_avg']:.3f} events={r['events']}"
        )
    lines.append("")

    lines.append("## Most Frequent Spike Outliers (Task)")
    for name, c in top_outlier_tasks.most_common(topn):
        lines.append(f"- {name}: {c}")
    lines.append("")

    lines.append("## Most Frequent Spike Outliers (Source)")
    for name, c in top_outlier_sources.most_common(topn):
        lines.append(f"- {name}: {c}")
    lines.append("")

    lines.append("## Most Repeated Exact Samples (task, source, sample_idx)")
    for (t, s, sid), c in top_outlier_samples.most_common(topn):
        lines.append(f"- ({t}, {s}, {sid}): {c}")
    lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot diagnostics JSONL from Phase-2 BC diagnostics hook.")
    p.add_argument("run_dir", type=str, help="Run directory containing phase2/diagnostics.jsonl")
    p.add_argument("--jsonl", type=str, default="", help="Explicit diagnostics JSONL path.")
    p.add_argument("--out_png", type=str, default="", help="Output PNG path.")
    p.add_argument("--out_txt", type=str, default="", help="Output text summary path.")
    p.add_argument("--smooth", type=int, default=5, help="Smoothing window for curves.")
    p.add_argument("--topn", type=int, default=10, help="Top-N rows in text summary.")
    p.add_argument("--yscale", type=str, default="log", choices=["linear", "log", "symlog"],
                   help="Y-axis scale for line plots.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    jsonl = Path(args.jsonl) if args.jsonl else (run_dir / "phase2" / "diagnostics.jsonl")
    out_png = Path(args.out_png) if args.out_png else (run_dir / "phase2_diagnostics.png")
    out_txt = Path(args.out_txt) if args.out_txt else (run_dir / "phase2_diagnostics_summary.txt")

    if not jsonl.exists():
        raise FileNotFoundError(f"Diagnostics JSONL not found: {jsonl}")

    events = load_events(jsonl)
    if len(events) == 0:
        raise RuntimeError(f"No valid diagnostics events found in: {jsonl}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    make_plot(events, out_png=out_png, smooth_w=max(1, int(args.smooth)), yscale=args.yscale)
    write_summary(events, out_txt=out_txt, topn=max(1, int(args.topn)))

    print(f"Saved plot: {out_png}")
    print(f"Saved summary: {out_txt}")
    print(f"Parsed events: {len(events)} from {jsonl}")


if __name__ == "__main__":
    main()
