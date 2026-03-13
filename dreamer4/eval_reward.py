#!/usr/bin/env python3
"""Evaluate reward model: correlation between predictions and targets.

Loads a Phase 2 checkpoint, runs the full pipeline (encode → dynamics → reward head)
on real data, and reports correlation + per-bucket accuracy.

Usage:
    python eval_reward.py /data/dreamer4/runs/walker-walk-v7/phase2/step_0000500.pt
"""
import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_imagination import load_finetuned_dynamics
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
from model import temporal_patchify, pack_bottleneck_to_spatial
from wm_dataset import WMDataset, collate_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase2_ckpt", type=str)
    p.add_argument("--tokenizer_ckpt", type=str, default="../logs/tokenizer.pt")
    p.add_argument("--data_dirs", nargs="+", default=[
        "/public/dreamer4/expert", "/public/dreamer4/mixed-small", "/public/dreamer4/mixed-large"])
    p.add_argument("--frame_dirs", nargs="+", default=[
        "/public/dreamer4/expert-shards", "/public/dreamer4/mixed-small-shards", "/public/dreamer4/mixed-large-shards"])
    p.add_argument("--tasks", nargs="+", default=["walker-walk"])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--packing_factor", type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device)

    # Frozen tokenizer
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device)
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    n_spatial = n_latents // args.packing_factor

    # Phase 2 checkpoint
    dyn, te, policy, rh, info = load_finetuned_dynamics(
        args.phase2_ckpt, device=device,
        d_bottleneck=d_bottleneck, n_latents=n_latents,
        packing_factor=args.packing_factor,
        override={"space_mode": "wm_agent"})
    for m in [dyn, te, rh]:
        m.eval()
    k_max = info["k_max"]
    emax = int(round(math.log2(k_max)))

    # Dataset
    ds = WMDataset(
        data_dir=args.data_dirs, frames_dir=args.frame_dirs,
        seq_len=args.seq_len, tasks=args.tasks,
        tasks_json=args.tasks_json, verbose=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_batch, drop_last=True)

    print(f"Evaluating {args.phase2_ckpt}")
    print(f"Data: {len(ds)} sequences, {args.num_batches} batches x {args.batch_size}")

    all_pred, all_tgt = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            obs = batch["obs"].to(device)
            act = batch["act"].to(device).clamp(-1, 1) * batch["act_mask"].to(device)
            rew = batch["rew"].to(device)
            emb_id = batch["emb_id"].to(device)
            mask = batch["act_mask"].to(device)

            frames = obs[:, :-1].float() / 255.0
            actions = torch.zeros_like(act)
            actions[:, 1:] = act[:, :-1]
            act_mask_shifted = torch.zeros_like(mask)
            act_mask_shifted[:, 1:] = mask[:, :-1]
            B, T = frames.shape[:2]

            patches = temporal_patchify(frames, patch)
            z, _ = encoder(patches)
            z1 = pack_bottleneck_to_spatial(z, n_spatial=n_spatial, k=args.packing_factor)

            agent_tokens = te(emb_id, B=B, T=T)
            step_idxs = torch.full((B, T), emax, device=device, dtype=torch.long)
            signal_idxs = torch.full((B, T), k_max - 1, device=device, dtype=torch.long)
            _, h_t = dyn(actions, step_idxs, signal_idxs, z1,
                         act_mask=act_mask_shifted, agent_tokens=agent_tokens)
            h = h_t[:, :, 0, :]

            pred = rh.predict(h, step=0)
            all_pred.append(pred.cpu().flatten())
            all_tgt.append(rew.cpu().flatten())

    pred = torch.cat(all_pred).numpy()
    tgt = torch.cat(all_tgt).numpy()
    valid = ~(np.isnan(pred) | np.isnan(tgt))
    pred, tgt = pred[valid], tgt[valid]

    corr = np.corrcoef(pred, tgt)[0, 1]
    print(f"\nN = {len(pred)}")
    print(f"Pred: mean={pred.mean():.3f}  std={pred.std():.3f}  min={pred.min():.3f}  max={pred.max():.3f}")
    print(f"Tgt:  mean={tgt.mean():.3f}  std={tgt.std():.3f}  min={tgt.min():.3f}  max={tgt.max():.3f}")
    print(f"Correlation: {corr:.4f}")
    print()

    buckets = [(0, 0.5, "low"), (0.5, 1.0, "mid"), (1.0, 1.5, "high"), (1.5, 2.1, "expert")]
    for lo, hi, label in buckets:
        m = (tgt >= lo) & (tgt < hi)
        if m.sum() > 0:
            print(f"  {label:7s} (tgt {lo:.1f}-{hi:.1f}): "
                  f"n={m.sum():5d}  pred_mean={pred[m].mean():.3f}  pred_std={pred[m].std():.3f}")


if __name__ == "__main__":
    main()
