#!/usr/bin/env python3
"""Test whether the dynamics model is sensitive to action inputs.

Loads a Phase 2 checkpoint, encodes a real context sequence, then:
1. Runs the dynamics with the ORIGINAL actions → gets latent z1 and h_t
2. Runs the dynamics with RANDOM actions → gets latent z2 and h_t'
3. Runs the dynamics with ZERO actions → gets latent z3 and h_t''
4. Runs denoising rollout with different actions → checks if z_next differs
5. Checks reward head predictions for different h_t

This tells us whether actions actually influence the model's predictions.
"""
import sys
import math
import torch
import numpy as np
from pathlib import Path

torch.manual_seed(42)

from model import (
    Encoder, Decoder, Tokenizer, Dynamics, TaskEmbedder,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from agent import PolicyHead, RewardHead
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt, make_tau_schedule
from train_imagination import load_phase2 as _load_phase2
from wm_dataset import WMDataset, collate_batch
from torch.utils.data import DataLoader
from torch.amp import autocast

DEVICE = torch.device("cuda:0")


def load_phase2(ckpt_path, device, space_mode="wm_agent"):
    """Load Phase 2 using the same loader as train_imagination.py."""
    d_bottleneck = 32
    n_latents = 16
    packing_factor = 2
    dyn, task_emb, policy, reward_head, info = _load_phase2(
        ckpt_path, device=device,
        d_bottleneck=d_bottleneck, n_latents=n_latents, packing_factor=packing_factor,
        space_mode_override=space_mode,
    )
    dyn.eval()
    task_emb.eval()
    policy.eval()
    reward_head.eval()
    info["d_bottleneck"] = d_bottleneck
    info["n_latents"] = n_latents
    return dyn, task_emb, policy, reward_head, info


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/data/dreamer4/runs/walker-walk/phase2/final.pt"
    tok_path = sys.argv[2] if len(sys.argv) > 2 else "../logs/tokenizer.pt"

    print(f"Loading Phase 2 checkpoint: {ckpt_path}")
    dyn, task_emb, policy, reward_head, info = load_phase2(ckpt_path, DEVICE)
    d_model = info["d_model"]
    k_max = info["k_max"]
    action_dim = info["action_dim"]
    n_spatial = info["n_spatial"]
    d_spatial = info["d_spatial"]
    emax = int(round(math.log2(k_max)))

    print(f"Loading tokenizer: {tok_path}")
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(tok_path, device=DEVICE)
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    packing_factor = 2

    # Load one real batch
    print("Loading data...")
    dataset = WMDataset(
        data_dir=["/public/dreamer4/expert"],
        frames_dir=["/public/dreamer4/expert-shards"],
        seq_len=16, img_size=128, action_dim=16,
        tasks=["walker-walk"], verbose=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,
                        collate_fn=collate_batch)
    batch = next(iter(loader))

    obs = batch["obs"].to(DEVICE)       # (B, T+1, 3, H, W)
    act = batch["act"].to(DEVICE)       # (B, T, 16)
    mask = batch["act_mask"].to(DEVICE) # (B, T, 16)
    emb_id = batch["emb_id"].to(DEVICE) # (B,)

    act = act.clamp(-1, 1) * mask
    frames = obs[:, :-1].float() / 255.0
    B, T = frames.shape[:2]

    # Shift actions (a_t at position t+1)
    actions_shifted = torch.zeros_like(act)
    actions_shifted[:, 1:] = act[:, :-1]
    mask_shifted = torch.zeros_like(mask)
    mask_shifted[:, 1:] = mask[:, :-1]

    # Encode frames
    with torch.no_grad():
        patches = temporal_patchify(frames, patch)
        z_btLd, _ = encoder(patches)
        z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)
        agent_tokens = task_emb(emb_id, B=B, T=T)

    step_idxs = torch.full((B, T), emax, device=DEVICE, dtype=torch.long)
    signal_idxs = torch.full((B, T), k_max - 1, device=DEVICE, dtype=torch.long)

    print(f"\nz1 shape: {z1.shape}")
    print(f"actions shape: {actions_shifted.shape}")
    print(f"agent_tokens shape: {agent_tokens.shape}")

    # ===================================================================
    # TEST 1: Same context, different actions → different h_t?
    # ===================================================================
    print("\n" + "="*60)
    print("TEST 1: Does action input change h_t (clean forward)?")
    print("="*60)

    with torch.no_grad():
        # Original actions
        _, h_orig = dyn(actions_shifted, step_idxs, signal_idxs, z1,
                        act_mask=mask_shifted, agent_tokens=agent_tokens)

        # Random actions
        rand_act = torch.randn_like(actions_shifted).clamp(-1, 1) * mask
        _, h_rand = dyn(rand_act, step_idxs, signal_idxs, z1,
                        act_mask=mask_shifted, agent_tokens=agent_tokens)

        # Zero actions
        zero_act = torch.zeros_like(actions_shifted)
        _, h_zero = dyn(zero_act, step_idxs, signal_idxs, z1,
                        act_mask=mask_shifted, agent_tokens=agent_tokens)

    h_o = h_orig[:, :, 0, :]  # (B, T, d_model)
    h_r = h_rand[:, :, 0, :]
    h_z = h_zero[:, :, 0, :]

    diff_rand = (h_o - h_r).abs()
    diff_zero = (h_o - h_z).abs()

    print(f"  h_orig vs h_rand — mean diff: {diff_rand.mean():.6f}, max: {diff_rand.max():.6f}")
    print(f"  h_orig vs h_zero — mean diff: {diff_zero.mean():.6f}, max: {diff_zero.max():.6f}")
    print(f"  h_orig norm:  mean={h_o.norm(dim=-1).mean():.4f}")
    print(f"  Relative diff (rand): {(diff_rand.mean() / h_o.norm(dim=-1).mean()):.6f}")
    print(f"  Relative diff (zero): {(diff_zero.mean() / h_o.norm(dim=-1).mean()):.6f}")

    # Per-timestep breakdown
    print("\n  Per-timestep h_t diff (orig vs rand):")
    for t in range(min(T, 8)):
        d = (h_o[:, t] - h_r[:, t]).abs().mean().item()
        n = h_o[:, t].norm(dim=-1).mean().item()
        print(f"    t={t}: abs_diff={d:.6f}, rel_diff={d/n:.6f}")

    # ===================================================================
    # TEST 2: Does action affect reward predictions?
    # ===================================================================
    print("\n" + "="*60)
    print("TEST 2: Does h_t difference affect reward predictions?")
    print("="*60)

    with torch.no_grad():
        r_orig = reward_head.predict(h_o, step=0)
        r_rand = reward_head.predict(h_r, step=0)
        r_zero = reward_head.predict(h_z, step=0)

    print(f"  Reward (orig actions): mean={r_orig.mean():.4f}, std={r_orig.std():.4f}")
    print(f"  Reward (rand actions): mean={r_rand.mean():.4f}, std={r_rand.std():.4f}")
    print(f"  Reward (zero actions): mean={r_zero.mean():.4f}, std={r_zero.std():.4f}")
    print(f"  Reward diff (orig vs rand): {(r_orig - r_rand).abs().mean():.6f}")
    print(f"  Reward diff (orig vs zero): {(r_orig - r_zero).abs().mean():.6f}")

    # ===================================================================
    # TEST 3: Does action affect denoised next latent?
    # ===================================================================
    print("\n" + "="*60)
    print("TEST 3: Does action affect denoised next latent z_{t+1}?")
    print("="*60)

    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=0.25)
    K = int(sched["K"])
    tau_sched = sched["tau"]
    tau_idx_sched = sched["tau_idx"]
    e = int(sched["e"])
    dt = float(sched["dt"])

    # Use first 8 timesteps as context, denoise step 9
    ctx_len = 8
    z_ctx = z1[:, :ctx_len]
    a_ctx_orig = actions_shifted[:, :ctx_len]
    a_ctx_rand = rand_act[:, :ctx_len]
    m_ctx = mask_shifted[:, :ctx_len]
    ag_ctx = agent_tokens[:, :ctx_len]

    # Action at step ctx_len (the action that produces z_{ctx_len+1})
    a_new_orig = act[:, ctx_len-1:ctx_len]  # (B, 1, 16) — actual next action
    a_new_rand = torch.randn(B, 1, 16, device=DEVICE).clamp(-1, 1) * mask[:, :1]
    a_new_zero = torch.zeros(B, 1, 16, device=DEVICE)

    def denoise_next(a_history, a_new):
        """Denoise z_{ctx_len+1} given action history + new action."""
        a_full = torch.cat([a_history, a_new], dim=1)
        m_full = torch.cat([m_ctx, mask[:, :1]], dim=1)
        ag_full = torch.cat([ag_ctx, ag_ctx[:, -1:]], dim=1)

        z = torch.randn(B, 1, n_spatial, d_spatial, device=DEVICE)

        step_ids = torch.full((B, ctx_len + 1), emax, device=DEVICE, dtype=torch.long)
        step_ids[:, -1] = e
        sig_ids = torch.full((B, ctx_len + 1), k_max - 1, device=DEVICE, dtype=torch.long)

        # Use SAME initial noise for fair comparison
        z_init = torch.randn(B, 1, n_spatial, d_spatial, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(123))

        z = z_init.clone()
        with torch.no_grad():
            for i in range(K):
                tau_i = float(tau_sched[i])
                sig_i = int(tau_idx_sched[i])
                sig_ids[:, -1] = sig_i

                packed = torch.cat([z_ctx, z], dim=1)
                with autocast(device_type="cuda"):
                    x1_hat, _ = dyn(a_full, step_ids, sig_ids, packed,
                                    act_mask=m_full, agent_tokens=ag_full)
                x1_hat_last = x1_hat[:, -1:]
                denom = max(1e-4, 1.0 - tau_i)
                b = (x1_hat_last.float() - z.float()) / denom
                z = (z.float() + b * dt).to(z_init.dtype)

        return z  # (B, 1, n_spatial, d_spatial)

    z_next_orig = denoise_next(a_ctx_orig, a_new_orig)
    z_next_rand = denoise_next(a_ctx_orig, a_new_rand)  # same history, different new action
    z_next_zero = denoise_next(a_ctx_orig, a_new_zero)
    z_next_diff_hist = denoise_next(a_ctx_rand, a_new_orig)  # different history, same new action

    diff_new_act = (z_next_orig - z_next_rand).abs()
    diff_zero_act = (z_next_orig - z_next_zero).abs()
    diff_hist = (z_next_orig - z_next_diff_hist).abs()

    print(f"  z_next norm: {z_next_orig.norm():.4f}")
    print(f"  Same history, diff NEW action:")
    print(f"    abs diff: mean={diff_new_act.mean():.6f}, max={diff_new_act.max():.6f}")
    print(f"    relative: {diff_new_act.mean() / z_next_orig.abs().mean():.6f}")
    print(f"  Same history, ZERO new action:")
    print(f"    abs diff: mean={diff_zero_act.mean():.6f}, max={diff_zero_act.max():.6f}")
    print(f"    relative: {diff_zero_act.mean() / z_next_orig.abs().mean():.6f}")
    print(f"  Diff history, same new action:")
    print(f"    abs diff: mean={diff_hist.mean():.6f}, max={diff_hist.max():.6f}")
    print(f"    relative: {diff_hist.mean() / z_next_orig.abs().mean():.6f}")

    # ===================================================================
    # TEST 4: Does policy sample different actions for different h_t?
    # ===================================================================
    print("\n" + "="*60)
    print("TEST 4: Policy action diversity")
    print("="*60)

    with torch.no_grad():
        # Sample multiple times from same h_t
        h_sample = h_o[:, 4:5]  # (B, 1, d_model) — single timestep
        samples = [policy.sample(h_sample, step=0).cpu() for _ in range(10)]
        samples = torch.stack(samples)  # (10, B, 1, action_dim)

        print(f"  Action samples from same h_t (10 draws):")
        print(f"    mean: {samples.mean(0).squeeze()}")
        print(f"    std:  {samples.std(0).squeeze()}")

        logits = policy(h_o[:, 4:5])
        dist = policy.dist(h_o[:, 4:5], step=0)
        print(f"  Policy mean (expected): {dist.mean[0, 0].cpu().tolist()}")
        print(f"  Policy entropy: {dist.entropy[0, 0].item():.4f}")

    # ===================================================================
    # TEST 5: Dynamics x1_hat sensitivity (before denoising integration)
    # ===================================================================
    print("\n" + "="*60)
    print("TEST 5: Raw x1_hat prediction sensitivity to action")
    print("="*60)

    # Just one denoising step — how much does x1_hat change with action?
    z_noise = torch.randn(B, 1, n_spatial, d_spatial, device=DEVICE)
    packed = torch.cat([z_ctx, z_noise], dim=1)
    step_ids = torch.full((B, ctx_len + 1), emax, device=DEVICE, dtype=torch.long)
    step_ids[:, -1] = e
    sig_ids = torch.full((B, ctx_len + 1), k_max - 1, device=DEVICE, dtype=torch.long)
    sig_ids[:, -1] = int(tau_idx_sched[0])
    m_full = torch.cat([m_ctx, mask[:, :1]], dim=1)
    ag_full = torch.cat([ag_ctx, ag_ctx[:, -1:]], dim=1)

    with torch.no_grad():
        a_full_orig = torch.cat([a_ctx_orig, a_new_orig], dim=1)
        a_full_rand = torch.cat([a_ctx_orig, a_new_rand], dim=1)

        with autocast(device_type="cuda"):
            x1_orig, h_orig2 = dyn(a_full_orig, step_ids, sig_ids, packed,
                                    act_mask=m_full, agent_tokens=ag_full)
            x1_rand, h_rand2 = dyn(a_full_rand, step_ids, sig_ids, packed,
                                    act_mask=m_full, agent_tokens=ag_full)

    x1_diff = (x1_orig[:, -1:] - x1_rand[:, -1:]).abs()
    h_diff2 = (h_orig2[:, -1:, 0] - h_rand2[:, -1:, 0]).abs()

    print(f"  x1_hat diff (last step): mean={x1_diff.mean():.6f}, max={x1_diff.max():.6f}")
    print(f"  x1_hat norm: {x1_orig[:, -1:].abs().mean():.4f}")
    print(f"  x1_hat relative diff: {x1_diff.mean() / x1_orig[:, -1:].abs().mean():.6f}")
    print(f"  h_t diff (last step): mean={h_diff2.mean():.6f}, max={h_diff2.max():.6f}")
    print(f"  h_t relative diff: {h_diff2.mean() / h_orig2[:, -1:, 0].abs().mean():.6f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    h_rel = diff_rand.mean() / h_o.norm(dim=-1).mean()
    z_rel = diff_new_act.mean() / z_next_orig.abs().mean()
    r_diff = (r_orig - r_rand).abs().mean()

    if h_rel < 1e-4:
        print("  ⚠ h_t is INSENSITIVE to actions (rel diff < 1e-4)")
    elif h_rel < 1e-2:
        print(f"  ⚡ h_t has WEAK action sensitivity (rel diff = {h_rel:.6f})")
    else:
        print(f"  ✓ h_t is action-sensitive (rel diff = {h_rel:.6f})")

    if z_rel < 1e-4:
        print("  ⚠ Denoised z_next is INSENSITIVE to actions")
    elif z_rel < 1e-2:
        print(f"  ⚡ Denoised z_next has WEAK action sensitivity (rel diff = {z_rel:.6f})")
    else:
        print(f"  ✓ Denoised z_next is action-sensitive (rel diff = {z_rel:.6f})")

    if r_diff < 0.01:
        print(f"  ⚠ Reward predictions UNCHANGED by actions (diff = {r_diff:.6f})")
    elif r_diff < 0.1:
        print(f"  ⚡ Reward predictions WEAKLY change with actions (diff = {r_diff:.4f})")
    else:
        print(f"  ✓ Reward predictions change with actions (diff = {r_diff:.4f})")


if __name__ == "__main__":
    main()
