#!/usr/bin/env python3
"""Phase 3: Imagination RL — PMPO + TD(λ) in imagined rollouts.

Loads Phase 2 checkpoint (dynamics + task_embedder + policy + reward heads),
freezes dynamics/task_embedder/reward, trains policy + value head via:
  1. Imagined rollouts in latent space (frozen dynamics, K denoising steps)
  2. TD(λ) value targets from imagined rewards + bootstrapped values
  3. PMPO policy loss (sign-split advantages + KL to behavioral prior)

Key: h_t is detached before the policy head (stop_gradient), but policy
parameters remain differentiable through the sampled actions.

Usage:
    python dreamer4/train_imagination.py \
        --tokenizer_ckpt logs/tokenizer.pt \
        --phase2_ckpt logs/agent_ckpts/final.pt \
        --data_dirs /public/dreamer4/expert \
        --frame_dirs /public/dreamer4/expert-shards
"""
import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from task_set import TASK_SET
from wm_dataset import WMDataset, collate_batch
from model import (
    Encoder, Decoder, Tokenizer, Dynamics, TaskEmbedder,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from agent import PolicyHead, RewardHead, ValueHead, compute_lambda_returns, pmpo_loss, ppo_loss, reinforce_loss
from train_dynamics import (
    get_dist_info, is_rank0, seed_everything, worker_init_fn,
    init_distributed, load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_ckpt(
    path: Path,
    *,
    step: int,
    epoch: int,
    policy: PolicyHead,
    value: ValueHead,
    opt,
    scaler,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(
    path: Path,
    *,
    policy: PolicyHead,
    value: ValueHead,
    opt,
    scaler,
) -> tuple:
    ckpt = torch.load(path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"], strict=True)
    value.load_state_dict(ckpt["value"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    saved_args = ckpt.get("args", {}) or {}
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0)), saved_args


def load_finetuned_dynamics(
    ckpt_path: str,
    *,
    device: torch.device,
    d_bottleneck: int,
    n_latents: int,
    packing_factor: int,
    override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dynamics, TaskEmbedder, PolicyHead, RewardHead, dict]:
    """Load Phase 2 checkpoint: dynamics + task_embedder + policy + reward."""
    override = override or {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    def _resolved_int(name: str, default: int) -> int:
        return int(override.get(name, a.get(name, default)))

    def _resolved_float(name: str, default: float) -> float:
        return float(override.get(name, a.get(name, default)))

    def _resolved_str(name: str, default: str) -> str:
        return str(override.get(name, a.get(name, default)))

    d_model = _resolved_int("d_model_dyn", 512)
    n_heads = _resolved_int("n_heads", 4)
    depth = _resolved_int("dyn_depth", 8)
    dropout = _resolved_float("dropout", 0.0)
    mlp_ratio = _resolved_float("mlp_ratio", 4.0)
    time_every = _resolved_int("time_every", 1)
    k_max = _resolved_int("k_max", 8)
    n_register = _resolved_int("n_register", 4)
    n_agent = _resolved_int("n_agent", 1)
    space_mode = _resolved_str("space_mode", "wm_agent")
    action_dim = _resolved_int("action_dim", 16)
    mtp_length = _resolved_int("mtp_length", 8)
    head_hidden = int(d_model * _resolved_float("head_mlp_ratio", 2.0))

    if n_latents % packing_factor != 0:
        raise ValueError(
            f"Incompatible tokenizer/dynamics packing: n_latents={n_latents} "
            f"is not divisible by packing_factor={packing_factor}."
        )
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    # Optional metadata checks from the phase2 checkpoint.
    if a.get("n_latents") is not None and int(a["n_latents"]) != int(n_latents):
        raise ValueError(
            f"Tokenizer/phase2 mismatch: tokenizer n_latents={n_latents}, "
            f"but phase2 ckpt expects n_latents={int(a['n_latents'])}."
        )
    if a.get("d_bottleneck") is not None and int(a["d_bottleneck"]) != int(d_bottleneck):
        raise ValueError(
            f"Tokenizer/phase2 mismatch: tokenizer d_bottleneck={d_bottleneck}, "
            f"but phase2 ckpt expects d_bottleneck={int(a['d_bottleneck'])}."
        )
    if a.get("packing_factor") is not None and int(a["packing_factor"]) != int(packing_factor):
        raise ValueError(
            f"Runtime/phase2 mismatch: packing_factor={packing_factor}, "
            f"but phase2 ckpt was trained with packing_factor={int(a['packing_factor'])}."
        )
    if a.get("n_spatial") is not None and int(a["n_spatial"]) != int(n_spatial):
        raise ValueError(
            f"Derived/ckpt mismatch: n_spatial={n_spatial}, "
            f"but phase2 ckpt metadata has n_spatial={int(a['n_spatial'])}."
        )
    if a.get("d_spatial") is not None and int(a["d_spatial"]) != int(d_spatial):
        raise ValueError(
            f"Derived/ckpt mismatch: d_spatial={d_spatial}, "
            f"but phase2 ckpt metadata has d_spatial={int(a['d_spatial'])}."
        )

    # Dynamics
    dyn = Dynamics(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=n_register,
        n_agent=n_agent,
        n_heads=n_heads,
        depth=depth,
        k_max=k_max,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        space_mode=space_mode,
    ).to(device)
    sd = ckpt["dynamics"]
    for pfx in ("module.", "dynamics.", "dyn."):
        if any(k.startswith(pfx) for k in sd):
            sd = {k[len(pfx):] if k.startswith(pfx) else k: v for k, v in sd.items()}
    dyn.load_state_dict(sd, strict=True)

    # TaskEmbedder
    task_embedder = TaskEmbedder(
        d_model=d_model,
        n_agent=n_agent,
        use_ids=True,
        n_tasks=128,
    ).to(device)
    task_embedder.load_state_dict(ckpt["task_embedder"], strict=True)

    # Policy (behavioral prior — will be frozen)
    policy = PolicyHead(
        d_model=d_model,
        action_dim=action_dim,
        hidden=head_hidden,
        mtp_length=mtp_length,
    ).to(device)
    policy.load_state_dict(ckpt["policy"], strict=True)

    # Reward
    reward_head = RewardHead(
        d_model=d_model,
        hidden=head_hidden,
        mtp_length=mtp_length,
    ).to(device)
    reward_head.load_state_dict(ckpt["reward"], strict=True)

    info = {
        "d_model": d_model,
        "k_max": k_max,
        "n_spatial": n_spatial,
        "d_spatial": d_spatial,
        "n_agent": n_agent,
        "n_heads": n_heads,
        "dyn_depth": depth,
        "dropout": dropout,
        "mlp_ratio": mlp_ratio,
        "time_every": time_every,
        "n_register": n_register,
        "space_mode": space_mode,
        "action_dim": action_dim,
        "mtp_length": mtp_length,
        "head_hidden": head_hidden,
        "ckpt_args": a,
    }
    return dyn, task_embedder, policy, reward_head, info


# ---------------------------------------------------------------------------
# Imagined rollout in latent space
# ---------------------------------------------------------------------------

def imagine_rollout(
    dyn: Dynamics,
    policy: PolicyHead,
    reward_head: RewardHead,
    value_head: ValueHead,
    *,
    z_context: torch.Tensor,         # (B, C_t, n_spatial, d_spatial) context latents
    actions_context: torch.Tensor,   # (B, C_t, 16) context actions
    act_mask_context: torch.Tensor,  # (B, C_t, 16) context action mask (shifted — for dynamics)
    task_act_mask: torch.Tensor,     # (B, T, 16) unshifted task mask — for policy and new-step mask
    agent_tokens_ctx: torch.Tensor,  # (B, C_t, n_agent, d_model) context agent tokens
    horizon: int,
    k_max: int,
    sched: dict,
    action_dim: int,
    use_amp: bool = True,
) -> Dict[str, torch.Tensor]:
    """Run imagined rollout for `horizon` steps from encoded context.

    Returns dict with:
        h_states: (B, horizon, d_model) — agent states (detached for policy)
        actions:  (B, horizon, action_dim) — sampled actions (differentiable)
        log_probs: (B, horizon) — log probs of sampled actions
        rewards:  (B, horizon) — predicted rewards
        values:   (B, horizon+1) — predicted values (includes bootstrap)
    """
    device = z_context.device
    dtype = z_context.dtype
    B, C_t = z_context.shape[:2]
    n_spatial = z_context.shape[2]
    d_spatial = z_context.shape[3]

    # Per-batch action validity mask — constant across time for each element.
    # task_act_mask is the unshifted mask; act_mask_context is shifted (pos 0 = zeros).
    # Using the unshifted mask avoids the all-zeros pi_mask bug.
    pi_mask = task_act_mask[:, 0, :action_dim].unsqueeze(1)  # (B, 1, action_dim)

    K = int(sched["K"])
    e = int(sched["e"])
    tau_sched = sched["tau"]
    tau_idx_sched = sched["tau_idx"]
    dt = float(sched["dt"])
    emax = int(round(math.log2(k_max)))

    h_states = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    values_list = []

    # Working latent/action history (grows by 1 each step, trimmed to max_ctx)
    z_hist = z_context  # (B, C_t, n_spatial, d_spatial)
    a_hist = actions_context  # (B, C_t, 16)
    am_hist = act_mask_context  # (B, C_t, 16)
    ag_hist = agent_tokens_ctx  # (B, C_t, n_agent, d_model)
    max_ctx = C_t + horizon

    # --- One-time clean forward on context to get h_0 ---
    # h_s is then carried forward: h_{s+1} from the end of each step becomes h_s
    # for the next step, eliminating the redundant Pass-1 that the old loop had.
    t0 = z_hist.shape[1]
    step_idxs0 = torch.full((B, t0), emax, device=device, dtype=torch.long)
    signal_idxs0 = torch.full((B, t0), k_max - 1, device=device, dtype=torch.long)
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            _, h_ctx = dyn(
                a_hist, step_idxs0, signal_idxs0, z_hist,
                act_mask=am_hist,
                agent_tokens=ag_hist,
            )
    h_prev = h_ctx.mean(dim=2)[:, -1]  # (B, d_model) = h_0

    for step in range(horizon):
        t = z_hist.shape[1]

        # --- Value + policy at current state h_s (= h_prev) ---
        with autocast(device_type="cuda", enabled=use_amp):
            v = value_head.predict(h_prev.unsqueeze(1))[:, 0]  # (B,)
        values_list.append(v)

        # stop_gradient before policy: dynamics state is a fixed feature
        h_detached = h_prev.detach()
        with autocast(device_type="cuda", enabled=use_amp):
            dist = policy.dist(h_detached.unsqueeze(1), step=0, mask=pi_mask)
            action_raw, lp_raw = dist.rsample_with_log_prob()  # (B, 1, A), (B, 1)
            action = action_raw[:, 0].clamp(-1, 1)  # (B, action_dim)
            lp = lp_raw[:, 0]  # (B,)

        h_states.append(h_detached)
        actions_list.append(action)
        log_probs_list.append(lp)

        # --- Extend action/mask/agent histories with a_s ---
        full_action = torch.zeros(B, 16, device=device, dtype=torch.float32)
        full_action[:, :action_dim] = action.detach()
        full_act_mask = task_act_mask[:, 0:1, :]  # (B, 1, 16) — unshifted task mask

        a_ext = torch.cat([a_hist, full_action.unsqueeze(1)], dim=1)   # (B, t+1, 16)
        am_ext = torch.cat([am_hist, full_act_mask], dim=1)             # (B, t+1, 16)
        ag_ext = torch.cat([ag_hist, ag_hist[:, -1:]], dim=1)           # (B, t+1, n_agent, d_model)

        # --- Denoise z_{s+1} (K steps, no grad) ---
        z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
        step_idxs_den = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
        step_idxs_den[:, -1] = e
        signal_idxs_den = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

        with torch.no_grad():
            for i in range(K):
                sig_i = int(tau_idx_sched[i])
                signal_idxs_den[:, -1] = sig_i
                packed_seq = torch.cat([z_hist, z], dim=1)
                with autocast(device_type="cuda", enabled=use_amp):
                    x1_hat_full, _ = dyn(
                        a_ext, step_idxs_den, signal_idxs_den, packed_seq,
                        act_mask=am_ext,
                        agent_tokens=ag_ext,
                    )
                tau_i = float(tau_sched[i])
                x1_hat = x1_hat_full[:, -1:, :, :]
                denom = max(1e-4, 1.0 - tau_i)
                z = (z.float() + (x1_hat.float() - z.float()) / denom * dt).to(dtype)

        # --- Clean forward on history + z_{s+1} → h_{s+1} ---
        # h_{s+1} serves dual purpose:
        #   (a) reward r_{s+1} = R(h_{s+1})
        #   (b) h_prev for the next step — avoids a redundant Pass-1 at the top
        #       of the next iteration (those two calls are identical).
        # After the final step, h_prev = h_H is the bootstrap state, so no
        # extra dynamics forward is needed for the bootstrap value either.
        z_hist_new = torch.cat([z_hist, z.detach()], dim=1)
        step_idxs_clean = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
        signal_idxs_clean = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=use_amp):
                _, h_new = dyn(
                    a_ext, step_idxs_clean, signal_idxs_clean, z_hist_new,
                    act_mask=am_ext,
                    agent_tokens=ag_ext,
                )
        h_prev = h_new.mean(dim=2)[:, -1]  # (B, d_model) = h_{s+1}

        with autocast(device_type="cuda", enabled=use_amp):
            r = reward_head.predict(h_prev.unsqueeze(1), step=0)[:, 0]  # (B,)
        rewards_list.append(r.detach())

        # --- Trim and carry forward histories ---
        z_hist = z_hist_new[:, -max_ctx:]
        a_hist = a_ext[:, -max_ctx:]
        am_hist = am_ext[:, -max_ctx:]
        ag_hist = ag_ext[:, -max_ctx:]

    # Bootstrap value: h_prev is already h_H from the last clean forward — no extra pass.
    with autocast(device_type="cuda", enabled=use_amp):
        v_boot = value_head.predict(h_prev.unsqueeze(1))[:, 0]
    values_list.append(v_boot)

    return {
        "h_states": torch.stack(h_states, dim=1),    # (B, H, d_model)
        "actions": torch.stack(actions_list, dim=1),  # (B, H, action_dim)
        "log_probs": torch.stack(log_probs_list, dim=1),  # (B, H)
        "rewards": torch.stack(rewards_list, dim=1),  # (B, H)
        "values": torch.stack(values_list, dim=1),    # (B, H+1)
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed + rank)

    # --- Dataset (for context frames) ---
    dataset = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.context_len,
        img_size=128,
        action_dim=16,
        tasks_json=args.tasks_json,
        tasks=args.tasks if args.tasks else TASK_SET,
        verbose=is_rank0(),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_batch,
    )

    # --- Frozen tokenizer ---
    tok_override = {}
    if args.H is not None: tok_override["H"] = args.H
    if args.W is not None: tok_override["W"] = args.W
    if args.C is not None: tok_override["C"] = args.C
    if args.patch is not None: tok_override["patch"] = args.patch
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device, override=(tok_override or None),
    )
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    n_spatial = n_latents // args.packing_factor

    # --- Load Phase 2: dynamics (frozen), task_embedder (frozen), policy (prior), reward (frozen) ---
    p2_override = {}
    p2_override["d_model_dyn"] = args.d_model_dyn
    p2_override["dyn_depth"] = args.dyn_depth
    p2_override["n_heads"] = args.n_heads
    p2_override["dropout"] = args.dropout
    p2_override["mlp_ratio"] = args.mlp_ratio
    p2_override["time_every"] = args.time_every
    p2_override["k_max"] = args.k_max
    p2_override["n_register"] = args.n_register
    p2_override["n_agent"] = args.n_agent
    p2_override["action_dim"] = args.action_dim
    p2_override["mtp_length"] = args.mtp_length
    p2_override["head_mlp_ratio"] = args.head_mlp_ratio
    if args.space_mode is not None: p2_override["space_mode"] = args.space_mode

    dyn, task_embedder, policy_prior, reward_head, p2_info = load_finetuned_dynamics(
        args.phase2_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        packing_factor=args.packing_factor,
        override=p2_override,
    )
    d_model = p2_info["d_model"]
    k_max = p2_info["k_max"]
    action_dim = p2_info["action_dim"]
    mtp_length = p2_info["mtp_length"]
    head_hidden = p2_info["head_hidden"]

    # --- Robustness checks: args / phase2 alignment ---
    runtime_vs_loaded = (
        ("d_model_dyn", args.d_model_dyn, p2_info["d_model"]),
        ("k_max",       args.k_max,       p2_info["k_max"]),
        ("n_agent",     args.n_agent,     p2_info["n_agent"]),
        ("n_heads",     args.n_heads,     p2_info["n_heads"]),
        ("dyn_depth",   args.dyn_depth,   p2_info["dyn_depth"]),
        ("n_register",  args.n_register,  p2_info["n_register"]),
        ("action_dim",  args.action_dim,  p2_info["action_dim"]),
        ("mtp_length",  args.mtp_length,  p2_info["mtp_length"]),
    )
    for name, expected, loaded in runtime_vs_loaded:
        if int(expected) != int(loaded):
            raise ValueError(
                f"Args/phase2 mismatch for {name}: args={int(expected)}, loaded={int(loaded)}."
            )
    if str(args.space_mode) != str(p2_info["space_mode"]):
        raise ValueError(
            f"Args/phase2 mismatch for space_mode: args={args.space_mode}, loaded={p2_info['space_mode']}."
        )

    # Freeze dynamics, task_embedder, reward, policy_prior
    for m in [dyn, task_embedder, reward_head, policy_prior]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # --- Trainable heads ---
    # New policy (initialized from Phase 2 prior)
    policy = PolicyHead(
        d_model=d_model,
        action_dim=action_dim,
        hidden=head_hidden,
        mtp_length=mtp_length,
    ).to(device)
    policy.load_state_dict(policy_prior.state_dict())

    # Value head (new, trained from scratch)
    value_head = ValueHead(
        d_model=d_model,
        hidden=head_hidden,
    ).to(device)

    if args.compile:
        dyn = torch.compile(dyn)

    if is_rank0():
        pi_params = sum(p.numel() for p in policy.parameters())
        val_params = sum(p.numel() for p in value_head.parameters())
        frozen_params = sum(p.numel() for p in dyn.parameters()) + \
                        sum(p.numel() for p in task_embedder.parameters()) + \
                        sum(p.numel() for p in reward_head.parameters())
        print(f"Trainable: policy={pi_params:,} value={val_params:,}")
        print(f"Frozen: {frozen_params:,}")

    # --- Denoising schedule for imagination ---
    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=args.imagination_d)
    if is_rank0():
        print(f"Imagination schedule: K={sched['K']}, d={sched['d']}, e={sched['e']}")

    # --- Optimizer ---
    opt = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": args.lr_policy},
            {"params": value_head.parameters(), "lr": args.lr_value},
        ],
        lr=args.lr_policy,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # --- Wandb ---
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online" if args.wandb_project else "disabled",
            config=vars(args),
        )

    # --- Resume ---
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch, resume_args = load_ckpt(
            Path(args.resume),
            policy=policy, value=value_head,
            opt=opt, scaler=scaler,
        )
        # Robustness checks: resumed ckpt / current run alignment
        # (resume_args contains vars(args) at save time — CLI args only)
        for name, current_val in (
            ("horizon", args.horizon),
            ("packing_factor", args.packing_factor),
        ):
            saved_val = resume_args.get(name)
            if saved_val is not None and int(saved_val) != int(current_val):
                raise ValueError(
                    f"Resume/current mismatch for {name}: current={current_val}, "
                    f"resumed ckpt={int(saved_val)}."
                )
        saved_phase2 = resume_args.get("phase2_ckpt")
        if saved_phase2 is not None and saved_phase2 != args.phase2_ckpt:
            print(
                f"[warn] Resume phase2_ckpt mismatch: current={args.phase2_ckpt}, "
                f"resumed ckpt={saved_phase2}."
            )
        if is_rank0():
            print(f"Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # --- Training loop ---
    policy.train()
    value_head.train()

    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.max_steps:
                    break

                # --- Unpack batch ---
                obs_u8 = batch["obs"].to(device, non_blocking=True)      # (B, T+1, 3, H, W) uint8
                act = batch["act"].to(device, non_blocking=True)         # (B, T, 16) float
                mask = batch["act_mask"].to(device, non_blocking=True)   # (B, T, 16) float
                emb_id = batch["emb_id"].to(device, non_blocking=True)   # (B,) long

                act = act.clamp(-1, 1) * mask
                frames = obs_u8[:, :-1].float() / 255.0  # (B, T, 3, H, W)
                actions = torch.zeros_like(act)
                actions[:, 1:] = act[:, :-1]
                act_mask_shifted = torch.zeros_like(mask)
                act_mask_shifted[:, 1:] = mask[:, :-1]

                B, T = frames.shape[:2]

                # --- Frozen encode ---
                with torch.no_grad():
                    patches = temporal_patchify(frames, patch)
                    z_btLd, _ = encoder(patches)
                    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)

                # Agent tokens from frozen task_embedder
                with torch.no_grad():
                    agent_tokens = task_embedder(emb_id, B=B, T=T)

                # --- Imagined rollout ---
                rollout = imagine_rollout(
                    dyn, policy, reward_head, value_head,
                    z_context=z1.detach(),
                    actions_context=actions,
                    act_mask_context=act_mask_shifted,
                    task_act_mask=mask,
                    agent_tokens_ctx=agent_tokens,
                    horizon=args.horizon,
                    k_max=k_max,
                    sched=sched,
                    action_dim=action_dim,
                    use_amp=use_amp,
                )

                # --- Compute TD(λ) returns ---
                with torch.no_grad():
                    lambda_returns = compute_lambda_returns(
                        rollout["rewards"],
                        rollout["values"],
                        gamma=args.gamma,
                        lam=args.lam,
                    )  # (B, horizon)

                # --- Shared setup for all policy loss variants ---
                advantages = (lambda_returns - rollout["values"][:, :-1]).detach()
                if args.time_normalize_adv:
                    # Center per-timestep across the batch: removes structural bias
                    # where early timesteps are almost always positive (value hasn't
                    # caught up to accumulated future reward) and late ones negative.
                    advantages = advantages - advantages.mean(dim=0, keepdim=True)
                act_mask_1d = mask[:, 0, :action_dim].unsqueeze(1)  # (B,1,A) — unshifted task mask
                H_roll = rollout["log_probs"].shape[1]
                # Per-sample valid dim counts (multi-task: different tasks may have
                # different action space sizes; normalize by each sample's own count)
                valid_dims = act_mask_1d.sum(dim=-1).expand(-1, H_roll).reshape(-1)  # (B*H,)
                policy_active = step >= args.policy_start

                def _grad_step(loss_val):
                    """One scaler backward + clip + optimizer step."""
                    scaler.scale(loss_val).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)
                        torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_norm=args.grad_clip)
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                ppo_handled = False

                if args.policy_loss == "pmpo":
                    with autocast(device_type="cuda", enabled=use_amp):
                        value_loss = value_head.loss(rollout["h_states"], lambda_returns.detach())
                        with torch.no_grad():
                            prior_dist = policy_prior.dist(rollout["h_states"], step=0,
                                                           mask=act_mask_1d)
                            lp_prior = prior_dist.log_prob(rollout["actions"].detach())  # (B, H)
                        policy_loss = pmpo_loss(
                            rollout["log_probs"].reshape(-1),
                            advantages.reshape(-1),
                            log_probs_prior=lp_prior.reshape(-1),
                            alpha=args.alpha,
                            beta=args.beta,
                            action_dim=valid_dims,
                            entropy_coef=args.entropy_coef,
                        )
                        total_loss = (args.w_policy * policy_loss if policy_active else 0.0) + args.w_value * value_loss

                elif args.policy_loss == "reinforce":
                    with autocast(device_type="cuda", enabled=use_amp):
                        value_loss = value_head.loss(rollout["h_states"], lambda_returns.detach())
                        policy_loss = reinforce_loss(
                            rollout["log_probs"].reshape(-1),
                            advantages.reshape(-1),
                            entropy_coef=args.entropy_coef,
                            action_dim=valid_dims,
                        )
                        total_loss = (args.w_policy * policy_loss if policy_active else 0.0) + args.w_value * value_loss

                elif args.policy_loss == "ppo":
                    # PPO runs K full gradient steps on the same rollout data.
                    # grad_accum is bypassed — each epoch is a complete update.
                    lp_old = rollout["log_probs"].detach()  # (B, H) — fixed across epochs
                    policy_loss = torch.tensor(0.0, device=device)
                    value_loss = torch.tensor(0.0, device=device)
                    total_loss = torch.tensor(0.0, device=device)
                    opt.zero_grad(set_to_none=True)
                    for _ in range(args.ppo_epochs):
                        with autocast(device_type="cuda", enabled=use_amp):
                            # Recompute log probs from current (updated) policy
                            lp_new = policy.dist(
                                rollout["h_states"], step=0, mask=act_mask_1d
                            ).log_prob(rollout["actions"].detach())  # (B, H)
                            policy_loss = ppo_loss(
                                lp_new.reshape(-1),
                                lp_old.reshape(-1),
                                advantages.reshape(-1),
                                eps_clip=args.eps_clip,
                                entropy_coef=args.entropy_coef,
                                action_dim=valid_dims,
                            )
                            value_loss = value_head.loss(rollout["h_states"], lambda_returns.detach())
                            total_loss = (args.w_policy * policy_loss if policy_active else 0.0) + args.w_value * value_loss
                        if not torch.isfinite(total_loss):
                            if is_rank0():
                                print(f"[warn] PPO epoch non-finite loss at step {step}, stopping epochs")
                            opt.zero_grad(set_to_none=True)
                            break
                        _grad_step(total_loss)
                    ppo_handled = True

                if not ppo_handled:
                    if not torch.isfinite(total_loss):
                        if is_rank0():
                            print(f"[warn] Non-finite loss at step {step}, skipping")
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        continue

                    loss_scaled = total_loss / grad_accum
                    scaler.scale(loss_scaled).backward()

                    do_step = ((step + 1) % grad_accum == 0)
                    if do_step:
                        if args.grad_clip > 0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)
                            torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_norm=args.grad_clip)
                        if use_amp:
                            scaler.step(opt)
                            scaler.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

                # --- Logging ---
                if is_rank0() and (step % args.log_every == 0):
                    elapsed = (time.time() - t0) / 3600.0
                    adv_pos_frac = (advantages > 0).float().mean().item()
                    mean_return = lambda_returns.mean().item()
                    mean_reward = rollout["rewards"].mean().item()
                    mean_value = rollout["values"][:, :-1].mean().item()

                    # Diagnostic stats
                    lp_vals = rollout["log_probs"].detach()
                    raw_adv = (lambda_returns - rollout["values"][:, :-1]).detach()
                    centered_adv = raw_adv - raw_adv.mean(dim=0, keepdim=True)
                    adv_pos_frac = (centered_adv > 0).float().mean().item()
                    mu_diag, std_diag = policy(rollout["h_states"][:, :1])
                    pi_std_mean = std_diag.mean().item()

                    # Per-timestep advantage breakdown to check temporal bias
                    adv_by_t = raw_adv  # (B, H)
                    adv_pos_by_t = (adv_by_t > 0).float().mean(dim=0)  # (H,)
                    adv_mean_by_t = adv_by_t.mean(dim=0)               # (H,)
                    t_pos_str  = " ".join(f"{adv_pos_by_t[t].item():.2f}" for t in range(adv_by_t.shape[1]))
                    t_mean_str = " ".join(f"{adv_mean_by_t[t].item():+.2f}" for t in range(adv_by_t.shape[1]))

                    print(
                        f"step {step:07d} | "
                        f"total={total_loss.item():.4f} "
                        f"pi={policy_loss.item():.4f} "
                        f"val={value_loss.item():.4f} "
                        f"| adv+={adv_pos_frac:.2f} "
                        f"R={mean_return:.3f} "
                        f"r={mean_reward:.3f} "
                        f"V={mean_value:.3f} "
                        f"S={ret_scale:.3f} "
                        f"| lp={lp_vals.mean().item():.1f}[{lp_vals.min().item():.1f},{lp_vals.max().item():.1f}] "
                        f"adv_raw={raw_adv.mean().item():.3f}±{raw_adv.std().item():.3f} "
                        f"std={pi_std_mean:.4f} "
                        f"| {elapsed:.2f}h"
                    )
                    print(f"         adv+ by t: [{t_pos_str}]")
                    print(f"         adv  by t: [{t_mean_str}]")
                    wandb.log({
                        "loss/total": total_loss.item(),
                        "loss/policy": policy_loss.item(),
                        "loss/value": value_loss.item(),
                        "stats/adv_pos_frac": adv_pos_frac,
                        "stats/mean_return": mean_return,
                        "stats/mean_reward": mean_reward,
                        "stats/mean_value": mean_value,
                        "time/hrs": elapsed,
                    }, step=step)

                # --- Checkpointing ---
                if is_rank0() and args.save_every > 0 and step > 0 and (step % args.save_every == 0) and do_step:
                    save_ckpt(
                        ckpt_dir / f"step_{step:07d}.pt",
                        step=step, epoch=epoch,
                        policy=policy, value=value_head,
                        opt=opt, scaler=scaler, args=args,
                    )
                    save_ckpt(
                        ckpt_dir / "latest.pt",
                        step=step, epoch=epoch,
                        policy=policy, value=value_head,
                        opt=opt, scaler=scaler, args=args,
                    )

                step += 1

            start_epoch = epoch + 1

    # Final checkpoint
    if is_rank0():
        save_ckpt(
            ckpt_dir / "final.pt",
            step=step, epoch=start_epoch,
            policy=policy, value=value_head,
            opt=opt, scaler=scaler, args=args,
        )

    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3: Imagination RL (PMPO + TD(λ))")

    # Data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[   # paths to raw data
        "/public/dreamer4/expert",
        "/public/dreamer4/mixed-small",
        "/public/dreamer4/mixed-large",
    ])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=[  # paths to preprocessed frames
        "/public/dreamer4/expert-shards",
        "/public/dreamer4/mixed-small-shards",
        "/public/dreamer4/mixed-large-shards",
    ])
    p.add_argument("--tasks_json", type=str, default="./tasks.json")
    p.add_argument("--context_len", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Task subset. Default: all 30 TASK_SET tasks.")

    # Checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    p.add_argument("--phase2_ckpt", type=str, required=True,
                   help="Path to Phase 2 agent checkpoint (dynamics + task_emb + policy + reward)")
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--space_mode", type=str, default="wm_agent",
                   help="Attention mode for agent tokens. 'wm_agent' = full attention.")

    # dynamics arch (must match the phase2 ckpt)
    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)

    # head arch (must match the phase2 ckpt)
    p.add_argument("--action_dim", type=int, default=16)
    p.add_argument("--mtp_length", type=int, default=8)
    p.add_argument("--head_mlp_ratio", type=float, default=2.0)

    # shortcut schedule
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--policy_start", type=int, default=0,
                   help="Step at which policy loss begins contributing. Value-only warmup before this.")

    # Imagination
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--imagination_d", type=float, default=0.25,
                   help="Shortcut d for denoising in imagination (K = 1/d steps)")
    p.add_argument("--gamma", type=float, default=0.997) # core hparam
    p.add_argument("--lam", type=float, default=0.95) # core hparam

    # RL algorithm — choices: pmpo | ppo | reinforce
    p.add_argument("--policy_loss", type=str, default="pmpo",
                   choices=["pmpo", "ppo", "reinforce"],
                   help="Policy gradient algorithm")
    # shared
    p.add_argument("--entropy_coef", type=float, default=0.0,
                   help="Entropy regularization coefficient (all algos)")
    p.add_argument("--time_normalize_adv", action="store_true",
                   help="Center advantages per-timestep across batch to remove "
                        "structural early-positive/late-negative temporal bias")
    # PMPO-specific
    p.add_argument("--alpha", type=float, default=0.5,
                   help="[pmpo] sign-split weight (0=rejection only, 1=acceptance only)")
    p.add_argument("--beta", type=float, default=0.3,
                   help="[pmpo] KL regularization weight against behavioral prior")
    # PPO-specific
    p.add_argument("--eps_clip", type=float, default=0.2,
                   help="[ppo] clipping radius ε for surrogate objective")
    p.add_argument("--ppo_epochs", type=int, default=4,
                   help="[ppo] number of gradient epochs per rollout")

    # Loss weights
    p.add_argument("--w_policy", type=float, default=1.0)
    p.add_argument("--w_value", type=float, default=1.0)

    # Optimization
    p.add_argument("--lr_policy", type=float, default=3e-5)
    p.add_argument("--lr_value", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # logging
    p.add_argument("--log_every", type=int, default=100)

    # wandb
    p.add_argument("--wandb_project", type=str, default="Dreamer 4 Continuous Control")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default="eqforcing")

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="./logs/imagination_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
