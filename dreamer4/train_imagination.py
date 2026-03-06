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
from agent import PolicyHead, RewardHead, ValueHead, compute_lambda_returns, pmpo_loss
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
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def load_phase2(
    ckpt_path: str,
    *,
    device: torch.device,
    d_bottleneck: int,
    n_latents: int,
    packing_factor: int,
) -> Tuple[Dynamics, TaskEmbedder, PolicyHead, RewardHead, dict]:
    """Load Phase 2 checkpoint: dynamics + task_embedder + policy + reward."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    d_model = int(a.get("d_model_dyn", a.get("d_model", 512)))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("dyn_depth", a.get("depth", 8)))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 1))
    k_max = int(a.get("k_max", 8))
    n_register = int(a.get("n_register", 4))
    n_agent = int(a.get("n_agent", 1))
    space_mode = str(a.get("space_mode", "wm_agent_isolated"))
    action_dim = int(a.get("action_dim", 16))
    mtp_length = int(a.get("mtp_length", 8))
    head_hidden = int(d_model * float(a.get("head_mlp_ratio", 2.0)))

    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

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
    dyn.load_state_dict(ckpt["dynamics"], strict=True)

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
        "action_dim": action_dim,
        "mtp_length": mtp_length,
        "head_hidden": head_hidden,
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
    act_mask_context: torch.Tensor,  # (B, C_t, 16) context action mask
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

    K = int(sched["K"])
    e = int(sched["e"])
    tau_sched = sched["tau"]
    tau_idx_sched = sched["tau_idx"]
    dt = float(sched["dt"])
    emax = int(round(math.log2(k_max)))

    # Collect outputs
    h_states = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    values_list = []

    # Working state: growing latent history
    z_hist = z_context  # (B, t, n_spatial, d_spatial), starts as context
    a_hist = actions_context  # (B, t, 16)
    am_hist = act_mask_context  # (B, t, 16)
    ag_hist = agent_tokens_ctx  # (B, t, n_agent, d_model)

    for step in range(horizon):
        t = z_hist.shape[1]

        # --- Get h_t from clean forward pass on current history ---
        step_idxs = torch.full((B, t), emax, device=device, dtype=torch.long)
        signal_idxs = torch.full((B, t), k_max - 1, device=device, dtype=torch.long)

        with autocast(device_type="cuda", enabled=use_amp):
            _, h_t_full = dyn(
                a_hist, step_idxs, signal_idxs, z_hist,
                act_mask=am_hist,
                agent_tokens=ag_hist,
            )
        # h_t_full: (B, t, n_agent, d_model) — take last timestep
        h_last = h_t_full[:, -1, 0, :]  # (B, d_model)

        # Value prediction at this state
        with autocast(device_type="cuda", enabled=use_amp):
            v = value_head.predict(h_last.unsqueeze(1))[:, 0]  # (B,)
        values_list.append(v)

        # --- Phase 3: stop_gradient on h_t before policy ---
        h_detached = h_last.detach()

        # Sample action from policy (differentiable through policy params)
        with autocast(device_type="cuda", enabled=use_amp):
            dist = policy.dist(h_detached.unsqueeze(1), step=0)  # over (B, 1)
            action_raw = dist.rsample()  # (B, 1, action_dim)
            action = action_raw[:, 0].clamp(-1, 1)  # (B, action_dim)
            lp = dist.log_prob(action_raw)[:, 0]  # (B,)

        h_states.append(h_detached)
        actions_list.append(action)
        log_probs_list.append(lp)

        # --- Denoise next latent z_{t+1} ---
        # Build full action for dynamics (pad to 16 dims)
        full_action = torch.zeros(B, 16, device=device, dtype=torch.float32)
        full_action[:, :action_dim] = action.detach()  # detach action for dynamics input
        full_act_mask = act_mask_context[:, 0:1, :]  # (B, 1, 16) — same mask

        # Extend histories for denoising
        a_ext = torch.cat([a_hist, full_action.unsqueeze(1)], dim=1)  # (B, t+1, 16)
        am_ext = torch.cat([am_hist, full_act_mask], dim=1)  # (B, t+1, 16)
        # Agent tokens: repeat last agent token for new step
        ag_ext = torch.cat([ag_hist, ag_hist[:, -1:]], dim=1)  # (B, t+1, n_agent, d_model)

        z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

        step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
        step_idxs_full[:, -1] = e
        signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

        with torch.no_grad():
            for i in range(K):
                tau_i = float(tau_sched[i])
                sig_i = int(tau_idx_sched[i])
                signal_idxs_full[:, -1] = sig_i

                packed_seq = torch.cat([z_hist, z], dim=1)  # (B, t+1, n_spatial, d_spatial)

                with autocast(device_type="cuda", enabled=use_amp):
                    x1_hat_full, _ = dyn(
                        a_ext, step_idxs_full, signal_idxs_full, packed_seq,
                        act_mask=am_ext,
                        agent_tokens=ag_ext,
                    )

                x1_hat = x1_hat_full[:, -1:, :, :]
                denom = max(1e-4, 1.0 - tau_i)
                b = (x1_hat.float() - z.float()) / denom
                z = (z.float() + b * dt).to(dtype)

        # Reward prediction for the new state
        # Run one more clean forward to get h_{t+1} for reward
        z_new = z.detach()
        z_hist_new = torch.cat([z_hist, z_new], dim=1)

        step_idxs_new = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
        signal_idxs_new = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=use_amp):
                _, h_new = dyn(
                    a_ext, step_idxs_new, signal_idxs_new, z_hist_new,
                    act_mask=am_ext,
                    agent_tokens=ag_ext,
                )
        h_next = h_new[:, -1, 0, :]  # (B, d_model)
        with autocast(device_type="cuda", enabled=use_amp):
            r = reward_head.predict(h_next.unsqueeze(1), step=0)[:, 0]  # (B,)
        rewards_list.append(r.detach())

        # --- Update history for next step ---
        # Limit context window to avoid unbounded growth
        max_ctx = z_context.shape[1] + horizon
        z_hist = z_hist_new[:, -max_ctx:]
        a_hist = a_ext[:, -max_ctx:]
        am_hist = am_ext[:, -max_ctx:]
        ag_hist = ag_ext[:, -max_ctx:]

    # Bootstrap value at final state
    t_final = z_hist.shape[1]
    step_idxs_final = torch.full((B, t_final), emax, device=device, dtype=torch.long)
    signal_idxs_final = torch.full((B, t_final), k_max - 1, device=device, dtype=torch.long)

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            _, h_final = dyn(
                a_hist, step_idxs_final, signal_idxs_final, z_hist,
                act_mask=am_hist,
                agent_tokens=ag_hist,
            )
    h_boot = h_final[:, -1, 0, :]
    with autocast(device_type="cuda", enabled=use_amp):
        v_boot = value_head.predict(h_boot.unsqueeze(1))[:, 0]
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
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device,
    )
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    n_spatial = n_latents // args.packing_factor

    # --- Load Phase 2: dynamics (frozen), task_embedder (frozen), policy (prior), reward (frozen) ---
    dyn, task_embedder, policy_prior, reward_head, p2_info = load_phase2(
        args.phase2_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        packing_factor=args.packing_factor,
    )
    d_model = p2_info["d_model"]
    k_max = p2_info["k_max"]
    action_dim = p2_info["action_dim"]
    mtp_length = p2_info["mtp_length"]
    head_hidden = p2_info["head_hidden"]

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
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                mode="online" if args.wandb_project else "disabled",
                config=vars(args),
            )
            use_wandb = True
        except Exception:
            use_wandb = False
            print("[warn] wandb init failed, logging to console only")
    else:
        use_wandb = False

    # --- Resume ---
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(
            Path(args.resume),
            policy=policy, value=value_head,
            opt=opt, scaler=scaler,
        )
        if is_rank0():
            print(f"Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # --- Training loop ---
    policy.train()
    value_head.train()

    t0 = time.time()

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

                # --- Value loss: two-hot CE against lambda returns ---
                with autocast(device_type="cuda", enabled=use_amp):
                    # Recompute values with gradient for the value head
                    h_for_value = rollout["h_states"]  # (B, H, d_model) — detached from dynamics
                    value_loss = value_head.loss(h_for_value, lambda_returns.detach())

                # --- Policy loss: PMPO ---
                with autocast(device_type="cuda", enabled=use_amp):
                    advantages = (lambda_returns - rollout["values"][:, :-1]).detach()

                    # Log probs under behavioral prior (frozen)
                    with torch.no_grad():
                        prior_dist = policy_prior.dist(rollout["h_states"], step=0)
                        lp_prior = prior_dist.log_prob(rollout["actions"].detach())  # (B, H)

                    policy_loss = pmpo_loss(
                        rollout["log_probs"].reshape(-1),
                        advantages.reshape(-1),
                        log_probs_prior=lp_prior.reshape(-1),
                        alpha=args.alpha,
                        beta=args.beta,
                    )

                    total_loss = args.w_policy * policy_loss + args.w_value * value_loss

                if not torch.isfinite(total_loss):
                    if is_rank0():
                        print(f"[warn] Non-finite loss at step {step}, skipping")
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    continue

                scaler.scale(total_loss).backward()

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

                    print(
                        f"step {step:07d} | "
                        f"total={total_loss.item():.4f} "
                        f"pi={policy_loss.item():.4f} "
                        f"val={value_loss.item():.4f} "
                        f"| adv+={adv_pos_frac:.2f} "
                        f"R={mean_return:.3f} "
                        f"r={mean_reward:.3f} "
                        f"V={mean_value:.3f} "
                        f"| {elapsed:.2f}h"
                    )
                    if use_wandb:
                        import wandb
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
                if is_rank0() and args.save_every > 0 and step > 0 and (step % args.save_every == 0):
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
    p.add_argument("--data_dirs", type=str, nargs="+", default=["/public/dreamer4/expert"])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=["/public/dreamer4/expert-shards"])
    p.add_argument("--tasks_json", type=str, default="tasks.json")
    p.add_argument("--context_len", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tasks", type=str, nargs="+", default=None)

    # Checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--phase2_ckpt", type=str, required=True,
                   help="Path to Phase 2 agent checkpoint (dynamics + task_emb + policy + reward)")
    p.add_argument("--packing_factor", type=int, default=2)

    # Imagination
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--imagination_d", type=float, default=0.25,
                   help="Shortcut d for denoising in imagination (K = 1/d steps)")
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)

    # PMPO
    p.add_argument("--alpha", type=float, default=0.5, help="PMPO sign-split weight")
    p.add_argument("--beta", type=float, default=0.3, help="PMPO KL regularization")

    # Loss weights
    p.add_argument("--w_policy", type=float, default=1.0)
    p.add_argument("--w_value", type=float, default=1.0)

    # Optimization
    p.add_argument("--lr_policy", type=float, default=3e-5)
    p.add_argument("--lr_value", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default="dreamer4-imagination")
    p.add_argument("--wandb_run_name", type=str, default="phase3")
    p.add_argument("--wandb_entity", type=str, default=None)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="./logs/imagination_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)

    train(p.parse_args())
