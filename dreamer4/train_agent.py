#!/usr/bin/env python3
"""Phase 2: Agent Finetuning — BC + Reward heads + Dynamics finetuning.

Loads pre-trained dynamics checkpoint, adds TaskEmbedder + PolicyHead + RewardHead,
and trains all jointly with:
  1. Shortcut/flow loss (dynamics, same as pretraining)
  2. BC loss (policy predicts actions from agent token outputs)
  3. Reward loss (reward head predicts rewards from agent token outputs)

Usage (single GPU):
    python dreamer4/train_agent.py \
        --tokenizer_ckpt logs/tokenizer.pt \
        --dynamics_ckpt logs/dynamics.pt \
        --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
        --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards

Usage (multi-GPU):
    torchrun --nproc_per_node=4 dreamer4/train_agent.py ...
"""
import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from task_set import TASK_SET
from wm_dataset import WMDataset, collate_batch
from model import (
    Encoder, Decoder, Tokenizer, Dynamics, TaskEmbedder,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from agent import PolicyHead, RewardHead
from train_dynamics import (
    get_dist_info, is_rank0, seed_everything, worker_init_fn,
    init_distributed, load_frozen_tokenizer_from_pt_ckpt,
    dynamics_pretrain_loss,
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
    dynamics,
    task_embedder: TaskEmbedder,
    policy: PolicyHead,
    reward: RewardHead,
    opt,
    scaler,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "dynamics": (dynamics.module.state_dict() if hasattr(dynamics, "module") else dynamics.state_dict()),
        "task_embedder": task_embedder.state_dict(),
        "policy": policy.state_dict(),
        "reward": reward.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "action_dim": args.action_dim,
        "mtp_length": args.mtp_length,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(
    path: Path,
    *,
    dynamics,
    task_embedder: TaskEmbedder,
    policy: PolicyHead,
    reward: RewardHead,
    opt,
    scaler,
) -> tuple:
    ckpt = torch.load(path, map_location="cpu")
    (dynamics.module if hasattr(dynamics, "module") else dynamics).load_state_dict(ckpt["dynamics"], strict=True)
    task_embedder.load_state_dict(ckpt["task_embedder"], strict=True)
    policy.load_state_dict(ckpt["policy"], strict=True)
    reward.load_state_dict(ckpt["reward"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def load_pretrained_dynamics(
    ckpt_path: str,
    *,
    device: torch.device,
    d_bottleneck: int,
    n_latents: int,
    packing_factor: int,
) -> tuple:
    """Load pre-trained dynamics and return model + info dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    d_model = int(a.get("d_model_dyn", 512))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("dyn_depth", 8))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 1))
    k_max = int(a.get("k_max", 8))
    n_register = int(a.get("n_register", 4))
    n_agent = int(a.get("n_agent", 1))
    space_mode = str(a.get("space_mode", "wm_agent_isolated"))

    assert n_latents % packing_factor == 0
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

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

    # Load with strict=False since pre-trained ckpt may lack TaskEmbedder keys
    sd = ckpt["dynamics"]
    # Strip common prefixes
    for pfx in ("module.", "dynamics.", "dyn."):
        if any(k.startswith(pfx) for k in sd):
            sd = {k[len(pfx):] if k.startswith(pfx) else k: v for k, v in sd.items()}
    dyn.load_state_dict(sd, strict=True)

    info = {
        "d_model": d_model,
        "k_max": k_max,
        "n_spatial": n_spatial,
        "d_spatial": d_spatial,
        "n_agent": n_agent,
    }
    return dyn, info


# ---------------------------------------------------------------------------
# BC + Reward loss computation
# ---------------------------------------------------------------------------

def bc_loss(
    policy: PolicyHead,
    h_t: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_dim: int,
    mtp_length: int,
) -> tuple:
    """Behavioral cloning loss: NLL of dataset actions under the policy.

    MTP: for step l, predict action at t+l+1 from h_t at time t.

    Args:
        h_t: (B, T, d_model) — detached agent token outputs
        actions: (B, T, 16) — dataset actions (aligned: action[t] produced obs[t+1])
        act_mask: (B, T, 16) — action mask
        action_dim: effective action dimensions for current batch
        mtp_length: multi-token prediction length

    Returns:
        loss: scalar
        aux: dict
    """
    mean, std = policy(h_t)  # (B, T, L, A), (B, T, L, A)
    B, T, L, A = mean.shape

    total_nll = torch.tensor(0.0, device=h_t.device)
    count = 0
    for l in range(min(L, mtp_length)):
        valid_T = T - l
        if valid_T <= 0:
            break
        # Actions at time t+l for input at time t
        target_actions = actions[:, l:l + valid_T, :A]  # (B, valid_T, A)
        target_mask = act_mask[:, l:l + valid_T, :A]

        d = torch.distributions.Independent(
            torch.distributions.Normal(mean[:, :valid_T, l, :A], std[:, :valid_T, l, :A]), 1
        )
        # Mask out inactive action dims by only computing loss on active ones
        nll = -d.log_prob(target_actions.clamp(-1, 1))  # (B, valid_T)
        total_nll = total_nll + nll.mean()
        count += 1

    loss = total_nll / max(count, 1)
    aux = {
        "bc_nll": loss.detach(),
        "bc_mean_std": std.mean().detach(),
    }
    return loss, aux


def reward_loss(
    reward_head: RewardHead,
    h_t: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple:
    """Reward prediction loss: two-hot CE of dataset rewards.

    Args:
        h_t: (B, T, d_model) — detached agent token outputs
        rewards: (B, T) — dataset rewards

    Returns:
        loss: scalar
        aux: dict
    """
    loss = reward_head.loss(h_t, rewards)
    pred = reward_head.predict(h_t, step=0)
    aux = {
        "reward_ce": loss.detach(),
        "reward_pred_mean": pred.mean().detach(),
        "reward_target_mean": rewards.mean().detach(),
    }
    return loss, aux


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed + rank)

    # --- Dataset ---
    dataset = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.seq_len,
        img_size=128,
        action_dim=16,
        tasks_json=args.tasks_json,
        tasks=TASK_SET,
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
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))

    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # --- Load pre-trained dynamics (trainable) ---
    dyn, dyn_info = load_pretrained_dynamics(
        args.dynamics_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        packing_factor=args.packing_factor,
    )
    d_model = dyn_info["d_model"]
    k_max = dyn_info["k_max"]
    n_agent = dyn_info["n_agent"]

    # --- New trainable heads ---
    task_embedder = TaskEmbedder(
        d_model=d_model,
        n_agent=n_agent,
        use_ids=True,
        n_tasks=128,
    ).to(device)

    policy = PolicyHead(
        d_model=d_model,
        action_dim=args.action_dim,
        mtp_length=args.mtp_length,
        mlp_ratio=args.head_mlp_ratio,
    ).to(device)

    reward_head = RewardHead(
        d_model=d_model,
        mtp_length=args.mtp_length,
        mlp_ratio=args.head_mlp_ratio,
    ).to(device)

    if is_rank0():
        dyn_params = sum(p.numel() for p in dyn.parameters())
        te_params = sum(p.numel() for p in task_embedder.parameters())
        pi_params = sum(p.numel() for p in policy.parameters())
        rw_params = sum(p.numel() for p in reward_head.parameters())
        print(f"Parameters: dynamics={dyn_params:,} task_emb={te_params:,} policy={pi_params:,} reward={rw_params:,}")
        print(f"Total trainable: {dyn_params + te_params + pi_params + rw_params:,}")

    # --- DDP ---
    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
        )

    # --- Optimizer ---
    all_params = [
        {"params": (dyn.module if hasattr(dyn, "module") else dyn).parameters(), "lr": args.lr_dynamics},
        {"params": task_embedder.parameters(), "lr": args.lr},
        {"params": policy.parameters(), "lr": args.lr},
        {"params": reward_head.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
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
            dynamics=dyn, task_embedder=task_embedder,
            policy=policy, reward=reward_head,
            opt=opt, scaler=scaler,
        )
        if is_rank0():
            print(f"Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # --- Training loop ---
    dyn.train()
    policy.train()
    reward_head.train()
    task_embedder.train()

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
                rew = batch["rew"].to(device, non_blocking=True)         # (B, T) float
                emb_id = batch["emb_id"].to(device, non_blocking=True)   # (B,) long

                act = act.clamp(-1, 1) * mask

                # frames: obs[0..T-1], action alignment: action[t] produced obs[t+1]
                frames = obs_u8[:, :-1].float() / 255.0                 # (B, T, 3, H, W)
                # Shift actions: action[0] = zeros (no action produced first frame)
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

                # --- Agent tokens from TaskEmbedder ---
                agent_tokens = task_embedder(emb_id, B=B, T=T)  # (B, T, n_agent, d_model)

                # --- Forward pass ---
                B_self = int(round(args.self_fraction * B))
                B_self = max(0, min(B - 1, B_self))

                with autocast(device_type="cuda", enabled=use_amp):
                    # 1) Dynamics shortcut/flow loss (same as pretraining)
                    dyn_loss, dyn_aux = dynamics_pretrain_loss(
                        dyn.module if hasattr(dyn, "module") else dyn,
                        z1=z1,
                        actions=actions,
                        act_mask=act_mask_shifted,
                        k_max=k_max,
                        B_self=B_self,
                        step=step,
                        bootstrap_start=args.bootstrap_start,
                        agent_tokens=agent_tokens,
                    )

                    # 2) Get h_t from a clean forward pass (tau=1, finest step)
                    # for BC and reward heads
                    dyn_raw = dyn.module if hasattr(dyn, "module") else dyn
                    emax = int(round(math.log2(k_max)))
                    step_idxs = torch.full((B, T), emax, device=device, dtype=torch.long)
                    signal_idxs = torch.full((B, T), k_max - 1, device=device, dtype=torch.long)

                    _, h_t = dyn_raw(
                        actions, step_idxs, signal_idxs, z1,
                        act_mask=act_mask_shifted,
                        agent_tokens=agent_tokens,
                    )
                    # h_t: (B, T, n_agent, d_model) — pool over agent dim
                    h_pooled = h_t.mean(dim=2)  # (B, T, d_model)

                    # Phase 2: do NOT detach h_t — gradients from heads
                    # flow back to task_embedder and dynamics agent-token path.
                    # (In isolated mode, agent tokens only self-attend, so this
                    # mainly updates the task_embedder + transformer MLP on agents.)

                    # 3) BC loss
                    bc_l, bc_aux = bc_loss(
                        policy, h_pooled, act, mask,
                        action_dim=args.action_dim,
                        mtp_length=args.mtp_length,
                    )

                    # 4) Reward loss
                    rw_l, rw_aux = reward_loss(reward_head, h_pooled, rew)

                    # Total loss
                    total_loss = (
                        args.w_dynamics * dyn_loss
                        + args.w_bc * bc_l
                        + args.w_reward * rw_l
                    )

                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: total={total_loss.item()}")

                loss_scaled = total_loss / grad_accum
                scaler.scale(loss_scaled).backward()

                do_step = ((step + 1) % grad_accum == 0)
                if do_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        all_modules = [
                            dyn.module if hasattr(dyn, "module") else dyn,
                            task_embedder, policy, reward_head,
                        ]
                        for m in all_modules:
                            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=args.grad_clip)

                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # --- Logging ---
                if is_rank0() and (step % args.log_every == 0):
                    elapsed = (time.time() - t0) / 3600.0
                    print(
                        f"step {step:07d} | "
                        f"total={total_loss.item():.4f} "
                        f"dyn={dyn_loss.item():.4f} "
                        f"bc={bc_l.item():.4f} "
                        f"rew={rw_l.item():.4f} "
                        f"| flow={dyn_aux['flow_mse'].item():.5f} "
                        f"boot={dyn_aux['bootstrap_mse'].item():.5f} "
                        f"| std={bc_aux['bc_mean_std'].item():.3f} "
                        f"r_pred={rw_aux['reward_pred_mean'].item():.3f} "
                        f"r_tgt={rw_aux['reward_target_mean'].item():.3f} "
                        f"| {elapsed:.2f}h"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "loss/total": total_loss.item(),
                            "loss/dynamics": dyn_loss.item(),
                            "loss/bc": bc_l.item(),
                            "loss/reward": rw_l.item(),
                            "loss/flow_mse": dyn_aux["flow_mse"].item(),
                            "loss/bootstrap_mse": dyn_aux["bootstrap_mse"].item(),
                            "stats/bc_mean_std": bc_aux["bc_mean_std"].item(),
                            "stats/reward_pred_mean": rw_aux["reward_pred_mean"].item(),
                            "stats/reward_target_mean": rw_aux["reward_target_mean"].item(),
                            "time/hrs": elapsed,
                        }, step=step)

                # --- Checkpointing ---
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and do_step:
                    save_ckpt(
                        ckpt_dir / f"step_{step:07d}.pt",
                        step=step, epoch=epoch,
                        dynamics=dyn, task_embedder=task_embedder,
                        policy=policy, reward=reward_head,
                        opt=opt, scaler=scaler, args=args,
                    )
                    save_ckpt(
                        ckpt_dir / "latest.pt",
                        step=step, epoch=epoch,
                        dynamics=dyn, task_embedder=task_embedder,
                        policy=policy, reward=reward_head,
                        opt=opt, scaler=scaler, args=args,
                    )

                step += 1

            start_epoch = epoch + 1

    # Final checkpoint
    if is_rank0():
        save_ckpt(
            ckpt_dir / "final.pt",
            step=step, epoch=start_epoch,
            dynamics=dyn, task_embedder=task_embedder,
            policy=policy, reward=reward_head,
            opt=opt, scaler=scaler, args=args,
        )

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2: Agent Finetuning (BC + Reward + Dynamics)")

    # Data
    p.add_argument("--data_dirs", type=str, nargs="+", default=["/public/dreamer4/expert"])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=["/public/dreamer4/expert-shards"])
    p.add_argument("--tasks_json", type=str, default="tasks.json")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)

    # Checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="logs/dynamics.pt")
    p.add_argument("--packing_factor", type=int, default=2)

    # Head architecture
    p.add_argument("--action_dim", type=int, default=16)
    p.add_argument("--mtp_length", type=int, default=8)
    p.add_argument("--head_mlp_ratio", type=float, default=2.0)

    # Shortcut/flow
    p.add_argument("--bootstrap_start", type=int, default=5_000)
    p.add_argument("--self_fraction", type=float, default=0.25)

    # Loss weights
    p.add_argument("--w_dynamics", type=float, default=1.0)
    p.add_argument("--w_bc", type=float, default=1.0)
    p.add_argument("--w_reward", type=float, default=1.0)

    # Optimization
    p.add_argument("--lr", type=float, default=3e-4, help="LR for heads + task embedder")
    p.add_argument("--lr_dynamics", type=float, default=1e-4, help="LR for dynamics (lower, since pre-trained)")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default="dreamer4-agent")
    p.add_argument("--wandb_run_name", type=str, default="phase2")
    p.add_argument("--wandb_entity", type=str, default=None)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="./logs/agent_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)

    train(p.parse_args())
