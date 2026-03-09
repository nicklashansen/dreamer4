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

import wandb

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from task_set import TASK_SET
from wm_dataset import WMDataset, collate_batch
from agent_diagnostics import AgentDiagnosticsHook, DiagnosticsConfig
from model import (
    Encoder, Decoder, Tokenizer, Dynamics, TaskEmbedder,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from agent import PolicyHead, RewardHead, TanhNormal
from train_dynamics import (
    get_dist_info, is_rank0, seed_everything, worker_init_fn,
    init_distributed, load_frozen_tokenizer_from_pt_ckpt,
    dynamics_pretrain_loss,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    override: Optional[Dict[str, Any]] = None
) -> tuple:
    """Load pre-trained dynamics and return model + info dict.
    """
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

    if n_latents % packing_factor != 0:
        raise ValueError(
            f"Incompatible tokenizer/dynamics packing: n_latents={n_latents} "
            f"is not divisible by packing_factor={packing_factor}."
        )
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    # Optional metadata checks from the dynamics checkpoint.
    if a.get("n_latents") is not None and int(a["n_latents"]) != int(n_latents):
        raise ValueError(
            f"Tokenizer/dynamics mismatch: tokenizer n_latents={n_latents}, "
            f"but dynamics ckpt expects n_latents={int(a['n_latents'])}."
        )
    if a.get("d_bottleneck") is not None and int(a["d_bottleneck"]) != int(d_bottleneck):
        raise ValueError(
            f"Tokenizer/dynamics mismatch: tokenizer d_bottleneck={d_bottleneck}, "
            f"but dynamics ckpt expects d_bottleneck={int(a['d_bottleneck'])}."
        )
    if a.get("packing_factor") is not None and int(a["packing_factor"]) != int(packing_factor):
        raise ValueError(
            f"Runtime/dynamics mismatch: packing_factor={packing_factor}, "
            f"but dynamics ckpt was trained with packing_factor={int(a['packing_factor'])}."
        )
    if a.get("n_spatial") is not None and int(a["n_spatial"]) != int(n_spatial):
        raise ValueError(
            f"Derived/ckpt mismatch: n_spatial={n_spatial}, "
            f"but dynamics ckpt metadata has n_spatial={int(a['n_spatial'])}."
        )
    if a.get("d_spatial") is not None and int(a["d_spatial"]) != int(d_spatial):
        raise ValueError(
            f"Derived/ckpt mismatch: d_spatial={d_spatial}, "
            f"but dynamics ckpt metadata has d_spatial={int(a['d_spatial'])}."
        )

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
        "n_heads": n_heads,
        "dyn_depth": depth,
        "dropout": dropout,
        "mlp_ratio": mlp_ratio,
        "time_every": time_every,
        "n_register": n_register,
        "space_mode": space_mode,
        "ckpt_args": a,
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
    """Behavioral cloning loss: masked NLL of dataset actions under the policy.

    MTP: for step l, predict action at t+l+1 from h_t at time t.
    Per-dim log probs are masked so padding dimensions don't contribute.

    If mask_l[..., j] = 0 for an invalid action dimension, the coordinate
    does not contribute to the loss. We also normalize losses by the number 
    of valid action dimensions 

    Args:
        h_t: (B, T, d_model)
        actions: (B, T, 16) — dataset actions
        act_mask: (B, T, 16) — 1.0 for valid dims, 0.0 for padding
        action_dim: effective action dimensions for current batch
        mtp_length: multi-token prediction length

    Returns:
        loss: scalar
        aux: dict
    """
    mu, std = policy(h_t)  # (B, T, L, A)
    B, T, L, A = mu.shape

    total_nll = torch.tensor(0.0, device=h_t.device)
    count = 0
    for l in range(min(L, mtp_length)):
        valid_T = T - l
        if valid_T <= 0:
            break
        target = actions[:, l:l + valid_T, :A].clamp(-1, 1)
        mask_l = act_mask[:, l:l + valid_T, :A]  # (B, valid_T, A)

        dist_l = TanhNormal(mu[:, :valid_T, l, :A], std[:, :valid_T, l, :A])
        # Per-dim log prob, masked so padding dims don't contribute
        per_dim = dist_l.log_prob_per_dim(target)  # (B, valid_T, A)
        masked = per_dim * mask_l
        nll = -masked.sum(dim=-1) / mask_l.sum(dim=-1).clamp(min=1)
        # Per-timestep NLL cap: prevents individual catastrophic samples
        # (e.g. random-policy mixed-large actions) from exploding the batch mean.
        # 50 nats ≈ chance-level for a 6-dim bounded action; anything higher
        # gives zero useful gradient signal.
        nll = nll.clamp(max=50.0)
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

    # Load frozen tokenizer
    tok_override = {}
    if args.H is not None: tok_override["H"] = args.H
    if args.W is not None: tok_override["W"] = args.W
    if args.C is not None: tok_override["C"] = args.C
    if args.patch is not None: tok_override["patch"] = args.patch
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device, override=(tok_override or None),
    )

    H_tokenizer = int(tok_args.get("H", 128))
    W_tokenizer = int(tok_args.get("W", 128))
    C_tokenizer = int(tok_args.get("C", 3))
    patch_tokenizer = int(tok_args.get("patch", 4))
    n_latents_tokenizer = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    H, W, C, patch = H_tokenizer, W_tokenizer, C_tokenizer, patch_tokenizer

    if n_latents_tokenizer % args.packing_factor != 0:
        raise ValueError(
            f"Incompatible tokenizer/packing: n_latents={n_latents_tokenizer} "
            f"is not divisible by packing_factor={args.packing_factor}."
        )
    n_spatial = n_latents_tokenizer // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Load pretrained dynamics model
    dyn_override = {}
    dyn_override["d_model_dyn"] = args.d_model_dyn
    dyn_override["dyn_depth"] = args.dyn_depth
    dyn_override["n_heads"] = args.n_heads
    dyn_override["dropout"] = args.dropout
    dyn_override["mlp_ratio"] = args.mlp_ratio
    dyn_override["time_every"] = args.time_every
    dyn_override["k_max"] = args.k_max
    dyn_override["n_register"] = args.n_register
    dyn_override["n_agent"] = args.n_agent
    if args.space_mode is not None: dyn_override["space_mode"] = args.space_mode
    
    dyn, dyn_info = load_pretrained_dynamics(
        args.dynamics_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents_tokenizer,
        packing_factor=args.packing_factor,
        override=dyn_override
    )
    dyn_ckpt_args = dyn_info.get("ckpt_args", {}) or {}
    for name, tok_val in (
        ("H", H_tokenizer),
        ("W", W_tokenizer),
        ("C", C_tokenizer),
        ("patch", patch_tokenizer),
        ("n_latents", n_latents_tokenizer),
        ("d_bottleneck", d_bottleneck),
        ("packing_factor", int(args.packing_factor)),
        ("n_spatial", n_spatial),
        ("d_spatial", d_spatial),
    ):
        if dyn_ckpt_args.get(name) is not None and int(dyn_ckpt_args[name]) != int(tok_val):
            raise ValueError(
                f"Tokenizer/dynamics mismatch for {name}: tokenizer/runtime={tok_val}, "
                f"dynamics ckpt={int(dyn_ckpt_args[name])}."
            )

    runtime_vs_loaded = (
        ("d_model_dyn", args.d_model_dyn, dyn_info["d_model"]),
        ("k_max", args.k_max, dyn_info["k_max"]),
        ("n_agent", args.n_agent, dyn_info["n_agent"]),
        ("n_heads", args.n_heads, dyn_info["n_heads"]),
        ("dyn_depth", args.dyn_depth, dyn_info["dyn_depth"]),
        ("n_register", args.n_register, dyn_info["n_register"]),
    )
    for name, expected, loaded in runtime_vs_loaded:
        if int(expected) != int(loaded):
            raise ValueError(
                f"Args/dynamics mismatch for {name}: args={int(expected)}, loaded={int(loaded)}."
            )
    if str(args.space_mode) != str(dyn_info["space_mode"]):
        raise ValueError(
            f"Args/dynamics mismatch for space_mode: args={args.space_mode}, loaded={dyn_info['space_mode']}."
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

    head_hidden = int(d_model * args.head_mlp_ratio)
    policy = PolicyHead(
        d_model=d_model,
        action_dim=args.action_dim,
        hidden=head_hidden,
        mtp_length=args.mtp_length,
    ).to(device)

    reward_head = RewardHead(
        d_model=d_model,
        hidden=head_hidden,
        mtp_length=args.mtp_length,
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

    # --- Diagnostics hook (optional) ---
    diag_path = args.diag_jsonl
    if (not diag_path) and args.diag_enable:
        diag_path = str(Path(args.ckpt_dir) / "diagnostics" / "phase2_diagnostics.jsonl")
    snapshot_dir = args.diag_snapshot_dir
    if (not snapshot_dir) and args.diag_enable:
        snapshot_dir = str(Path(args.ckpt_dir) / "diagnostics" / "spike_samples")
    diag_hook = AgentDiagnosticsHook(
        DiagnosticsConfig(
            enabled=bool(args.diag_enable),
            every=int(args.diag_every),
            topk=int(args.diag_topk),
            bc_spike_mult=float(args.diag_bc_spike_mult),
            bc_abs_threshold=float(args.diag_bc_abs_threshold),
            ema_decay=float(args.diag_ema_decay),
            jsonl_path=diag_path,
            snapshot_dir=snapshot_dir,
            snapshot_topk=int(args.diag_snapshot_topk),
        ),
        task_names=list(dataset.tasks),
        source_names=[f"{dd}|{fd}" for dd, fd in dataset.sources],
    )

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
        step, start_epoch = load_ckpt(
            Path(args.resume),
            dynamics=dyn, task_embedder=task_embedder,
            policy=policy, reward=reward_head,
            opt=opt, scaler=scaler,
        )
        if is_rank0():
            print(f"Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # --- RMS loss normalization (Dreamer v4, Section 3) ---
    # "We normalize all loss terms by running estimates of their RMS."
    # Each loss is divided by max(1, EMA(sqrt(loss^2))) so all terms
    # contribute ~equal gradient scale regardless of raw magnitude.
    rms_ema = {"dyn": 1.0, "bc": 1.0, "rew": 1.0}
    rms_decay = 0.99

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

                # Unpack batch
                obs_u8 = batch["obs"].to(device, non_blocking=True)      # (B, T+1, 3, H, W) uint8
                act = batch["act"].to(device, non_blocking=True)         # (B, T, 16) float
                mask = batch["act_mask"].to(device, non_blocking=True)   # (B, T, 16) float
                rew = batch["rew"].to(device, non_blocking=True)         # (B, T) float
                emb_id = batch["emb_id"].to(device, non_blocking=True)   # (B,) long
                task_id = batch["task_id"].to(device, non_blocking=True)       # (B,) long
                source_id = batch["source_id"].to(device, non_blocking=True)   # (B,) long
                segment_idx = batch["segment_idx"].to(device, non_blocking=True)   # (B,) long
                window_start = batch["window_start"].to(device, non_blocking=True) # (B,) long
                episode_id = batch["episode_id"].to(device, non_blocking=True)     # (B,) long
                sample_idx = batch["sample_idx"].to(device, non_blocking=True)     # (B,) long

                act = act.clamp(-1, 1) * mask

                # frames: obs[0..T-1], action alignment: action[t] produced obs[t+1]
                frames = obs_u8[:, :-1].float() / 255.0                 # (B, T, 3, H, W)
                # Shift actions: action[0] = zeros (no action produced first frame)
                actions = torch.zeros_like(act)
                actions[:, 1:] = act[:, :-1]
                act_mask_shifted = torch.zeros_like(mask)
                act_mask_shifted[:, 1:] = mask[:, :-1]

                B, T = frames.shape[:2]

                # Encode
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

                    # 2) Use h_t from the dynamics forward pass (noisy inputs)
                    # Per Dreamer4 paper (under Eq. 9) and JAX reference:
                    # BC/reward heads are supervised on h_t from the SAME forward
                    # pass as the dynamics loss, not from a separate clean pass.
                    h_pooled = dyn_aux["h_t"]  # (B, T, d_model)

                    # Phase 2: do NOT detach h_t — gradients from heads
                    # flow back to task_embedder and dynamics agent-token path.

                    # 3) BC loss
                    bc_l, bc_aux = bc_loss(
                        policy, h_pooled, act, mask,
                        action_dim=args.action_dim,
                        mtp_length=args.mtp_length,
                    )

                    # 4) Reward loss
                    rw_l, rw_aux = reward_loss(reward_head, h_pooled, rew)

                    # RMS-normalize each loss (Dreamer v4 paper) + loss clipping
                    with torch.no_grad():
                        rms_ema["dyn"] = rms_decay * rms_ema["dyn"] + (1 - rms_decay) * abs(dyn_loss.item())
                        rms_ema["bc"] = rms_decay * rms_ema["bc"] + (1 - rms_decay) * abs(bc_l.item())
                        rms_ema["rew"] = rms_decay * rms_ema["rew"] + (1 - rms_decay) * abs(rw_l.item())

                    # Clip each loss to 5x its running average to prevent spike-driven collapse
                    clip_mult = 5.0
                    dyn_clip = max(5.0, clip_mult * rms_ema["dyn"])
                    bc_clip = max(5.0, clip_mult * rms_ema["bc"])
                    rew_clip = max(5.0, clip_mult * rms_ema["rew"])

                    total_loss = (
                        args.w_dynamics * dyn_loss.clamp(max=dyn_clip) / max(1.0, rms_ema["dyn"])
                        + args.w_bc * bc_l.clamp(max=bc_clip) / max(1.0, rms_ema["bc"])
                        + args.w_reward * rw_l.clamp(max=rew_clip) / max(1.0, rms_ema["rew"])
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

                if is_rank0():
                    elapsed = (time.time() - t0) / 3600.0
                    spike_msg = diag_hook.maybe_record(
                        step=step,
                        bc_loss=float(bc_l.item()),
                        dyn_loss=float(dyn_loss.item()),
                        rew_loss=float(rw_l.item()),
                        total_loss=float(total_loss.item()),
                        flow_mse=float(dyn_aux["flow_mse"].item()),
                        boot_mse=float(dyn_aux["bootstrap_mse"].item()),
                        h_t=h_pooled.detach(),
                        policy=policy,
                        actions=act.detach(),
                        act_mask=mask.detach(),
                        rewards=rew.detach(),
                        obs_u8=obs_u8.detach(),
                        mtp_length=int(args.mtp_length),
                        task_ids=task_id.detach(),
                        source_ids=source_id.detach(),
                        segment_idx=segment_idx.detach(),
                        window_start=window_start.detach(),
                        episode_id=episode_id.detach(),
                        sample_idx=sample_idx.detach(),
                        reward_pred_mean=float(rw_aux["reward_pred_mean"].item()),
                        reward_target_mean=float(rw_aux["reward_target_mean"].item()),
                        elapsed_hours=float(elapsed),
                    )
                    if spike_msg is not None:
                        print(spike_msg)

                # --- Checkpointing ---
                if is_rank0() and args.save_every > 0 and step > 0 and (step % args.save_every == 0) and do_step:
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
    diag_hook.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2: Agent Finetuning (BC + Reward + Dynamics)")

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
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Task subset. Default: all 30 TASK_SET tasks.")

    # tokenizer restore
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    # dynamics arch
    p.add_argument("--dynamics_ckpt", type=str, default="logs/dynamics.pt")
    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)

    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)
    p.add_argument("--space_mode", type=str, default="wm_agent",
                   help="Attention mode for agent tokens. Use 'wm_agent' so agent tokens "
                        "can attend to actions/state (required for RL).")

    # behavior clone and reward head architecture
    p.add_argument("--action_dim", type=int, default=16)
    p.add_argument("--mtp_length", type=int, default=8)
    p.add_argument("--head_mlp_ratio", type=float, default=2.0)

    # shortcut / schedule
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--bootstrap_start", type=int, default=0) # NOTE: may need to be 0, chnged from 5_000
    p.add_argument("--self_fraction", type=float, default=0.25)

    # loss weights
    p.add_argument("--w_dynamics", type=float, default=1.0)
    p.add_argument("--w_bc", type=float, default=1.0)
    p.add_argument("--w_reward", type=float, default=1.0)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4, help="LR for heads + task embedder")
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

    # Diagnostics
    p.add_argument("--diag_enable", action="store_true",
                   help="Enable detailed per-batch diagnostics for BC outlier analysis.")
    p.add_argument("--diag_every", type=int, default=25,
                   help="Emit detailed diagnostics every N steps.")
    p.add_argument("--diag_topk", type=int, default=3,
                   help="Top-K outlier samples (by BC NLL) to log per diagnostic event.")
    p.add_argument("--diag_bc_spike_mult", type=float, default=8.0,
                   help="Spike threshold multiplier over BC EMA.")
    p.add_argument("--diag_bc_abs_threshold", type=float, default=100.0,
                   help="Absolute BC threshold for spike detection.")
    p.add_argument("--diag_ema_decay", type=float, default=0.99,
                   help="EMA decay used for BC spike thresholding.")
    p.add_argument("--diag_jsonl", type=str, default="",
                   help="Path to JSONL diagnostics file (default: <ckpt_dir>/diagnostics/phase2_diagnostics.jsonl).")
    p.add_argument("--diag_snapshot_dir", type=str, default="",
                   help="Directory to dump full top-K spike samples as .pt files.")
    p.add_argument("--diag_snapshot_topk", type=int, default=3,
                   help="Top-K outlier samples to dump per spike event.")

    train(p.parse_args())
