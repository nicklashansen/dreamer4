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
from agent import PolicyHead, RewardHead, TanhNormal, TwoHotDist, _check_head_config
from train_dynamics import (
    get_dist_info, is_rank0, seed_everything, worker_init_fn,
    init_distributed, load_frozen_tokenizer_from_pt_ckpt,
    dynamics_pretrain_loss, make_tau_schedule, run_dynamics_eval,
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
        "rw_num_bins": args.rw_num_bins,
        "rw_low": args.rw_low,
        "rw_high": args.rw_high,
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
    _check_head_config(policy, ckpt, "policy")
    _check_head_config(reward, ckpt, "rw")
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
    sample_weight: Optional[torch.Tensor] = None,
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
        sample_weight: optional (B,) float tensor; rows with weight 0 are
            excluded from the mean (e.g. pass expert_mask to train BC only
            on expert data).  If None, all rows contribute equally.

    Returns:
        loss: scalar
        aux: dict
    """

    # only one forward pass for the l tokens
    mu, std = policy(h_t)  # (B, T, L, A)
    B, T, L, A = mu.shape

    # expert_mask: (B,) float, 1.0 for expert rows, 0.0 otherwise
    if sample_weight is not None:
        expert_mask = sample_weight.float().to(h_t.device)   # (B,)
    else:
        expert_mask = torch.ones(B, device=h_t.device)

    expert_n = expert_mask.sum().clamp(min=1e-6)              # number of expert samples

    total_nll = torch.tensor(0.0, device=h_t.device)
    count = 0
    for l in range(min(L, mtp_length)):
        valid_T = T - l
        if valid_T <= 0:
            break
        target = actions[:, l:l + valid_T, :A].clamp(-1, 1)
        mask_l = act_mask[:, l:l + valid_T, :A]  # (B, valid_T, A)

        dist_l = TanhNormal(mu[:, :valid_T, l, :A], std[:, :valid_T, l, :A])
        # Per-dim log prob, masked so padding dims don't contribute.
        # Also zero out non-expert rows so they don't affect the loss.
        per_dim = dist_l.log_prob_per_dim(target)              # (B, valid_T, A)
        masked = per_dim * mask_l * expert_mask[:, None, None] # zero non-expert rows
        nll = -masked.sum(dim=-1) / mask_l.sum(dim=-1).clamp(min=1)  # (B, valid_T)
        # Per-timestep NLL cap: prevents individual catastrophic samples
        # (e.g. random-policy mixed-large actions) from exploding the batch mean.
        # 50 nats ≈ chance-level for a 6-dim bounded action; anything higher
        # gives zero useful gradient signal.
        nll = nll.clamp(max=50.0)
        # Average only over expert rows (non-expert rows are 0 so excluded from sum)
        total_nll = total_nll + nll.sum() / (expert_n * valid_T)
        count += 1

    loss = total_nll / max(count, 1)
    # std over expert rows only (non-expert rows would inflate/deflate the metric)
    exp_w = expert_mask[:, None, None, None].expand_as(std)
    expert_std = (std * exp_w).sum() / exp_w.sum().clamp(min=1e-6)
    aux = {
        "bc_nll": loss.detach(),
        "bc_mean_std": expert_std.detach(),
        "bc_expert_frac": (expert_mask > 0).float().mean().detach(),
    }
    return loss, aux


def reward_loss(
    reward_head: RewardHead,
    h_t: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple:
    """Reward prediction loss: two-hot CE of dataset rewards.

    MTP: for step l, predict reward at t+l from h_t at time t. note that this is the reward that comes after taking a_t.

    Args:
        h_t: (B, T, d_model) — detached agent token outputs
        rewards: (B, T) — dataset rewards

    Returns:
        loss: scalar
        aux: dict
    """
    logits = reward_head(h_t)  # (B, T, L, num_bins)
    B, T, L, _ = logits.shape

    total_nll = torch.tensor(0.0, device=h_t.device)
    count = 0
    for l in range(L):
        # For MTP step l, target at time t is reward at t+l+1
        # But targets are already aligned: targets[:, t] = reward for transition at t
        # So for step l, we shift: target is targets[:, t+l] for input h_t[:, t]
        valid_T = T - l
        if valid_T <= 0:
            break
        dist_l = TwoHotDist(logits[:, :valid_T, l], low=reward_head.low, high=reward_head.high)
        target_l = rewards[:, l:l + valid_T]
        nll = -dist_l.log_prob(target_l)  # (B, valid_T)
        total_nll = total_nll + nll.mean()
        count += 1

    loss = total_nll / max(count, 1)
    pred = reward_head.predict(h_t, step=0)
    aux = {
        "reward_ce": loss.detach(),
        "reward_pred_mean": pred.mean().detach(),
        "reward_target_mean": rewards.mean().detach(),
    }
    return loss, aux


# ---------------------------------------------------------------------------
# BC eval (clean forward pass, wandb logging)
# ---------------------------------------------------------------------------


def log_bc_eval_wandb(
    *,
    frames: torch.Tensor,       # (B, T, C, H, W) float [0,1]
    mu: torch.Tensor,           # (B, T, L, A) policy mean (pre-tanh)
    act: torch.Tensor,          # (B, T, 16) ground-truth actions
    act_mask: torch.Tensor,     # (B, T, 16)
    reward_pred: torch.Tensor,  # (B, T)
    rewards: torch.Tensor,      # (B, T)
    step: int,
    tag: str = "bc_eval",
    max_items: int = 4,
    bar_h: int = 8,
    gap_px: int = 4,
):
    """Log a visual panel to wandb: frame strip + action prediction bars.

    For each example in the batch (up to max_items):
      Row 1: GT frames tiled across time
      Row 2: Predicted action dims (policy mean, l=0, clamped to [-1,1]) as
             a color bar per timestep — blue=−1, white=0, red=+1
      Row 3: GT action dims, same coloring
      Row 4: Absolute error |pred - gt| — white=0, red=1
      Row 5: Reward predicted (red) vs GT (blue) as a thin color bar

    The panel is logged as a single wandb.Image per batch item.
    """
    B, T, C, H, W = frames.shape
    Bv = min(B, max_items)
    A_full = act.shape[2]
    # number of action dims actually used (from mask first row)
    n_act = int(act_mask[0, 0].sum().long().clamp(min=1, max=A_full).item())

    action_pred = mu[:Bv, :, 0, :n_act].tanh().float().cpu()  # (Bv, T, n_act)
    action_gt = act[:Bv, :, :n_act].clamp(-1, 1).float().cpu()  # (Bv, T, n_act)
    act_err = (action_pred - action_gt).abs()                    # (Bv, T, n_act)
    r_pred = reward_pred[:Bv].float().cpu()  # (Bv, T)
    r_gt = rewards[:Bv].float().cpu()        # (Bv, T)

    def _action_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """x: (...) in [-1,1] -> (..., 3) uint8. Blue=-1, White=0, Red=+1."""
        x = x.clamp(-1, 1)
        pos = x.clamp(0, 1)
        neg = (-x).clamp(0, 1)
        r = (1.0 - neg).clamp(0, 1)
        g = (1.0 - pos - neg).clamp(0, 1)
        b = (1.0 - pos).clamp(0, 1)
        return (torch.stack([r, g, b], dim=-1) * 255).to(torch.uint8)

    def _err_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """x: (...) in [0,1] -> (..., 3) uint8. White=0, Red=1."""
        x = x.clamp(0, 1)
        r = torch.ones_like(x)
        g = (1.0 - x)
        b = (1.0 - x)
        return (torch.stack([r, g, b], dim=-1) * 255).to(torch.uint8)

    def _reward_bar(r: torch.Tensor, color: tuple, h: int, W_: int) -> torch.Tensor:
        """r: (T,) scalar reward -> (h, W_, 3) uint8 color strip, intensity ~ |r|."""
        r_norm = r.clamp(-1, 1) * 0.5 + 0.5  # [0,1]
        strip = torch.zeros(h, W_, 3, dtype=torch.uint8)
        col = torch.tensor(color, dtype=torch.float32)
        for t in range(T):
            x0 = t * (W_ // T)
            x1 = (t + 1) * (W_ // T) if t < T - 1 else W_
            strip[:, x0:x1] = (col * float(r_norm[t])).to(torch.uint8)
        return strip

    panels = []
    for i in range(Bv):
        # --- Frame strip: (C, H, T*W) ---
        fr = (frames[i].float().cpu().clamp(0, 1) * 255).to(torch.uint8)  # (T,C,H,W)
        frame_strip = fr.permute(1, 2, 0, 3).reshape(C, H, T * W)  # (C, H, T*W)
        panel_W = T * W

        # --- Action bars: each action dim becomes bar_h rows ---
        # pred: (n_act, T) -> (n_act, bar_h, T, 3)
        pred_rgb = _action_to_rgb(action_pred[i].T)  # (n_act, T, 3)
        gt_rgb = _action_to_rgb(action_gt[i].T)       # (n_act, T, 3)
        err_rgb = _err_to_rgb(act_err[i].T)            # (n_act, T, 3)

        def _expand_bar(x_rgb, bar_h_, W_):
            # x_rgb: (n_act, T, 3) -> (3, n_act*bar_h_, W_)
            n = x_rgb.shape[0]
            out = torch.zeros(3, n * bar_h_, W_, dtype=torch.uint8)
            for d in range(n):
                col = x_rgb[d]  # (T, 3)
                for t in range(T):
                    x0 = t * (W_ // T)
                    x1 = (t + 1) * (W_ // T) if t < T - 1 else W_
                    for c in range(3):
                        out[c, d * bar_h_:(d + 1) * bar_h_, x0:x1] = col[t, c]
            return out

        pred_bar = _expand_bar(pred_rgb, bar_h, panel_W)
        gt_bar = _expand_bar(gt_rgb, bar_h, panel_W)
        err_bar = _expand_bar(err_rgb, bar_h, panel_W)

        # Reward strips (red = predicted, blue = gt)
        r_pred_strip = _reward_bar(r_pred[i], (255, 80, 80), bar_h, panel_W)   # red
        r_gt_strip = _reward_bar(r_gt[i], (80, 80, 255), bar_h, panel_W)        # blue
        r_pred_strip = r_pred_strip.permute(2, 0, 1)  # (3, bar_h, panel_W)
        r_gt_strip = r_gt_strip.permute(2, 0, 1)

        gap = torch.zeros(3, gap_px, panel_W, dtype=torch.uint8)

        # Stack: frame | gap | pred_actions | gap | gt_actions | gap | err | gap | rew_pred | rew_gt
        rows = [frame_strip, gap, pred_bar, gap, gt_bar, gap, err_bar, gap, r_pred_strip, r_gt_strip]
        panel_img = torch.cat(rows, dim=1)  # (3, total_H, panel_W)
        panels.append(panel_img)

    # Stack all examples vertically
    big = torch.cat(panels, dim=1)  # (3, Bv*total_H, panel_W)
    big_hwc = big.permute(1, 2, 0).numpy()

    wandb.log(
        {
            f"{tag}/viz": wandb.Image(
                big_hwc,
                caption=(
                    f"rows per example: frames | pred_actions | gt_actions | err | rew_pred(red) rew_gt(blue)"
                    f" | ctx=all T={T} A={n_act}"
                ),
            )
        },
        step=step,
    )


@torch.no_grad()
def run_bc_eval(
    *,
    encoder: Encoder,
    dyn,
    task_embedder: TaskEmbedder,
    policy: PolicyHead,
    reward_head: RewardHead,
    frames: torch.Tensor,   # (B, T, C, H, W) float [0,1]
    act: torch.Tensor,      # (B, T, 16) raw actions (unshifted)
    act_mask: torch.Tensor, # (B, T, 16)
    rewards: torch.Tensor,  # (B, T)
    emb_ids: torch.Tensor,  # (B,)
    task_ids: torch.Tensor, # (B,)
    n_spatial: int,
    packing_factor: int,
    patch: int,
    k_max: int,
    action_dim: int,
    mtp_length: int,
    step: int,
    task_names: Optional[list] = None,
    tag: str = "bc_eval",
):
    """Evaluate BC and reward heads via a clean (noise-free) dynamics forward pass.

    Unlike training (which uses noisy diffusion inputs), we run at the finest
    step (emax) with clean signal (sigma_idx = k_max-1) on ground-truth latents.
    This gives h_t that matches what the policy sees during actual rollout, so
    the BC metrics are not confounded by noise corruption.

    Logs to wandb:
      bc_eval/bc_nll              -- mean NLL across MTP steps
      bc_eval/bc_mean_std         -- mean policy std (lower = more confident)
      bc_eval/action_mae          -- |policy_mean - target| for valid dims
      bc_eval/reward_mse          -- MSE predicted vs actual rewards
      bc_eval/reward_mae          -- MAE predicted vs actual rewards
      bc_eval/reward_corr         -- Pearson r(predicted, actual)
      bc_eval/reward_pred_mean
      bc_eval/reward_target_mean
      bc_eval/task_<name>/bc_nll  -- per-task NLL breakdown
    """
    device = frames.device
    B, T = frames.shape[:2]

    was_training = {
        "dyn": dyn.training,
        "te": task_embedder.training,
        "pi": policy.training,
        "rw": reward_head.training,
    }
    dyn.eval(); task_embedder.eval(); policy.eval(); reward_head.eval()

    # Encode frames (encoder is always frozen/eval)
    patches = temporal_patchify(frames, patch)
    z_btLd, _ = encoder(patches)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)  # (B,T,Sz,Dz)

    agent_tokens = task_embedder(emb_ids, B=B, T=T)  # (B, T, n_agent, d_model)

    # Clean dynamics forward: finest step + clean signal, no noise
    emax = int(round(math.log2(k_max)))
    step_idxs = torch.full((B, T), emax, device=device, dtype=torch.long)
    signal_idxs = torch.full((B, T), k_max - 1, device=device, dtype=torch.long)

    # Shift actions (first timestep = zeros), same convention as training
    actions_shifted = torch.zeros_like(act)
    actions_shifted[:, 1:] = act[:, :-1]
    mask_shifted = torch.zeros_like(act_mask)
    mask_shifted[:, 1:] = act_mask[:, :-1]

    _, h_t_full = dyn(
        actions_shifted, step_idxs, signal_idxs, z1,
        act_mask=mask_shifted, agent_tokens=agent_tokens,
    )
    h_pooled = h_t_full.mean(dim=2)  # (B, T, d_model)

    # BC NLL
    mu, std = policy(h_pooled)  # (B, T, L, A)
    L_mtp, A = mu.shape[2], mu.shape[3]

    nll_steps = []
    for l in range(min(L_mtp, mtp_length)):
        valid_T = T - l
        if valid_T <= 0:
            break
        target = act[:, l:l + valid_T, :A].clamp(-1, 1)
        mask_l = act_mask[:, l:l + valid_T, :A]
        dist_l = TanhNormal(mu[:, :valid_T, l, :A], std[:, :valid_T, l, :A]) # gets masked right after
        per_dim = dist_l.log_prob_per_dim(target)
        nll = -(per_dim * mask_l).sum(dim=-1) / mask_l.sum(dim=-1).clamp(min=1)
        nll_steps.append(nll.clamp(max=50.0))  # (B, valid_T)

    if nll_steps:
        nll_per_sample = torch.stack([s.mean(dim=1) for s in nll_steps]).mean(dim=0)  # (B,)
        bc_nll_mean = float(nll_per_sample.mean().item())
    else:
        nll_per_sample = torch.zeros(B, device=device)
        bc_nll_mean = 0.0

    bc_std_mean = float(std[..., :action_dim].mean().item())

    # Action MAE: policy mean (l=0) vs target
    action_pred = mu[:, :, 0, :action_dim].tanh()
    action_tgt = act[:, :, :action_dim].clamp(-1, 1)
    action_msk = act_mask[:, :, :action_dim]
    action_mae = float(
        ((action_pred - action_tgt).abs() * action_msk).sum() / action_msk.sum().clamp(min=1)
    )

    # Reward quality
    rw_dist = reward_head.dist(h_pooled, step=0)          # TwoHotDist over (B, T, num_bins)
    reward_pred = rw_dist.mean                             # (B, T)
    reward_mse = float((reward_pred - rewards).pow(2).mean().item())
    reward_mae_val = float((reward_pred - rewards).abs().mean().item())
    rp, rt = reward_pred.flatten(), rewards.flatten()
    reward_corr = float(torch.corrcoef(torch.stack([rp, rt]))[0, 1].item()) \
        if rp.std() > 1e-6 and rt.std() > 1e-6 else 0.0

    # Bin collapse diagnostics (reward head)
    rw_probs = rw_dist.probs                               # (B, T, num_bins)
    rw_entropy = -(rw_probs * (rw_probs + 1e-8).log()).sum(dim=-1).mean()  # scalar
    rw_max_entropy = math.log(rw_dist.num_bins)
    rw_entropy_ratio = float(rw_entropy.item() / rw_max_entropy)
    rw_argmax = rw_probs.argmax(dim=-1)                    # (B, T)
    rw_n_unique = int(rw_argmax.unique().numel())

    # Per-task NLL breakdown
    per_task_logs = {}
    if task_names is not None:
        for tid in task_ids.unique():
            mask_b = task_ids == tid
            tid_int = int(tid.item())
            tname = task_names[tid_int] if tid_int < len(task_names) else f"task_{tid_int}"
            per_task_logs[f"{tag}/task_{tname}/bc_nll"] = float(nll_per_sample[mask_b].mean().item())

    log_dict = {
        f"{tag}/bc_nll":             bc_nll_mean,
        f"{tag}/bc_mean_std":        bc_std_mean,
        f"{tag}/action_mae":         action_mae,
        f"{tag}/reward_mse":         reward_mse,
        f"{tag}/reward_mae":         reward_mae_val,
        f"{tag}/reward_corr":        reward_corr,
        f"{tag}/reward_pred_mean":   float(reward_pred.mean().item()),
        f"{tag}/reward_target_mean": float(rewards.mean().item()),
        f"{tag}/rw_entropy_ratio":   rw_entropy_ratio,       # 0=collapsed, 1=max uncertainty
        f"{tag}/rw_n_unique_bins":   rw_n_unique,            # out of reward_head.num_bins
    }
    log_dict.update(per_task_logs)
    wandb.log(log_dict, step=step)

    print(
        f"[bc_eval step {step:07d}] "
        f"bc_nll={bc_nll_mean:.4f} "
        f"bc_std={bc_std_mean:.3f} "
        f"act_mae={action_mae:.4f} "
        f"rew_mse={reward_mse:.5f} "
        f"rew_corr={reward_corr:.3f} "
        f"rw_ent={rw_entropy_ratio:.2f} rw_bins={rw_n_unique}/{reward_head.num_bins}"
    )

    # Visual panel: frame strip + action prediction heatmap
    log_bc_eval_wandb(
        frames=frames,
        mu=mu,
        act=act,
        act_mask=act_mask,
        reward_pred=reward_pred,
        rewards=rewards,
        step=step,
        tag=tag,
        max_items=4,
    )

    if was_training["dyn"]: dyn.train()
    if was_training["te"]:  task_embedder.train()
    if was_training["pi"]:  policy.train()
    if was_training["rw"]:  reward_head.train()


def check_agreement(args, dyn_info, H_tokenizer, W_tokenizer, C_tokenizer,
                    patch_tokenizer, n_latents_tokenizer, d_bottleneck,
                    n_spatial, d_spatial):
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


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

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
    # Map source_id (int) -> short dataset name (basename of data_dir)
    source_id_to_name = {i: os.path.basename(dd) for i, (dd, _) in enumerate(dataset.sources)}

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
    check_agreement(args, dyn_info, H_tokenizer, W_tokenizer, C_tokenizer,
                    patch_tokenizer, n_latents_tokenizer, d_bottleneck,
                    n_spatial, d_spatial)

    d_model = dyn_info["d_model"]
    k_max = dyn_info["k_max"]
    n_agent = dyn_info["n_agent"]

    # Build heads for BC & R
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
        num_bins=args.rw_num_bins,
        low=args.rw_low,
        high=args.rw_high,
    ).to(device)

    if is_rank0():
        dyn_params = sum(p.numel() for p in dyn.parameters())
        te_params = sum(p.numel() for p in task_embedder.parameters())
        pi_params = sum(p.numel() for p in policy.parameters())
        rw_params = sum(p.numel() for p in reward_head.parameters())
        print(f"Parameters: dynamics={dyn_params:,} task_emb={te_params:,} policy={pi_params:,} reward={rw_params:,}")
        print(f"Total trainable: {dyn_params + te_params + pi_params + rw_params:,}")

    if args.compile:
        dyn = torch.compile(dyn)
    
    # --- DDP ---
    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
        )

    # Optimizer and scaler
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
        source_names=[source_id_to_name[i] for i in range(len(dataset.sources))],
    )

    # Initialize wandb
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online" if args.wandb_project else "disabled",
            config=vars(args),
        )

    # Resume from checkpoint
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

    # RMS loss normalization (Dreamer v4, Section 3)
    # "We normalize all loss terms by running estimates of their RMS."
    # Each loss is divided by max(1, EMA(sqrt(loss^2))) so all terms
    # contribute ~equal gradient scale regardless of raw magnitude.
    rms_ema = {"dyn": 1.0, "bc": 1.0, "rew": 1.0}
    rms_decay = 0.99

    # Source IDs that correspond to expert data — computed once, used every step
    expert_src_ids = {i for i, name in source_id_to_name.items() if name == "expert"}

    # Training loop
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
                source_id = batch["source_id"].to(device, non_blocking=True)   # (B,) long; 
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

                # Frozen encoder -> packed spatial tokens z1
                with torch.no_grad():
                    patches = temporal_patchify(frames, patch)
                    z_btLd, _ = encoder(patches)
                    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)

                # Agent tokens from TaskEmbedder
                B, T = frames.shape[:2]
                agent_tokens = task_embedder(emb_id, B=B, T=T)  # (B, T, n_agent, d_model)

                B_self = int(round(args.self_fraction * B))
                B_self = max(0, min(B - 1, B_self))

                # Expert mask: 1.0 for expert rows, 0.0 for non-expert
                expert_mask = torch.zeros(source_id.shape[0], device=device)
                for _sid in expert_src_ids:
                    expert_mask = expert_mask + (source_id == _sid).float()
                expert_mask = expert_mask.clamp(max=1.0)
                nonexpert_mask = 1.0 - expert_mask  # dynamics trains on non-expert only

                # Masking expert rows from the diffusion loss avoids optimistic generations:
                # a world model trained on expert data learns to hallucinate expert-quality
                # transitions, causing imagination rollouts to be unrealistically good.
                if args.mask_expert_diffusion_loss:
                    diffusion_mask = nonexpert_mask  # train dynamics on non-expert rows only
                else:
                    diffusion_mask = None  # no masking: all rows contribute equally

                with autocast(device_type="cuda", enabled=use_amp):
                    # 1) Dynamics shortcut/flow loss — non-expert rows only
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
                        diffusion_mask=diffusion_mask,
                    )

                    # 2) Use h_t from the dynamics forward pass (noisy inputs)
                    # Per Dreamer4 paper (under Eq. 9) and JAX reference:
                    # BC/reward heads are supervised on h_t from the SAME forward
                    # pass as the dynamics loss, not from a separate clean pass.
                    h_pooled = dyn_aux["h_t"]  # (B, T, d_model)

                    # Phase 2: do NOT detach h_t — gradients from heads
                    # flow back to task_embedder and dynamics agent-token path
                    # But for early training, we may want to detach to stabilize BC / R
                    if step < args.bc_rew_warmup:
                        h_pooled = h_pooled.detach()


                    # 3) BC loss — only on expert data
                    bc_l, bc_aux = bc_loss(
                        policy, h_pooled, act, mask,
                        action_dim=args.action_dim,
                        mtp_length=args.mtp_length,
                        sample_weight=expert_mask,
                    )

                    # 4) Reward loss
                    rw_l, rw_aux = reward_loss(reward_head, h_pooled, rew)

                    # RMS-normalize each loss (Dreamer v4 paper) + loss clipping
                    with torch.no_grad():
                        rms_ema["dyn"] = args.rms_decay * rms_ema["dyn"] + (1 - args.rms_decay) * abs(dyn_loss.item())
                        rms_ema["bc"] = args.rms_decay * rms_ema["bc"] + (1 - args.rms_decay) * abs(bc_l.item())
                        rms_ema["rew"] = args.rms_decay * rms_ema["rew"] + (1 - args.rms_decay) * abs(rw_l.item())

                    # Clip each loss to clip_mult x its running average (clip_mult=0 disables clipping)
                    def _maybe_clip(loss, ema_val):
                        if args.clip_mult == 0:
                            return loss
                        return loss.clamp(max=max(5.0, args.clip_mult * ema_val))

                    total_loss = (
                        args.w_dynamics * _maybe_clip(dyn_loss, rms_ema["dyn"]) / max(1.0, rms_ema["dyn"])
                        + args.w_bc * _maybe_clip(bc_l, rms_ema["bc"]) / max(1.0, rms_ema["bc"])
                        + args.w_reward * _maybe_clip(rw_l, rms_ema["rew"]) / max(1.0, rms_ema["rew"])
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
                        f"outputs {args.ckpt_dir} | "
                        f"step {step:07d} | "
                        f"total={total_loss.item():.4f} "
                        f"dyn={dyn_loss.item():.4f} "
                        f"bc={bc_l.item():.4f}(exp={bc_aux['bc_expert_frac'].item():.2f}) "
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
                        "stats/bc_expert_frac": bc_aux["bc_expert_frac"].item(),
                        "stats/reward_pred_mean": rw_aux["reward_pred_mean"].item(),
                        "stats/reward_target_mean": rw_aux["reward_target_mean"].item(),
                        "time/hrs": elapsed,
                    }, step=step)

                # --- BC eval + dynamics visual eval ---
                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0):
                    B_ev = min(B, args.eval_batch_size)
                    run_bc_eval(
                        encoder=encoder,
                        dyn=dyn.module if hasattr(dyn, "module") else dyn,
                        task_embedder=task_embedder,
                        policy=policy,
                        reward_head=reward_head,
                        frames=frames[:B_ev],
                        act=act[:B_ev],
                        act_mask=mask[:B_ev],
                        rewards=rew[:B_ev],
                        emb_ids=emb_id[:B_ev],
                        task_ids=task_id[:B_ev],
                        n_spatial=n_spatial,
                        packing_factor=args.packing_factor,
                        patch=patch,
                        k_max=k_max,
                        action_dim=args.action_dim,
                        mtp_length=args.mtp_length,
                        step=step,
                        task_names=list(dataset.tasks),
                    )
                    # Dynamics visual eval: autoregressive rollout + GT/Pred frame panel
                    sched = make_tau_schedule(
                        k_max=k_max, schedule=args.eval_schedule, d=args.eval_d
                    )
                    run_dynamics_eval(
                        encoder=encoder,
                        decoder=decoder,
                        dyn=dyn.module if hasattr(dyn, "module") else dyn,
                        frames=frames[:B_ev],
                        actions=actions[:B_ev],
                        act_mask=act_mask_shifted[:B_ev],
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=args.packing_factor,
                        k_max=k_max,
                        ctx_length=args.eval_ctx,
                        horizon=args.eval_horizon,
                        sched=sched,
                        max_items=args.eval_batch_size,
                        step=step,
                    )

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
                if is_rank0() and args.save_every > 0 and step > 0 and (step % args.save_every == 0):
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

    # data (if using multiple datasets, make sure they align in order)
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
    p.add_argument("--tasks_json", type=str, default="./tasks.json") # task metadata
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Task subset. Default: all 30 TASK_SET tasks.") # tasks to use here

    # tokenizer restore
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    # dynamics restore and arch
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

    # shortcut / schedule
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--bootstrap_start", type=int, default=0) # NOTE: may need to be 0, chnged from 5_000
    p.add_argument("--self_fraction", type=float, default=0.25)

    # behavior clone and reward head architecture
    p.add_argument("--action_dim", type=int, default=16)
    p.add_argument("--mtp_length", type=int, default=8)
    p.add_argument("--head_mlp_ratio", type=float, default=2.0)
    p.add_argument("--rw_num_bins", type=int, default=11,
                   help="Number of TwoHot bins for reward head")
    p.add_argument("--rw_low", type=float, default=0.0,
                   help="Lower bound of reward head bins in symlog space")
    p.add_argument("--rw_high", type=float, default=1.25,
                   help="Upper bound of reward head bins in symlog space")

    # loss weights
    p.add_argument("--w_dynamics", type=float, default=1.0)
    p.add_argument("--w_bc", type=float, default=1.0)
    p.add_argument("--w_reward", type=float, default=1.0)

    # data filter
    p.add_argument("--mask_expert_diffusion_loss", action="store_true") # specified as true in github/pretrained
    
    # optim
    p.add_argument("--lr", type=float, default=1e-4, help="LR for heads + task embedder")
    p.add_argument("--lr_dynamics", type=float, default=1e-4, help="LR for dynamics (lower, since pre-trained)")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # extra training stabilization
    p.add_argument("--bc_rew_warmup", type=int, default=0) # train only heads to stabilize (?)
    p.add_argument("--rms_decay", type=float, default=0.99,
                   help="EMA decay for RMS loss normalization. Set to 1.0 to disable decay (static normalization).")
    p.add_argument("--clip_mult", type=float, default=5.0,
                   help="Loss clipping multiplier relative to RMS EMA (e.g. 5 = clip at 5x running avg).")


    # eval / viz
    p.add_argument("--eval_every", type=int, default=1_000,
                   help="Run BC + dynamics eval every N steps. 0 to disable.")
    p.add_argument("--eval_batch_size", type=int, default=24,
                   help="Batch size for eval (subset of training batch).")
    p.add_argument("--eval_ctx", type=int, default=8,
                   help="Context length for dynamics autoregressive eval.")
    p.add_argument("--eval_horizon", type=int, default=16,
                   help="Prediction horizon for dynamics autoregressive eval.")
    p.add_argument("--eval_schedule", type=str, default="shortcut", choices=["finest", "shortcut"],
                   help="Denoising schedule for dynamics eval rollout.")
    p.add_argument("--eval_d", type=float, default=0.25,
                   help="Step size d for shortcut schedule during eval.")

    # logging
    p.add_argument("--log_every", type=int, default=100)

    # wandb
    p.add_argument("--wandb_project", type=str, default="Dreamer 4 Continuous Control")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default="eqforcing")

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/agent_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

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
