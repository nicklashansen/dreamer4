# train_dynamics.py
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

import wandb

from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset

from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
    Dynamics,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def save_ckpt(path: Path, *, step: int, epoch: int, dyn_model, opt, scaler, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "dynamics": (dyn_model.module.state_dict() if hasattr(dyn_model, "module") else dyn_model.state_dict()),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, dyn_model, opt, scaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["dynamics"]
    (dyn_model.module if hasattr(dyn_model, "module") else dyn_model).load_state_dict(state, strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


@torch.no_grad()
def load_frozen_tokenizer_from_pt_ckpt(
    ckpt_path: str,
    *,
    device: torch.device,
    override: Optional[Dict[str, Any]] = None,
) -> tuple[Encoder, Decoder, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = dict(ckpt.get("args", {}))
    if override:
        tok_args.update(override)

    # Required keys (fall back to defaults if missing)
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        d_model=int(tok_args.get("d_model", 256)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
    )

    tok = Tokenizer(enc, dec)
    tok.load_state_dict(ckpt["model"], strict=True)

    tok = tok.to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok.encoder, tok.decoder, tok_args


def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be power of two"
    return emax


def _sample_step_excluding_dmin(device: torch.device, B: int, T: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    # step_idx in [0, emax) i.e. excludes emax (d_min)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # K = 2^step_idx
    K = (1 << step_idx).to(torch.long)  # (B,T)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long)  # (B,T) in [0,K)
    tau = j_idx.to(torch.float32) / K.to(torch.float32)          # (B,T)
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")  # (B,T)
    tau_idx = j_idx * scale                                      # (B,T) <= k_max-1
    return tau, tau_idx


def dynamics_pretrain_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,                    # (B,T,Sz,Dz) packed clean targets
    actions: Optional[torch.Tensor],     # (B,T) or None
    act_mask: Optional[torch.Tensor],    # (A,) or None
    k_max: int,
    B_self: int,
    step: int,
    bootstrap_start: int,
    agent_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z1.device
    B, T = z1.shape[:2]
    assert 0 <= B_self < B
    B_emp = B - B_self
    emax = _emax_from_kmax(k_max)

    # action mask slices
    act_mask_full = act_mask
    act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

    # step idx: empirical rows are finest (d_min), self rows sample coarser
    step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max)
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
        step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_idx_full = step_idx_emp

    # sigma/tau per row/time
    sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # Corrupt inputs
    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    # Weights
    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # Main forward
    z1_hat_full, _ = dynamics(actions, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=act_mask_full, agent_tokens=agent_tokens)
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]

    flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)

    do_boot = (B_self > 0) and (step >= bootstrap_start)
    if do_boot:
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (torch.tensor(k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)

        z1_hat_half1, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=act_mask_self, agent_tokens=agent_tokens[B_emp:] if agent_tokens is not None else None)
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

        z1_hat_half2, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=act_mask_self, agent_tokens=agent_tokens[B_emp:] if agent_tokens is not None else None)
        b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

        vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()

        boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))  # (B_self,T)
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    # Combine losses
    loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

    aux = {
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_full.mean().detach(),
    }
    return loss, aux


def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns a schedule dict:
      K = number of integration steps (also grid size)
      e = log2(K)  (step_idx)
      scale = k_max // K
      tau_idx[i] = discrete signal index at step i
      tau[i] = i/K
      dt = 1/K
    """
    assert _is_pow2(k_max), "k_max must be power of two"
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires --eval_d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "eval_d must be 1/(power of two)"
        assert inv <= k_max, "eval_d must be >= 1/k_max"
        assert (k_max % inv) == 0, "k_max must be divisible by 1/eval_d"
        K = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    scale = k_max // K
    tau = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]  # allow final clean index
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K, schedule=schedule, d=1.0 / K)


@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,          # (B,t,n_spatial,d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,     # (B,T,A) aligned to frames or None
    act_mask: Optional[torch.Tensor] = None,    # (B,T,A) or (A,) or None
) -> torch.Tensor:
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    # start from noise at tau=0
    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

    emax = int(round(math.log2(k_max)))

    step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e  # only the sampled timestep uses the shortcut step

    signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    # broadcast (A,) -> (B,T,A) if needed (only if actions are present)
    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])

        signal_idxs_full[:, -1] = sig_i
        packed_seq = torch.cat([past_packed, z], dim=1)  # (B,t+1,...)

        actions_in = None if actions is None else actions[:, : t + 1]
        actmask_in = None if act_mask is None else act_mask[:, : t + 1]

        x1_hat_full, _ = dyn(
            actions_in,
            step_idxs_full,
            signal_idxs_full,
            packed_seq,
            act_mask=actmask_in,
            agent_tokens=None,
        )
        x1_hat = x1_hat_full[:, -1:, :, :]  # (B,1,n_spatial,d_spatial)

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    return z[:, 0]  # (B,n_spatial,d_spatial)


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,                  # (B,T,n_spatial,d_spatial)
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,     # (B,T,A) or None
    act_mask: Optional[torch.Tensor] = None,    # (B,T,A) or (A,) or None
) -> torch.Tensor:
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]

    for t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)  # (B,t,...)
        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=actions,
            act_mask=act_mask,
        )
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_packed_to_frames(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,     # (B,T',n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    z_btLd = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)  # (B,T',L,D_b)
    patches_btnd = decoder(z_btLd)                                     # (B,T',Np,Dp) in [0,1]
    frames = temporal_unpatchify(patches_btnd, H, W, C, patch)         # (B,T',C,H,W) in [0,1]
    return frames.clamp(0, 1)


@torch.no_grad()
def log_dynamics_eval_wandb(
    *,
    gt: torch.Tensor,          # (B,T,C,H,W) float [0,1]
    pred: torch.Tensor,        # (B,T,C,H,W) float [0,1]
    ctx_length: int,
    step: int,
    tag: str,
    max_items: int = 4,
    gap_px: int = 16,
):
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))

        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)

        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            left = y[..., :split]
            right = y[..., split:]
            gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)

    # Stack rows: GT / Pred
    panel = torch.cat([gt_t, pr_t], dim=2)   # (Bv,C,2H,TW+gap)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*2H,TW+gap)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {f"{tag}/viz": wandb.Image(big_hwc, caption=f"rows=GT/Pred | ctx={ctx_length} | T={T}")},
        step=step,
    )


@torch.no_grad()
def run_dynamics_eval(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    frames: torch.Tensor,            # (B,T,C,H,W) float [0,1]
    actions: Optional[torch.Tensor],    # (B,T,A) or None
    act_mask: Optional[torch.Tensor], # (A,) or None
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    k_max: int,
    ctx_length: int,
    horizon: int,
    sched: Dict[str, Any],
    max_items: int,
    step: int,
):
    dyn_was_training = dyn.training
    dyn.eval()

    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]

    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)  # (B,T_eval,L,D_b)
    assert z_btLd.shape[2] % packing_factor == 0
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)  # (B,T_eval,Sz,Dz)

    actions_eval = None if actions is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask

    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=k_max,
        sched=sched,
        actions=actions_eval,
        act_mask=act_mask_eval,
    )

    pred_frames = decode_packed_to_frames(
        decoder,
        z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch,
        packing_factor=packing_factor,
    )

    # floor baseline: repeat last context frame over horizon
    floor = frames_eval.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames_eval[:, ctx_length - 1:ctx_length].expand(-1, horizon, -1, -1, -1)

    # metric on horizon only
    gt_h    = frames_eval[:, ctx_length:ctx_length + horizon]         # (B,Hz,C,H,W)
    pred_h  = pred_frames[:, ctx_length:ctx_length + horizon]
    floor_h = floor[:, ctx_length:ctx_length + horizon]

    mse_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()

    psnr_pred  = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    mse_ratio = mse_pred / mse_floor.clamp_min(1e-12)     # <1 is better
    psnr_gain = psnr_pred - psnr_floor                    # >0 is better

    # per-timestep horizon MSE: log first/mid/last
    # per_t: (Hz,)
    per_t_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean(dim=(0,2,3,4))
    per_t_floor = (floor_h.float() - gt_h.float()).pow(2).mean(dim=(0,2,3,4))

    if horizon > 0:
        i0 = 0
        im = (horizon - 1) // 2
        i1 = horizon - 1

        wandb.log(
            {
                "eval/mse_pred": float(mse_pred.item()),
                "eval/mse_floor": float(mse_floor.item()),
                "eval/mse_ratio_pred_over_floor": float(mse_ratio.item()),

                "eval/psnr_pred": float(psnr_pred.item()),
                "eval/psnr_floor": float(psnr_floor.item()),
                "eval/psnr_gain_over_floor_db": float(psnr_gain.item()),

                # 1-indexed step labels in the horizon
                "eval/mse_pred_t1": float(per_t_pred[i0].item()),
                "eval/mse_pred_tmid": float(per_t_pred[im].item()),
                "eval/mse_pred_tend": float(per_t_pred[i1].item()),

                "eval/mse_floor_t1": float(per_t_floor[i0].item()),
                "eval/mse_floor_tmid": float(per_t_floor[im].item()),
                "eval/mse_floor_tend": float(per_t_floor[i1].item()),
            },
            step=step,
        )

    log_dynamics_eval_wandb(
        gt=frames_eval,
        pred=pred_frames,
        ctx_length=ctx_length,
        step=step,
        tag="eval",
        max_items=max_items,
    )

    if dyn_was_training:
        dyn.train()


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    # Dataset and DataLoader
    if args.use_actions:
        from wm_dataset import WMDataset, collate_batch
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
    else:
        dataset = ShardedFrameDataset(
            outdirs=args.frame_dirs,
            tasks=TASK_SET,
            seq_len=args.seq_len,
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
        )

    # Load frozen tokenizer
    tok_override = {}
    if args.H is not None: tok_override["H"] = args.H
    if args.W is not None: tok_override["W"] = args.W
    if args.C is not None: tok_override["C"] = args.C
    if args.patch is not None: tok_override["patch"] = args.patch

    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device, override=tok_override
    )

    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))

    assert H % patch == 0 and W % patch == 0
    assert n_latents % args.packing_factor == 0
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Build dynamics model
    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode=args.space_mode,
    ).to(device)

    if is_rank0():
        print(dyn)
        param_count = sum(p.numel() for p in dyn.parameters() if p.requires_grad)
        print(f"Learnable parameters (dynamics): {param_count:,}")
        print(f"[tokenizer] H={H} W={W} C={C} patch={patch} n_lat={n_latents} d_b={d_bottleneck} packing={args.packing_factor}")

    if args.compile:
        dyn = torch.compile(dyn)

    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # Optimizer and scaler
    opt = torch.optim.AdamW(
        dyn.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # Initialize wandb
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online",
            config=vars(args),
        )

    # Resume from checkpoint
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), dyn_model=dyn, opt=opt, scaler=scaler)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # Training loop
    dyn.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.max_steps:
                    break

                if args.use_actions:
                    obs_u8 = batch["obs"].to(device, non_blocking=True)          # (B,T+1,3,H,W) uint8
                    act    = batch["act"].to(device, non_blocking=True)          # (B,T,16) float
                    mask   = batch["act_mask"].to(device, non_blocking=True)     # (B,T,16) float (optional but good)

                    act = act.clamp(-1, 1) * mask

                    # Keep obs[0..T-1], align action[t] as action that produced obs[t]
                    frames = obs_u8[:, :-1].float() / 255.0                      # (B,T,3,H,W)
                    actions = torch.zeros_like(act)
                    actions[:, 1:] = act[:, :-1]
                    act_mask = torch.zeros_like(mask)
                    act_mask[:, 1:] = mask[:, :-1]
                else:
                    frames = batch.to(device, non_blocking=True)                 # (B,T,3,H,W)
                    actions = None
                    act_mask = None

                # Safeguard: convert to [0, 1] if dataset returns uint8
                if frames.dtype == torch.uint8:
                    frames = frames.float() / 255.0

                # Frozen encoder -> packed spatial tokens z1
                with torch.no_grad():
                    patches = temporal_patchify(frames, patch)  # (B,T,Np,Dp)
                    z_btLd, _ = encoder(patches)                # (B,T,n_latents,d_b)
                    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)  # (B,T,Sz,Dz)

                if actions is not None:
                    actions = actions.to(device, non_blocking=True)

                B = z1.shape[0]
                B_self = int(round(args.self_fraction * B))
                B_self = max(0, min(B - 1, B_self))

                with autocast(device_type="cuda", enabled=use_amp):
                    loss, aux = dynamics_pretrain_loss(
                        dyn.module if hasattr(dyn, "module") else dyn,
                        z1=z1,
                        actions=actions,
                        act_mask=act_mask,
                        k_max=args.k_max,
                        B_self=B_self,
                        step=step,
                        bootstrap_start=args.bootstrap_start,
                        agent_tokens=None,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss}")

                loss_to_backprop = loss / grad_accum
                scaler.scale(loss_to_backprop).backward()

                do_step = ((step + 1) % grad_accum == 0)
                if do_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            (dyn.module if hasattr(dyn, "module") else dyn).parameters(),
                            max_norm=args.grad_clip,
                        )

                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # Evaluation / visualization
                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0):

                    # Evaluate on a small slice of the current batch
                    B_eval = min(frames.shape[0], args.eval_batch_size)
                    frames_eval = frames[:B_eval]

                    if args.use_actions:
                        actions_eval = actions[:B_eval]
                        act_mask_eval = act_mask[:B_eval]
                    else:
                        actions_eval = None
                        act_mask_eval = None

                    sched = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

                    run_dynamics_eval(
                        encoder=encoder,
                        decoder=decoder,
                        dyn=(dyn.module if hasattr(dyn, "module") else dyn),
                        frames=frames_eval,
                        actions=actions_eval,
                        act_mask=act_mask_eval,
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=args.packing_factor,
                        k_max=args.k_max,
                        ctx_length=args.eval_ctx,
                        horizon=args.eval_horizon,
                        sched=sched,
                        max_items=args.eval_max_items,
                        step=step,
                    )

                # Logging
                if is_rank0() and (step % args.log_every == 0):
                    
                    # Action shuffle loss ratio
                    if actions is not None:
                        with torch.no_grad():
                            loss_real, _ = dynamics_pretrain_loss(
                                dyn.module if hasattr(dyn, "module") else dyn,
                                z1=z1,
                                actions=actions,
                                act_mask=act_mask,
                                k_max=args.k_max,
                                B_self=B_self,
                                step=step,
                                bootstrap_start=args.bootstrap_start,
                                agent_tokens=None,
                            )
                            perm = torch.randperm(actions.shape[0], device=actions.device)
                            loss_shuffled, _ = dynamics_pretrain_loss(
                                dyn.module if hasattr(dyn, "module") else dyn,
                                z1=z1,
                                actions=actions[perm],
                                act_mask=act_mask,
                                k_max=args.k_max,
                                B_self=B_self,
                                step=step,
                                bootstrap_start=args.bootstrap_start,
                                agent_tokens=None,
                            )
                        action_shuffle_loss_ratio = loss_shuffled / loss
                    else:
                        action_shuffle_loss_ratio = torch.tensor(0., device=device)

                    # Log to wandb
                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/flow_mse": float(aux["flow_mse"].item()),
                            "loss/bootstrap_mse": float(aux["bootstrap_mse"].item()),
                            "loss/loss_emp": float(aux["loss_emp"].item()),
                            "loss/loss_self": float(aux["loss_self"].item()),
                            "stats/action_shuffle_loss_ratio": float(action_shuffle_loss_ratio.item()),
                            "stats/sigma_mean": float(aux["sigma_mean"].item()),
                            "stats/B_self": float(B_self),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        },
                        step=step,
                    )

                    # Log to console
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| flow_mse={aux['flow_mse'].item():.6f} "
                        f"| boot_mse={aux['bootstrap_mse'].item():.6f} "
                        f"| sigma={aux['sigma_mean'].item():.3f} | B_self={B_self}"
                    )

                # Checkpointing
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and do_step:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, dyn_model=dyn, opt=opt, scaler=scaler, args=args)
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, dyn_model=dyn, opt=opt, scaler=scaler, args=args)

                step += 1

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data (if using multiple datasets, make sure they align in order)
    p.add_argument("--data_dirs", type=str, nargs="+", default=[   # paths to raw data
        "/<path>/expert",
        "/<path>/mixed-small",
        "/<path>/mixed-large",
    ])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=[  # paths to preprocessed frames
        "/<path>/expert-shards",
        "/<path>/mixed-small-shards",
        "/<path>/mixed-large-shards",
    ])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")  # task metadata
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)

    # tokenizer restore
    p.add_argument("--tokenizer_ckpt", type=str, default="./logs/tokenizer_ckpts/latest.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    # dynamics arch
    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)

    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)
    p.add_argument("--space_mode", type=str, default="wm_agent_isolated", choices=["wm_agent_isolated", "wm_agent"])

    # shortcut / schedule
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--bootstrap_start", type=int, default=5_000)
    p.add_argument("--self_fraction", type=float, default=0.25)

    # actions
    p.add_argument("--use_actions", action="store_true")

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # eval / viz
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_items", type=int, default=4)
    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)
    p.add_argument("--eval_schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.25)

    # logging
    p.add_argument("--log_every", type=int, default=200)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-dynamics")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/dynamics_ckpts")
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
