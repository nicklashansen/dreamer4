#!/usr/bin/env python3
"""Level 2: In-world-model rollout evaluation for the RL agent.

Loads pre-trained tokenizer + dynamics, plus trained agent heads (policy,
reward, value), and evaluates the agent by rolling out inside the world model.

Usage:
    python dreamer4/eval_agent.py \
        --tokenizer_ckpt logs/tokenizer.pt \
        --dynamics_ckpt logs/dynamics.pt \
        --agent_ckpt logs/agent.pt \
        --data_dir /public/dreamer4/expert \
        --frames_dir /public/dreamer4/expert-shards \
        --tasks walker-walk cheetah-run \
        --horizon 50 \
        --num_episodes 5

If --agent_ckpt is not provided, runs with random actions (baseline).
"""
import argparse
import os
import sys
import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from model import (
    Encoder, Decoder, Tokenizer, Dynamics,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)
from interactive import (
    load_tokenizer_from_ckpt,
    load_dynamics_from_ckpt,
    make_tau_schedule,
    find_episode_starts,
    load_frame_from_shards,
    load_task_action_dim,
    _as_2d_packed,
)
from agent import PolicyHead, RewardHead, ValueHead
from task_set import TASK_SET

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_agent_heads(
    ckpt_path: str,
    device: torch.device,
    d_model: int = 512,
) -> Tuple[PolicyHead, RewardHead, Optional[ValueHead]]:
    """Load trained agent heads from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Determine action_dim from checkpoint
    action_dim = ckpt.get("action_dim", 16)
    mtp_length = ckpt.get("mtp_length", 8)

    policy = PolicyHead(d_model=d_model, action_dim=action_dim, mtp_length=mtp_length).to(device)
    reward = RewardHead(d_model=d_model, mtp_length=mtp_length).to(device)

    policy.load_state_dict(ckpt["policy"])
    reward.load_state_dict(ckpt["reward"])
    policy.eval()
    reward.eval()

    value = None
    if "value" in ckpt:
        value = ValueHead(d_model=d_model).to(device)
        value.load_state_dict(ckpt["value"])
        value.eval()

    return policy, reward, value


@torch.inference_mode()
def encode_frame(
    encoder: Encoder,
    frame: torch.Tensor,
    *,
    n_spatial: int,
    packing_factor: int,
    patch: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a single frame to packed spatial latent.

    Args:
        frame: (3, H, W) float [0,1]

    Returns:
        z_packed: (n_spatial, d_spatial)
    """
    C, H, W = frame.shape
    patches = temporal_patchify(frame.view(1, 1, C, H, W), patch)
    z_btLd, _ = encoder(patches.to(device))
    z_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)
    return z_packed[0, 0]  # (n_spatial, d_spatial)


@torch.inference_mode()
def decode_frame(
    decoder: Decoder,
    z_packed: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    d_bottleneck: int,
) -> torch.Tensor:
    """Decode packed spatial latent to frame. Returns (C, H, W) float [0,1]."""
    z2 = _as_2d_packed(z_packed).unsqueeze(0).unsqueeze(0).float()  # (1,1,n_spatial,d_spatial)
    z_btLd = unpack_spatial_to_bottleneck(z2, k=packing_factor)
    patches = decoder(z_btLd)  # decoder is fp32
    frames = temporal_unpatchify(patches, H, W, C, patch)
    return frames[0, 0].clamp(0, 1)


@torch.inference_mode()
def sample_one_step_with_policy(
    dyn: Dynamics,
    policy: Optional[PolicyHead],
    reward_head: Optional[RewardHead],
    *,
    past_packed: torch.Tensor,       # (1, t, n_spatial, d_spatial)
    past_actions: torch.Tensor,      # (1, t+1, 16) — first action is zeros
    k_max: int,
    sched: dict,
    act_mask: torch.Tensor,          # (16,)
    action_dim: int,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Roll forward one step: sample action from policy, denoise next latent.

    Returns:
        z_next: (n_spatial, d_spatial)
        action: (16,)
        reward_pred: float
        value_pred: float (0.0 if no value head)
    """
    device = past_packed.device
    dtype = past_packed.dtype
    B = 1
    t = past_packed.shape[1]
    n_spatial = past_packed.shape[2]
    d_spatial = past_packed.shape[3]

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    # We need to first get h_t to sample the action, then do the denoising loop.
    # Run one forward pass with noise=0 just to get h_t for the policy.
    # Then use that action in the full denoising loop.

    # Get agent token output from a single forward pass at current context
    emax = int(round(math.log2(k_max)))
    step_idxs = torch.full((B, t), emax, device=device, dtype=torch.long)
    signal_idxs = torch.full((B, t), k_max - 1, device=device, dtype=torch.long)

    act_mask_bt = act_mask.view(1, 1, -1).expand(B, t, -1)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        _, h_t = dyn(
            past_actions[:, :t],
            step_idxs,
            signal_idxs,
            past_packed,
            act_mask=act_mask_bt,
        )

    # Sample action from policy (or random)
    if policy is not None and h_t is not None:
        h = h_t[:, -1:, 0, :]  # (1, 1, d_model) — last timestep, agent token
        action = policy.sample(h, step=0)[0, 0]  # (action_dim,)
        # Pad to 16 dims
        full_action = torch.zeros(16, device=device, dtype=torch.float32)
        full_action[:action_dim] = action[:action_dim]
    else:
        full_action = torch.randn(16, device=device, dtype=torch.float32).clamp(-1, 1) * act_mask

    # Build actions tensor for denoising: past_actions + new action
    actions_full = torch.cat([past_actions, full_action.view(1, 1, 16)], dim=1)  # (1, t+1, 16)
    act_mask_full = act_mask.view(1, 1, -1).expand(B, t + 1, -1)

    # Denoising loop for next latent
    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

    step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e
    signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])
        signal_idxs_full[:, -1] = sig_i

        packed_seq = torch.cat([past_packed, z], dim=1)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            x1_hat_full, h_t_full = dyn(
                actions_full,
                step_idxs_full,
                signal_idxs_full,
                packed_seq,
                act_mask=act_mask_full,
            )

        x1_hat = x1_hat_full[:, -1:, :, :]
        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    # Get reward/value predictions from last h_t
    reward_pred = 0.0
    if reward_head is not None and h_t_full is not None:
        h_last = h_t_full[:, -1:, 0, :]  # (1, 1, d_model)
        reward_pred = float(reward_head.predict(h_last, step=0)[0, 0].item())

    return z[0, 0], full_action, reward_pred


@torch.inference_mode()
def rollout_episode(
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    policy: Optional[PolicyHead],
    reward_head: Optional[RewardHead],
    *,
    initial_frame: torch.Tensor,   # (3, H, W) float [0,1]
    horizon: int,
    ctx_window: int,
    k_max: int,
    sched: dict,
    n_spatial: int,
    packing_factor: int,
    d_bottleneck: int,
    act_mask: torch.Tensor,
    action_dim: int,
    H: int, W: int, C: int, patch: int,
    device: torch.device,
    use_amp: bool = True,
    save_frames: bool = False,
) -> Dict:
    """Roll out the agent for `horizon` steps inside the world model.

    Returns dict with:
        imagined_return: float
        rewards: list of floats
        frames: list of (C,H,W) tensors if save_frames else empty
    """
    # Encode initial frame
    z0 = encode_frame(
        encoder, initial_frame.to(device),
        n_spatial=n_spatial, packing_factor=packing_factor,
        patch=patch, device=device,
    )
    z0 = z0.to(torch.float16 if use_amp else torch.float32)

    z_hist = [z0]
    a_hist = [torch.zeros(16, device=device, dtype=torch.float32)]
    rewards = []
    frames = []

    if save_frames:
        frames.append(initial_frame.cpu())

    for step in range(horizon):
        # Build context window
        g = len(z_hist)
        s = max(0, g - ctx_window)
        past = torch.stack(z_hist[s:g], dim=0).unsqueeze(0)  # (1, t, n_spatial, d_spatial)
        t = past.shape[1]

        actions_local = torch.zeros((1, t, 16), device=device, dtype=torch.float32)
        if t >= 2:
            actions_local[0, 1:t] = torch.stack(a_hist[s + 1:s + t], dim=0)

        z_next, action, reward_pred = sample_one_step_with_policy(
            dyn, policy, reward_head,
            past_packed=past,
            past_actions=actions_local,
            k_max=k_max,
            sched=sched,
            act_mask=act_mask,
            action_dim=action_dim,
            use_amp=use_amp,
        )

        z_hist.append(_as_2d_packed(z_next.detach()))
        a_hist.append(action.detach())
        rewards.append(reward_pred)

        if save_frames and (step % max(1, horizon // 10) == 0 or step == horizon - 1):
            fr = decode_frame(
                decoder, z_next,
                H=H, W=W, C=C, patch=patch,
                packing_factor=packing_factor,
                d_bottleneck=d_bottleneck,
            )
            frames.append(fr.cpu())

    imagined_return = sum(rewards)

    return {
        "imagined_return": imagined_return,
        "rewards": rewards,
        "frames": frames,
    }


def main():
    p = argparse.ArgumentParser(description="In-world-model RL agent evaluation")

    # Checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="logs/tokenizer.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="logs/dynamics.pt")
    p.add_argument("--agent_ckpt", type=str, default=None,
                   help="Path to agent heads checkpoint. If None, uses random actions.")

    # Data (for initial frames)
    p.add_argument("--data_dir", type=str, default="/public/dreamer4/expert")
    p.add_argument("--frames_dir", type=str, default="/public/dreamer4/expert-shards")
    p.add_argument("--tasks_json", type=str, default="tasks.json")
    p.add_argument("--shard_size", type=int, default=2048)

    # Tasks
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Tasks to evaluate. Default: all 30 TASK_SET tasks.")

    # Rollout
    p.add_argument("--horizon", type=int, default=50, help="Rollout length in steps")
    p.add_argument("--num_episodes", type=int, default=5, help="Episodes per task")
    p.add_argument("--ctx_window", type=int, default=24)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--amp", action="store_true", default=True)

    # Output
    p.add_argument("--save_frames", action="store_true", help="Save decoded frames for visualization")
    p.add_argument("--output_dir", type=str, default=None, help="Directory for frame outputs")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    print(f"Loading tokenizer from {args.tokenizer_ckpt}...")
    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    encoder, decoder = tok.encoder, tok.decoder
    H, W, C, patch = tok_info["H"], tok_info["W"], tok_info["C"], tok_info["patch"]
    d_bottleneck, n_latents = tok_info["d_bottleneck"], tok_info["n_latents"]

    print(f"Loading dynamics from {args.dynamics_ckpt}...")
    dyn, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device,
        d_bottleneck=d_bottleneck, n_latents=n_latents,
        packing_factor=args.packing_factor,
    )
    k_max = dyn_info["k_max"]
    n_spatial = dyn_info["n_spatial"]
    d_spatial = dyn_info["d_spatial"]

    sched = make_tau_schedule(
        k_max=k_max, schedule=args.schedule,
        d=args.eval_d if args.schedule == "shortcut" else None,
    )

    # --- Load agent heads (optional) ---
    policy, reward_head, value_head = None, None, None
    if args.agent_ckpt is not None:
        print(f"Loading agent heads from {args.agent_ckpt}...")
        policy, reward_head, value_head = load_agent_heads(
            args.agent_ckpt, device, d_model=dyn.d_model,
        )
    else:
        print("No agent checkpoint — using random actions (baseline).")

    # --- Determine tasks ---
    tasks = args.tasks if args.tasks else list(TASK_SET)

    # Filter to tasks that have data
    available = []
    for task in tasks:
        dp = os.path.join(args.data_dir, f"{task}.pt")
        if os.path.exists(dp):
            available.append(task)
        else:
            print(f"  Skipping {task}: no data at {dp}")
    tasks = available

    if not tasks:
        print("ERROR: No tasks with available data found!")
        sys.exit(1)

    print(f"\nEvaluating {len(tasks)} tasks, {args.num_episodes} episodes x {args.horizon} steps each")
    print(f"Schedule: {args.schedule} (d={args.eval_d}), ctx={args.ctx_window}, K={sched['K']}")
    print()

    # --- Evaluate ---
    all_results = {}
    for task in tasks:
        action_dim = load_task_action_dim(args.tasks_json, task, default_dim=16)
        act_mask = torch.zeros(16, dtype=torch.float32, device=device)
        if action_dim > 0:
            act_mask[:action_dim] = 1.0

        starts = find_episode_starts(args.data_dir, task)
        if not starts:
            print(f"  {task}: no episode starts found, skipping")
            continue

        episode_returns = []
        for ep in range(args.num_episodes):
            start_idx = starts[ep % len(starts)]
            frame0 = load_frame_from_shards(
                args.frames_dir, task, start_idx, shard_size=args.shard_size,
            )

            save_ep = args.save_frames and ep == 0
            result = rollout_episode(
                encoder, decoder, dyn, policy, reward_head,
                initial_frame=frame0,
                horizon=args.horizon,
                ctx_window=args.ctx_window,
                k_max=k_max,
                sched=sched,
                n_spatial=n_spatial,
                packing_factor=args.packing_factor,
                d_bottleneck=d_bottleneck,
                act_mask=act_mask,
                action_dim=action_dim,
                H=H, W=W, C=C, patch=patch,
                device=device,
                use_amp=args.amp,
                save_frames=save_ep,
            )
            episode_returns.append(result["imagined_return"])

            # Save frames if requested
            if save_ep and result["frames"] and args.output_dir:
                os.makedirs(os.path.join(args.output_dir, task), exist_ok=True)
                for i, fr in enumerate(result["frames"]):
                    path = os.path.join(args.output_dir, task, f"frame_{i:04d}.pt")
                    torch.save(fr, path)

        mean_ret = np.mean(episode_returns)
        std_ret = np.std(episode_returns)
        all_results[task] = {"mean": mean_ret, "std": std_ret, "episodes": episode_returns}

        print(f"  {task:30s}  return = {mean_ret:8.2f} ± {std_ret:6.2f}  (act_dim={action_dim})")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    returns = [r["mean"] for r in all_results.values()]
    if returns:
        print(f"  Mean across tasks:  {np.mean(returns):.2f}")
        print(f"  Median:             {np.median(returns):.2f}")
        print(f"  Best:  {max(all_results, key=lambda t: all_results[t]['mean']):30s} = {max(returns):.2f}")
        print(f"  Worst: {min(all_results, key=lambda t: all_results[t]['mean']):30s} = {min(returns):.2f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump({k: {"mean": v["mean"], "std": v["std"]} for k, v in all_results.items()}, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
