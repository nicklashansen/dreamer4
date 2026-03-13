# interactive.py
import os
import math
import json
import time
import glob
import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Set, Tuple

import numpy as np
import torch
from aiohttp import web, WSMsgType
from PIL import Image
import io

from task_set import TASK_SET
from model import (
    Encoder, Decoder, Tokenizer, Dynamics, TaskEmbedder,
    temporal_patchify, temporal_unpatchify,
    denoise_one_step,
)
from agent import PolicyHead, RewardHead, ValueHead

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    # (B,T,L,Db) -> (B,T,n_spatial,k*Db) with L == n_spatial*k
    B, T, L, Db = z_btLd.shape
    assert L == n_spatial * k, f"L={L} != n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k, Db).reshape(B, T, n_spatial, k * Db)


def unpack_spatial_to_bottleneck(z_packed: torch.Tensor, *, k: int, d_bottleneck: int) -> torch.Tensor:
    # (B,T,n_spatial,k*Db) -> (B,T,n_spatial*k,Db)
    B, T, n_spatial, Dz = z_packed.shape
    assert Dz == k * d_bottleneck, f"Dz={Dz} != k*Db={k*d_bottleneck}"
    return z_packed.view(B, T, n_spatial, k, d_bottleneck).reshape(B, T, n_spatial * k, d_bottleneck)


def _as_2d_packed(z: torch.Tensor) -> torch.Tensor:
    # ensure (n_spatial, d_spatial)
    if z.dim() == 2:
        return z
    if z.dim() == 3 and z.shape[0] == 1:
        return z[0]
    raise RuntimeError(f"Unexpected packed latent shape: {tuple(z.shape)}")


def _is_pow2_frac(x: float) -> bool:
    if x <= 0 or x > 1:
        return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0


def make_tau_schedule(*, k_max: int, schedule: str = "finest", d: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns:
      K: Euler steps
      e: log2(K) (rounded)
      dt: step size
      tau: [i/K]
      tau_idx: discrete indices on k_max grid
    """
    schedule = str(schedule)
    if schedule == "finest":
        K = int(k_max)
        dt = 1.0 / float(K)
    elif schedule == "shortcut":
        assert d is not None and _is_pow2_frac(float(d)), "shortcut requires d = 1/(power of two)"
        dt = float(d)
        K = int(round(1.0 / dt))
        if dt < 1.0 / float(k_max):
            raise ValueError(f"shortcut d={dt} is finer than finest 1/k_max={1.0/k_max}")
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    tau = [i / float(K) for i in range(K)]
    stride = k_max // K
    if stride <= 0:
        raise ValueError(f"k_max={k_max} must be >= K={K}")
    tau_idx = [i * stride for i in range(K)]
    return {"K": K, "e": e, "dt": dt, "tau": tau, "tau_idx": tau_idx}



def load_task_action_dim(tasks_json: str, task: str, *, default_dim: int = 16) -> int:
    try:
        with open(tasks_json, "r") as f:
            meta = json.load(f)
        if task in meta and "action_dim" in meta[task]:
            return int(meta[task]["action_dim"])
    except Exception:
        pass
    return int(default_dim)


def find_episode_starts(data_dir: str, task: str) -> List[int]:
    path = os.path.join(data_dir, f"{task}.pt")
    td = torch.load(path, map_location="cpu", weights_only=False)
    ep = td["episode"]
    if hasattr(ep, "detach"):
        ep = ep.detach().cpu()
    ep = ep.to(torch.int64)
    if ep.numel() == 0:
        return [0]
    starts = [0]
    diff = (ep[1:] != ep[:-1]).nonzero(as_tuple=False).flatten()
    if diff.numel() > 0:
        starts.extend((diff + 1).tolist())
    return starts


def load_frame_from_shards(frames_dir: str, task: str, index: int, *, shard_size: int = 2048) -> torch.Tensor:
    task_dir = os.path.join(frames_dir, task)
    shard_paths = sorted(glob.glob(os.path.join(task_dir, "*shard*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {task_dir}")

    shard_idx = int(index) // int(shard_size)
    off = int(index) % int(shard_size)
    if shard_idx >= len(shard_paths):
        raise IndexError(f"index={index} -> shard_idx={shard_idx} but only {len(shard_paths)} shards")

    td = torch.load(shard_paths[shard_idx], map_location="cpu", weights_only=False)
    frames = td["frames"]

    # normalize to (N,3,H,W) uint8
    if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2).contiguous()

    fr = frames[off]
    if fr.dtype != torch.uint8:
        frf = fr.to(torch.float32)
        mx = float(frf.max().item()) if frf.numel() > 0 else 0.0
        if mx > 1.5:
            fr = frf.clamp(0, 255).to(torch.uint8)
        else:
            fr = (frf.clamp(0, 1) * 255.0).to(torch.uint8)

    return fr.to(torch.float32) / 255.0


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    k0 = next(iter(d.keys()))
    v0 = d[k0]
    return isinstance(k0, str) and (torch.is_tensor(v0) or isinstance(v0, torch.nn.Parameter))


def _get_state_dict(ckpt: dict) -> dict:
    if _looks_like_state_dict(ckpt):
        sd = ckpt
    else:
        for k in ("dynamics", "dyn_model", "model", "dyn", "state_dict"):
            v = ckpt.get(k, None)
            if isinstance(v, dict):
                if "state_dict" in v and isinstance(v["state_dict"], dict) and _looks_like_state_dict(v["state_dict"]):
                    v = v["state_dict"]
                if _looks_like_state_dict(v):
                    sd = v
                    break
        else:
            raise KeyError(f"Could not find state dict in checkpoint keys={list(ckpt.keys())}")

    for pfx in ("module.", "dynamics.", "dyn."):
        sd = _strip_prefix(sd, pfx)
    return sd


def load_tokenizer_from_ckpt(tokenizer_ckpt: str, device: torch.device):
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    H = int(a.get("H", 128))
    W = int(a.get("W", 128))
    C = int(a.get("C", 3))
    patch = int(a.get("patch", 4))
    d_model = int(a.get("d_model", 256))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("depth", 8))
    n_latents = int(a.get("n_latents", 16))
    d_bottleneck = int(a.get("d_bottleneck", 32))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 1))

    assert H % patch == 0 and W % patch == 0
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=d_model,
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=n_heads,
        depth=depth,
        d_bottleneck=d_bottleneck,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=d_bottleneck,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
    )
    tok = Tokenizer(enc, dec).to(device)
    tok.load_state_dict(_get_state_dict(ckpt), strict=True)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    info = dict(H=H, W=W, C=C, patch=patch, n_latents=n_latents, d_bottleneck=d_bottleneck)
    return tok, info


def load_dynamics_from_ckpt(
    dynamics_ckpt: str,
    *,
    device: torch.device,
    d_bottleneck: int,
    n_latents: int,
    packing_factor: int,
):
    ckpt = torch.load(dynamics_ckpt, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    d_model = int(a.get("d_model_dyn", a.get("dyn_d_model", a.get("d_model", 256))))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("dyn_depth", a.get("depth", 8)))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 4))
    k_max = int(a.get("k_max", 8))
    n_register = int(a.get("n_register", 4))
    n_agent = int(a.get("n_agent", 0))
    space_mode = str(a.get("space_mode", a.get("agent_space_mode", "wm_agent_isolated")))

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
    dyn.load_state_dict(_get_state_dict(ckpt), strict=True)
    dyn.eval()
    return dyn, {"k_max": k_max, "n_spatial": n_spatial, "d_spatial": d_spatial}


@torch.inference_mode()
def decode_single_packed_frame(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,   # (n_spatial,d_spatial) or (1,n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    d_bottleneck: int,
) -> torch.Tensor:
    z2 = _as_2d_packed(z_packed)
    z_bt = z2.unsqueeze(0).unsqueeze(0)  # (1,1,n_spatial,d_spatial)
    z_btLd = unpack_spatial_to_bottleneck(z_bt, k=packing_factor, d_bottleneck=d_bottleneck)
    patches = decoder(z_btLd)  # (1,1,Np,Dp)
    frames = temporal_unpatchify(patches, H, W, C, patch)  # (1,1,C,H,W)
    return frames[0, 0].clamp(0, 1)


def frame_to_jpeg_bytes(frame_chw_01: torch.Tensor, *, quality: int = 85) -> bytes:
    fr_u8 = (frame_chw_01.clamp(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
    hwc = np.transpose(fr_u8, (1, 2, 0))
    im = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def build_action_from_keys(keys_down: Set[str], *, selected_dim: int, act_dim: int, A: int = 16) -> torch.Tensor:
    a = torch.zeros(A, dtype=torch.float32)
    if act_dim <= 0:
        return a
    v = 0.0
    if ("ArrowUp" in keys_down) and ("ArrowDown" not in keys_down):
        v = +1.0
    elif ("ArrowDown" in keys_down) and ("ArrowUp" not in keys_down):
        v = -1.0
    if 0 <= selected_dim < act_dim:
        a[selected_dim] = v
    return a


def load_agent_heads(
    ckpt_path: str,
    device: torch.device,
    d_model: int = 512,
    n_agent: int = 1,
) -> Tuple[PolicyHead, Optional[RewardHead], Optional[TaskEmbedder]]:
    """Load trained policy (and optionally reward/task_embedder) from checkpoint.

    Supports both Phase 2 (dynamics+task_emb+policy+reward) and
    Phase 3 (policy+value) checkpoint formats.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    action_dim = ckpt.get("action_dim")
    mtp_length = ckpt.get("mtp_length")

    # Phase 3 checkpoints don't store these; load from the Phase 2 checkpoint they reference
    if action_dim is None or mtp_length is None:
        p2_path = (ckpt.get("args") or {}).get("phase2_ckpt")
        if p2_path and os.path.isfile(p2_path):
            p2 = torch.load(p2_path, map_location="cpu")
            action_dim = action_dim or int(p2.get("action_dim", 16))
            mtp_length = mtp_length or int(p2.get("mtp_length", 8))
            del p2
    action_dim = int(action_dim or 16)
    mtp_length = int(mtp_length or 8)

    # Infer hidden dim from gate_proj weight: shape is (2*hidden, input_dim)
    gate_w = ckpt["policy"].get("mlp.gate_projs.0.weight")
    hidden = gate_w.shape[0] // 2 if gate_w is not None else 512

    policy = PolicyHead(d_model=d_model, action_dim=action_dim, hidden=hidden, mtp_length=mtp_length).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    reward = None
    if "reward" in ckpt:
        rw_gate = ckpt["reward"].get("mlp.gate_projs.0.weight")
        rw_hidden = rw_gate.shape[0] // 2 if rw_gate is not None else 512
        reward = RewardHead(d_model=d_model, hidden=rw_hidden, mtp_length=mtp_length).to(device)
        reward.load_state_dict(ckpt["reward"])
        reward.eval()
        for p in reward.parameters():
            p.requires_grad_(False)

    task_emb = None
    if "task_embedder" in ckpt:
        task_emb = TaskEmbedder(d_model=d_model, n_agent=n_agent, use_ids=True, n_tasks=128).to(device)
        task_emb.load_state_dict(ckpt["task_embedder"])
        task_emb.eval()
        for p in task_emb.parameters():
            p.requires_grad_(False)

    return policy, reward, task_emb


def build_task_emb_map(*ckpt_paths: str) -> Dict[str, int]:
    """Reconstruct task name -> emb_id mapping from checkpoint training args.

    Replicates WMDataset's discovery logic: sorted glob across data_dirs, dedup,
    optional tasks filter. Uses the checkpoint that trained the task_embedder
    (i.e. the one that contains a 'task_embedder' key), since that's the
    checkpoint whose task ordering determines the embedding indices.
    """
    # Prioritize checkpoints that contain the task_embedder (Phase 2)
    ordered = []
    rest = []
    for path in ckpt_paths:
        if not path or not os.path.isfile(path):
            continue
        ckpt = torch.load(path, map_location="cpu")
        if "task_embedder" in ckpt:
            ordered.append(ckpt)
        else:
            rest.append(ckpt)
    ordered.extend(rest)

    for ckpt in ordered:
        a = ckpt.get("args") or {}
        data_dirs = a.get("data_dirs")
        if not data_dirs:
            continue
        tasks_filter = a.get("tasks")  # None means all tasks

        # Discover tasks in same order as WMDataset
        found = []
        seen: Set[str] = set()
        for dd in data_dirs:
            for p in sorted(glob.glob(os.path.join(dd, "*.pt"))):
                t = os.path.splitext(os.path.basename(p))[0]
                if t not in seen:
                    seen.add(t)
                    found.append(t)

        # Apply filter if present
        if tasks_filter is not None:
            found = [t for t in found if t in set(tasks_filter)]

        return {t: i for i, t in enumerate(found)}
    return {}


def load_html(path: Optional[str], *, fallback: str = "") -> str:
    if not path:
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback


@dataclass
class SessionState:
    task: str
    keys_down: Set[str]
    selected_dim: int
    paused: bool
    reset_requested: bool
    step: int
    last_action_val: float

    z0_packed: torch.Tensor
    z_hist: List[torch.Tensor]
    a_hist: List[torch.Tensor]

    act_dim: int
    act_mask_1d: torch.Tensor
    action_beta: float
    a_smooth: torch.Tensor   # (16,)
    ctx_window: int
    fps: float

    autopilot: bool

    current_reward: float
    cumulative_reward: float

    cached_frame_id: int
    cached_jpeg: Optional[bytes]


class InteractiveServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(args.amp) and (self.device.type == "cuda")
        self.infer_lock = asyncio.Lock()

        # HTML
        self.html = load_html(args.html, fallback="<html><body>missing html</body></html>")

        # task
        self.tasks = list(TASK_SET)
        self.initial_task = args.task if args.task in self.tasks else self.tasks[0]

        # tokenizer
        tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, self.device)
        self.encoder: Encoder = tok.encoder
        self.decoder: Decoder = tok.decoder

        self.d_bottleneck = int(tok_info["d_bottleneck"])
        self.n_latents = int(tok_info["n_latents"])
        self.H = int(tok_info["H"])
        self.W = int(tok_info["W"])
        self.C = int(tok_info["C"])
        self.patch = int(tok_info["patch"])

        # dynamics
        self.dyn, dyn_info = load_dynamics_from_ckpt(
            args.dynamics_ckpt,
            device=self.device,
            d_bottleneck=self.d_bottleneck,
            n_latents=self.n_latents,
            packing_factor=args.packing_factor,
        )
        self.k_max = int(dyn_info["k_max"])
        self.n_spatial = int(dyn_info["n_spatial"])
        self.d_spatial = int(dyn_info["d_spatial"])

        self.sched = make_tau_schedule(
            k_max=self.k_max,
            schedule=args.schedule,
            d=(args.eval_d if args.schedule == "shortcut" else None),
        )

        # action mask/dims
        act_dim = load_task_action_dim(args.tasks_json, self.initial_task, default_dim=16)
        act_dim = max(0, min(16, int(act_dim)))
        self.act_dim = act_dim

        mask = torch.zeros(16, dtype=torch.float32)
        if act_dim > 0:
            mask[:act_dim] = 1.0
        self.act_mask_1d = mask.to(self.device)

        # policy (optional)
        self.policy: Optional[PolicyHead] = None
        self.reward_head: Optional[RewardHead] = None
        self.task_embedder: Optional[TaskEmbedder] = None
        self.task_emb_ids: Dict[str, int] = {}  # task name -> emb_id
        self.policy_action_dim: int = 16
        if args.agent_ckpt and os.path.isfile(args.agent_ckpt):
            self.policy, self.reward_head, self.task_embedder = load_agent_heads(
                args.agent_ckpt, self.device,
                d_model=self.dyn.d_model,
                n_agent=self.dyn.n_agent,
            )
            self.policy_action_dim = self.policy.action_dim
            # Phase 3 checkpoints don't include task_embedder; fall back to dynamics checkpoint
            if self.task_embedder is None and self.dyn.n_agent > 0:
                _, _, self.task_embedder = load_agent_heads(
                    args.dynamics_ckpt, self.device,
                    d_model=self.dyn.d_model,
                    n_agent=self.dyn.n_agent,
                )
            self.task_emb_ids = build_task_emb_map(args.agent_ckpt, args.dynamics_ckpt)
            te_str = "yes" if self.task_embedder is not None else "no"
            print(f"[web] loaded policy from {args.agent_ckpt} (action_dim={self.policy_action_dim}, task_emb={te_str}, {len(self.task_emb_ids)} tasks)")

        # choose episode start + encode initial latent
        self.start_idx = self._choose_start_idx(self.initial_task)
        self.z0_packed = self._encode_initial_latent(self.initial_task, self.start_idx)
        self.act_dim, self.act_mask_1d = self._compute_act_mask(self.initial_task)

    @torch.inference_mode()
    def _encode_initial_latent(self, task: str, start_idx: int) -> torch.Tensor:
        frame0 = load_frame_from_shards(self.args.frames_dir, task, start_idx, shard_size=self.args.shard_size).to(self.device)
        patches0 = temporal_patchify(frame0.view(1, 1, self.C, self.H, self.W), self.patch)
        z0_btLd, _ = self.encoder(patches0)
        z0_packed = pack_bottleneck_to_spatial(z0_btLd, n_spatial=self.n_spatial, k=self.args.packing_factor)[0, 0]
        z0_packed = z0_packed.to(torch.float32)
        return z0_packed.detach()

    def _compute_act_mask(self, task: str) -> Tuple[int, torch.Tensor]:
        act_dim = load_task_action_dim(self.args.tasks_json, task, default_dim=16)
        act_dim = max(0, min(16, int(act_dim)))
        mask = torch.zeros(16, dtype=torch.float32)
        if act_dim > 0:
            mask[:act_dim] = 1.0
        return act_dim, mask.to(self.device)

    def _choose_start_idx(self, task: str) -> int:
        starts = find_episode_starts(self.args.data_dir, task)
        return int(np.random.choice(starts)) if starts else 0

    def new_session(self) -> SessionState:
        task = self.initial_task
        act_dim, act_mask = self._compute_act_mask(task)
        start_idx = self._choose_start_idx(task)
        z0 = _as_2d_packed(self._encode_initial_latent(task, start_idx))
        beta = float(self.args.action_smooth_beta)
        a0 = torch.zeros(16, device=self.device, dtype=torch.float32)

        return SessionState(
            task=task,
            keys_down=set(),
            selected_dim=0,
            paused=False,
            reset_requested=False,
            step=0,
            last_action_val=0.0,
            z0_packed=z0,
            z_hist=[z0],
            a_hist=[torch.zeros(16, device=self.device, dtype=torch.float32)],
            act_dim=act_dim,
            act_mask_1d=act_mask,
            action_beta=beta,
            a_smooth=a0,
            ctx_window=int(self.args.ctx_window),
            fps=float(self.args.fps),
            autopilot=self.policy is not None,
            current_reward=0.0,
            cumulative_reward=0.0,
            cached_frame_id=-1,
            cached_jpeg=None,
        )

    def _reset_session(self, st: SessionState):
        starts = find_episode_starts(self.args.data_dir, st.task)
        self.start_idx = int(np.random.choice(starts))
        st.z0_packed = self._encode_initial_latent(st.task, self.start_idx)

        z0 = _as_2d_packed(st.z0_packed.detach())
        st.z_hist = [z0]
        st.a_hist = [torch.zeros(16, device=self.device, dtype=torch.float32)]
        st.a_smooth = torch.zeros(16, device=self.device, dtype=torch.float32)

        st.keys_down.clear()
        st.selected_dim = 0
        st.paused = False
        st.reset_requested = False
        st.step = 0
        st.last_action_val = 0.0
        st.current_reward = 0.0
        st.cumulative_reward = 0.0
        st.cached_frame_id = -1
        st.cached_jpeg = None

    def _switch_task_sync(self, st: SessionState, new_task: str):
        if new_task not in self.tasks:
            return

        st.task = new_task
        st.act_dim, st.act_mask_1d = self._compute_act_mask(new_task)

        start_idx = self._choose_start_idx(new_task)
        st.z0_packed = _as_2d_packed(self._encode_initial_latent(new_task, start_idx))

        st.z_hist = [st.z0_packed]
        st.a_hist = [torch.zeros(16, device=self.device, dtype=torch.float32)]
        st.a_smooth = torch.zeros(16, device=self.device, dtype=torch.float32)
        st.keys_down.clear()
        st.selected_dim = 0
        st.reset_requested = False
        st.step = 0
        st.last_action_val = 0.0
        st.current_reward = 0.0
        st.cumulative_reward = 0.0

        st.cached_frame_id = -1
        st.cached_jpeg = None

    def _build_local_window(self, st: SessionState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          past: (1,t,n_spatial,d_spatial)
          actions_local: (1,t+1,16) with actions_local[:,0]=0
          actmask_local: (1,t+1,16)
        """
        g = len(st.z_hist)  # next frame index
        s = max(0, g - int(st.ctx_window))

        past_list = st.z_hist[s:g]  # list of (n_spatial,d_spatial)
        if len(past_list) == 0:
            past = torch.empty((1, 0, self.n_spatial, self.d_spatial),
                               device=self.device, dtype=st.z_hist[-1].dtype)
        else:
            past = torch.stack(past_list, dim=0).unsqueeze(0)  # (1,t,...)
        t = past.shape[1]

        actions_local = torch.zeros((1, t + 1, 16), device=self.device, dtype=torch.float32)
        if t >= 1:
            # a_hist has an entry for each generated step
            actions_local[0, 1:t + 1] = torch.stack(st.a_hist[s + 1: s + t + 1], dim=0)

        actmask_local = st.act_mask_1d.view(1, 1, 16).expand(1, t + 1, 16).contiguous()
        return past, actions_local, actmask_local

    @torch.inference_mode()
    def _make_agent_tokens(self, T: int, task: str) -> Optional[torch.Tensor]:
        """Generate agent tokens from task embedder if available."""
        if self.task_embedder is None:
            return None
        idx = self.task_emb_ids.get(task, 0)
        emb_id = torch.tensor([idx], device=self.device, dtype=torch.long)
        return self.task_embedder(emb_id, B=1, T=T)  # (1, T, n_agent, d_model)

    @torch.inference_mode()
    def _get_h_t_for_policy(self, st: SessionState) -> torch.Tensor:
        """Run a clean forward pass on the current context to get h_t for the policy."""
        import math as _math
        g = len(st.z_hist)
        s = max(0, g - int(st.ctx_window))
        past_list = st.z_hist[s:g]
        past = torch.stack(past_list, dim=0).unsqueeze(0)  # (1, t, n_spatial, d_spatial)
        t = past.shape[1]

        emax = int(round(_math.log2(self.k_max)))
        step_idxs = torch.full((1, t), emax, device=self.device, dtype=torch.long)
        signal_idxs = torch.full((1, t), self.k_max - 1, device=self.device, dtype=torch.long)

        actions_ctx = torch.zeros((1, t, 16), device=self.device, dtype=torch.float32)
        if t >= 2:
            actions_ctx[0, 1:t] = torch.stack(st.a_hist[s + 1: s + t], dim=0)
        act_mask_ctx = st.act_mask_1d.view(1, 1, 16).expand(1, t, 16)

        agent_tokens = self._make_agent_tokens(t, st.task)

        with torch.autocast(device_type=self.device.type, enabled=(self.use_amp and self.device.type == "cuda")):
            _, h_t_full = self.dyn(
                actions_ctx, step_idxs, signal_idxs, past,
                act_mask=act_mask_ctx,
                agent_tokens=agent_tokens,
            )
        # h_t_full: (1, t, n_agent, d_model) — take last timestep, first agent token
        return h_t_full[:, -1:, 0, :]  # (1, 1, d_model)

    def _render_step_sync(self, st: SessionState) -> Tuple[bytes, Dict[str, Any]]:
        """
        Runs at most one WM step (if not paused), then decodes the current frame.
        Called via asyncio.to_thread.
        """
        if st.reset_requested:
            self._reset_session(st)

        # --- Determine action ---
        if st.autopilot and self.policy is not None and not st.paused:
            # Policy-driven: get h_t and sample action
            h_t = self._get_h_t_for_policy(st)  # (1, 1, d_model)
            action_raw = self.policy.sample(h_t, step=0)[0, 0]  # (action_dim,)
            a = torch.zeros(16, device=self.device, dtype=torch.float32)
            a[:self.policy_action_dim] = action_raw[:self.policy_action_dim].clamp(-1, 1)
            a = (a * st.act_mask_1d).to(torch.float32)
            st.last_action_val = float(a[st.selected_dim].item()) if st.act_dim > 0 else 0.0
        else:
            # Manual keyboard control
            a_raw = build_action_from_keys(
                st.keys_down, selected_dim=st.selected_dim, act_dim=st.act_dim, A=16
            ).to(self.device)
            a_raw = (a_raw.clamp(-1, 1) * st.act_mask_1d).to(torch.float32)

            beta = float(st.action_beta)
            if beta > 0.0:
                beta = min(max(beta, 0.0), 0.999)
                st.a_smooth = (beta * st.a_smooth + (1.0 - beta) * a_raw).to(torch.float32)
                a = st.a_smooth
            else:
                a = a_raw
            st.last_action_val = float(a[st.selected_dim].item()) if st.act_dim > 0 else 0.0

        # --- Step the world model ---
        if not st.paused and st.act_dim >= 0:
            st.a_hist.append(a)

            past, actions_local, actmask_local = self._build_local_window(st)
            agent_tokens = self._make_agent_tokens(actions_local.shape[1], st.task)

            z_next, h_t_out = denoise_one_step(
                self.dyn,
                past_packed=past,
                k_max=self.k_max,
                sched=self.sched,
                actions=actions_local,
                act_mask=actmask_local,
                agent_tokens=agent_tokens,
                use_amp=self.use_amp,
            )
            st.z_hist.append(_as_2d_packed(z_next.detach()))
            st.step += 1

            # Predict reward for the current frame (zero extra cost — h_t already computed)
            if self.reward_head is not None and h_t_out is not None:
                h_rew = h_t_out[:, :1, :]  # (1, 1, d_model)
                r = float(self.reward_head.predict(h_rew, step=0).item())
                st.current_reward = r
                st.cumulative_reward += r

        frame_id = len(st.z_hist) - 1  # stable identifier for "current displayed frame"
        need_encode = (st.cached_jpeg is None) or (st.cached_frame_id != frame_id)

        jpeg: Optional[bytes] = None
        if need_encode:
            z_cur = st.z_hist[-1]
            fr = decode_single_packed_frame(
                self.decoder,
                z_packed=z_cur,
                H=self.H, W=self.W, C=self.C, patch=self.patch,
                packing_factor=self.args.packing_factor,
                d_bottleneck=self.d_bottleneck,
            )
            st.cached_jpeg = frame_to_jpeg_bytes(fr, quality=int(self.args.jpeg_quality))
            st.cached_frame_id = frame_id
            jpeg = st.cached_jpeg
        else:
            jpeg = None

        mode = "autopilot" if st.autopilot else "manual"
        rew_str = ""
        if self.reward_head is not None:
            rew_str = f" | r={st.current_reward:.3f} R={st.cumulative_reward:.1f}"
        status = {
            "type": "status",
            "task": st.task,
            "paused": bool(st.paused),
            "autopilot": bool(st.autopilot),
            "has_policy": self.policy is not None,
            "reward": st.current_reward,
            "cumulative_reward": st.cumulative_reward,
            "text": (
                f"step={st.step} | "
                f"dim={st.selected_dim}/{max(st.act_dim - 1, 0)} | "
                f"a={st.last_action_val:+.1f} | "
                f"{mode} | fps={st.fps:.1f}"
                f"{rew_str}"
            ),
        }
        return jpeg, status

    async def index(self, request: web.Request) -> web.Response:
        html = self.html
        html = html.replace("__TASK_SET__", json.dumps(self.tasks))
        html = html.replace("__INITIAL_TASK__", self.initial_task)
        html = html.replace("__HAS_POLICY__", "true" if self.policy is not None else "false")
        return web.Response(text=html, content_type="text/html")

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
        await ws.prepare(request)

        st = self.new_session()
        await ws.send_str(json.dumps({
            "type": "status",
            "task": st.task,
            "paused": bool(st.paused),
            "text": "connected",
        }))

        async def recv_loop():
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        return
                    continue

                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                t = data.get("type", "")

                if t == "keydown":
                    k = str(data.get("key", ""))

                    if k == "ArrowLeft" and st.act_dim > 0:
                        st.selected_dim = (st.selected_dim - 1) % st.act_dim
                    elif k == "ArrowRight" and st.act_dim > 0:
                        st.selected_dim = (st.selected_dim + 1) % st.act_dim
                    elif k == "Space":
                        st.paused = not st.paused
                    elif k in ("r", "R"):
                        st.reset_requested = True
                    elif k in ("a", "A"):
                        st.autopilot = not st.autopilot
                    elif k in ("q", "Q", "Escape"):
                        await ws.close()
                        return
                    else:
                        st.keys_down.add(k)

                elif t == "keyup":
                    k = str(data.get("key", ""))
                    st.keys_down.discard(k)

                elif t == "set_task":
                    new_task = str(data.get("task", ""))
                    async with self.infer_lock:
                        await asyncio.to_thread(self._switch_task_sync, st, new_task)

                elif t == "toggle_pause":
                    st.paused = not st.paused

                elif t == "reset":
                    st.reset_requested = True

                elif t == "toggle_autopilot":
                    st.autopilot = not st.autopilot

                elif t == "disconnect":
                    await ws.close()
                    return

        async def send_loop():
            import traceback as _tb
            dt = 1.0 / max(1e-6, float(st.fps))
            next_t = time.monotonic()

            while not ws.closed:
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                    continue
                next_t += dt

                try:
                    async with self.infer_lock:
                        jpeg, status = await asyncio.to_thread(self._render_step_sync, st)
                except Exception:
                    _tb.print_exc()
                    break

                if ws.closed:
                    break
                try:
                    await ws.send_str(json.dumps(status))
                    if jpeg is not None:
                        await ws.send_bytes(jpeg)
                except Exception:
                    break

        recv = asyncio.create_task(recv_loop())
        send = asyncio.create_task(send_loop())
        done, pending = await asyncio.wait({recv, send}, return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        return ws


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--task", type=str, default="finger-turn-hard")

    # data
    p.add_argument("--data_dir", type=str, default="/<path>/data")   # path to raw data
    p.add_argument("--frames_dir", type=str, default="/<path>/frames128")   # path to processed frames
    p.add_argument("--tasks_json", type=str, default="../tasks.json")   # task metadata
    p.add_argument("--shard_size", type=int, default=2048)

    # checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, default="./logs/tokenizer_ckpts/step_0085000.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="./logs/dynamics_ckpts/latest.pt")
    p.add_argument("--agent_ckpt", type=str, default=None,
                   help="Path to agent checkpoint (policy head). Enables autopilot mode.")

    # rollout
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--ctx_window", type=int, default=24)
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--action_smooth_beta", type=float, default=0.817)

    # web server
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--html", type=str, default="interactive.html")

    # misc
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    server = InteractiveServer(args)

    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/ws", server.ws_handler)

    print(f"[web] serving on http://{args.host}:{args.port}  (task={args.task})")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
