# wm_dataset.py
import os
import glob
import json
import bisect
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


class WMDataset(Dataset):
    """
    RGB-native world model dataset.

    Now supports multiple roots for demos/frames:
      data_dir:   str or list[str]
      frames_dir: str or list[str]

    If lists are provided, they are treated as paired roots. If one is length-1
    and the other is longer, the length-1 root is broadcast.
    """
    def __init__(
        self,
        data_dir: Union[str, Sequence[str]],
        frames_dir: Union[str, Sequence[str]],
        seq_len: int,
        img_size: int = 128,
        action_dim: int = 16,
        lang_dim: int = 512,
        shard_size: int = 2048,
        cache_mb: int = 2048,
        verbose: bool = True,
        tasks_json: str = "../tasks.json",
        tasks: Optional[list[str]] = None,
        strict_tasks: bool = True,
    ):
        super().__init__()

        # --- NEW: normalize data_dir / frames_dir to lists, and pair them ---
        if isinstance(data_dir, (str, os.PathLike)):
            data_dirs = [str(data_dir)]
        else:
            data_dirs = [str(x) for x in data_dir]

        if isinstance(frames_dir, (str, os.PathLike)):
            frames_dirs = [str(frames_dir)]
        else:
            frames_dirs = [str(x) for x in frames_dir]

        if len(data_dirs) != len(frames_dirs):
            if len(data_dirs) == 1:
                data_dirs = data_dirs * len(frames_dirs)
            elif len(frames_dirs) == 1:
                frames_dirs = frames_dirs * len(data_dirs)
            else:
                raise ValueError(f"data_dir and frames_dir must have same length (or one must be length-1). "
                                 f"Got {len(data_dirs)} and {len(frames_dirs)}")

        self.data_dirs = data_dirs
        self.frames_dirs = frames_dirs
        self.sources = list(zip(self.data_dirs, self.frames_dirs))
        # --- END NEW ---

        self.T = int(seq_len)
        self.H = int(img_size)
        self.W = int(img_size)
        self.A = int(action_dim)
        self.lang_dim = int(lang_dim)
        self.shard_size = int(shard_size)
        self.cache_bytes = int(cache_mb) * 1024 * 1024
        self.verbose = bool(verbose)
        self.tasks_filter = None if tasks is None else set(tasks)
        self.strict_tasks = bool(strict_tasks)

        # --- Task metadata (action_dim + text_embedding) ---
        self.task_meta: Optional[dict] = None
        if tasks_json and os.path.exists(tasks_json):
            try:
                with open(tasks_json, "r") as f:
                    self.task_meta = json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[WMDataset] Warning: failed to load tasks_json={tasks_json}: {e}")
                self.task_meta = None
        elif tasks_json and self.verbose:
            print(f"[WMDataset] Warning: tasks_json not found at {tasks_json} (continuing with zeros lang_emb + default action masks).")

        self._zero_lang = torch.zeros(self.lang_dim, dtype=torch.float32)

        # LRU cache for shards: key=(task_idx, seg_idx, shard_idx) -> frames tensor
        self._cache = OrderedDict()
        self._cache_nbytes = 0

        # --- NEW: Discover tasks from ALL data_dirs/*.pt (dedup, preserve first-seen order) ---
        found_tasks = []
        seen = set()
        for dd in self.data_dirs:
            demo_paths = sorted(glob.glob(os.path.join(dd, "*.pt")))
            for p in demo_paths:
                t = os.path.splitext(os.path.basename(p))[0]
                if t not in seen:
                    seen.add(t)
                    found_tasks.append(t)
        # --- END NEW ---

        if self.tasks_filter is not None:
            requested = [t for t in tasks if t in self.tasks_filter] if tasks is not None else []
            if len(requested) == 0:
                requested = [t for t in found_tasks if t in self.tasks_filter]
            tasks = requested

            if self.verbose:
                missing = [t for t in self.tasks_filter if t not in set(found_tasks)]
                print(f"[WMDataset] Task filter: keeping {len(tasks)}/{len(found_tasks)} tasks")
                if missing:
                    msg = f"[WMDataset] WARNING: {len(missing)} requested tasks not found in data_dir(s) (e.g. {missing[:5]})"
                    if self.strict_tasks:
                        raise FileNotFoundError(msg)
                    else:
                        print(msg)
        else:
            tasks = found_tasks

        # Stored per-task
        self.tasks = []
        self.demo_paths = []     # kept for compatibility/debug; now stores list of per-task demo paths (joined)
        self.shard_lists = []    # NOW: per task -> list[segments], each segment is list[shard_paths]
        self.seg_cum_frames = [] # per task -> cumulative frame counts across segments (for indexing)

        self.ep = []
        self.act = []
        self.rew = []
        self.valid_starts = []
        self._cum_counts = []

        # Precomputed per-task metadata used by __getitem__
        self._emb_ids = []
        self._act_dims = []
        self._act_mask_1d = []
        self._lang_embs = []

        total = 0
        for task in tasks:
            # --- NEW: gather segments for this task from each (data_dir, frames_dir) source ---
            seg_eps = []
            seg_acts = []
            seg_rews = []
            seg_shards = []
            seg_num_frames = []
            seg_demo_paths = []

            ep_offset = 0  # ensures episode ids are unique across segments

            for (dd, fd) in self.sources:
                dp = os.path.join(dd, f"{task}.pt")
                shard_glob = os.path.join(fd, task, "*shard*.pt")
                shards = sorted(glob.glob(shard_glob))
                if not os.path.exists(dp) or len(shards) == 0:
                    continue

                # Load demo tensors
                try:
                    td = torch.load(dp, map_location="cpu", weights_only=False)
                except Exception as e:
                    if self.verbose:
                        print(f"[WMDataset] Skipping task={task} source=({dd},{fd}): torch.load demo failed: {e}")
                    continue

                try:
                    ep = td["episode"].to(torch.int64).cpu()
                    act = td["action"].cpu()
                    rew = td["reward"].cpu()
                except Exception as e:
                    if self.verbose:
                        print(f"[WMDataset] Skipping task={task} source=({dd},{fd}): missing keys in demo: {e}")
                    continue

                if rew.ndim == 2 and rew.shape[-1] == 1:
                    rew = rew.squeeze(-1)
                rew = rew.to(torch.float32)

                if act.ndim == 1:
                    act = act.unsqueeze(-1)
                act = act.to(torch.float32)

                N = int(rew.shape[0])
                if act.shape[0] != N or ep.shape[0] != N:
                    if self.verbose:
                        print(f"[WMDataset] Skipping task={task} source=({dd},{fd}): length mismatch ep/act/rew.")
                    continue

                # Determine frames available in this source segment (load only last shard)
                try:
                    last = torch.load(shards[-1], map_location="cpu", weights_only=False)
                    frames_last = last["frames"]
                    last_len = int(frames_last.shape[0])
                except Exception as e:
                    if self.verbose:
                        print(f"[WMDataset] Skipping task={task} source=({dd},{fd}): torch.load last shard failed: {e}")
                    continue

                N_frames_avail = (len(shards) - 1) * self.shard_size + last_len
                N_eff = min(N, N_frames_avail)
                if N_eff < (self.T + 1):
                    if self.verbose:
                        print(f"[WMDataset] Skipping task={task} source=({dd},{fd}): not enough frames (N_eff={N_eff}) for T={self.T}.")
                    continue

                ep = ep[:N_eff]
                act = act[:N_eff]
                rew = rew[:N_eff]

                # Make episode IDs unique across segments to prevent windows crossing boundaries
                if ep.numel() > 0:
                    seg_max = int(ep.max().item())
                else:
                    seg_max = 0
                ep = ep + ep_offset
                ep_offset += seg_max + 1

                seg_eps.append(ep)
                seg_acts.append(act)
                seg_rews.append(rew)
                seg_shards.append(shards)
                seg_num_frames.append(int(N_eff))
                seg_demo_paths.append(dp)

            if len(seg_eps) == 0:
                if self.verbose:
                    print(f"[WMDataset] Skipping task={task}: missing demo+shards across all sources.")
                continue

            # Concatenate segments for this task
            ep = torch.cat(seg_eps, dim=0)
            act = torch.cat(seg_acts, dim=0)
            rew = torch.cat(seg_rews, dim=0)

            N_eff = int(rew.shape[0])
            # --- END NEW segment gathering/concat ---

            # Valid starts: need obs indices i..i+T and transitions at indices i+1..i+T
            start_count = N_eff - self.T  # i in [0, start_count-1]

            # episode consistency: ensure the whole window is within same episode
            ep_ok = (ep[:start_count] == ep[self.T:self.T + start_count])

            # filter invalid transitions (nan action or nan reward)
            act_nan = torch.isnan(act).any(dim=-1)
            rew_nan = torch.isnan(rew)
            step_ok = ~(act_nan | rew_nan)  # length N_eff

            # transitions live at indices 1..N_eff-1
            step_ok2 = step_ok[1:]          # length N_eff-1

            # for each start i, need step_ok at indices (i+1 .. i+T) all true
            cs = torch.cumsum(step_ok2.to(torch.int32), dim=0)
            end = torch.arange(start_count) + (self.T - 1)
            prev = torch.arange(start_count) - 1
            prev_cs = torch.zeros(start_count, dtype=cs.dtype)
            m = prev >= 0
            prev_cs[m] = cs[prev[m]]
            window_sum = cs[end] - prev_cs
            window_ok = (window_sum == self.T)

            valid = ep_ok & window_ok
            valid_idx = valid.nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                if self.verbose:
                    print(f"[WMDataset] Skipping task={task}: no valid windows after filtering.")
                continue

            # --- per-task action_dim + mask from tasks.json ---
            act_dim = self.A
            if self.task_meta is not None and task in self.task_meta:
                md = self.task_meta[task]
                if "action_dim" in md:
                    try:
                        act_dim = int(md["action_dim"])
                    except Exception:
                        act_dim = self.A
            act_dim = max(0, min(act_dim, self.A))

            act_mask_1d = torch.zeros(self.A, dtype=torch.float32)
            if act_dim > 0:
                act_mask_1d[:act_dim] = 1.0

            # --- per-task language embedding from tasks.json ---
            lang = self._zero_lang
            if self.task_meta is not None and task in self.task_meta and "text_embedding" in self.task_meta[task]:
                te = self.task_meta[task]["text_embedding"]
                l = torch.tensor(te, dtype=torch.float32)
                if l.numel() != self.lang_dim:
                    raise RuntimeError(f"text_embedding dim mismatch for task {task}: {tuple(l.shape)} vs {self.lang_dim}")
                lang = l

            # Store
            self.tasks.append(task)

            # keep a debug string of demo paths used
            self.demo_paths.append(seg_demo_paths)

            # NEW: per-task segments of shard lists + cumulative frame counts
            self.shard_lists.append(seg_shards)
            cum = []
            s = 0
            for nf in seg_num_frames:
                s += int(nf)
                cum.append(s)
            self.seg_cum_frames.append(cum)

            self.ep.append(ep)
            self.act.append(act)
            self.rew.append(rew)
            self.valid_starts.append(valid_idx)

            # precomputed metadata per task index
            task_idx = len(self.tasks) - 1
            self._emb_ids.append(torch.tensor(task_idx, dtype=torch.long))
            self._act_dims.append(act_dim)
            self._act_mask_1d.append(act_mask_1d)
            self._lang_embs.append(lang)

            total += int(valid_idx.numel())
            self._cum_counts.append(total)

            if self.verbose:
                print(f"[WMDataset] task={task} segments={len(seg_shards)} N_eff={N_eff} "
                      f"valid={valid_idx.numel()} act_dim={act_dim} lang={'yes' if lang is not self._zero_lang else 'no'}")

        self.num_tasks = len(self.tasks)
        assert self.num_tasks > 0, "No tasks found with both demo .pt and frame shards."
        if self.verbose:
            print(f"[WMDataset] Total valid sequences: {self._cum_counts[-1]} across {self.num_tasks} tasks.")

    def __len__(self):
        return self._cum_counts[-1]

    def _lookup(self, idx: int):
        lo, hi = 0, len(self._cum_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_counts[mid]:
                hi = mid
            else:
                lo = mid + 1
        task_idx = lo
        prev = 0 if task_idx == 0 else self._cum_counts[task_idx - 1]
        local = idx - prev
        start = int(self.valid_starts[task_idx][local].item())
        return task_idx, start

    def _cache_get(self, key):
        if key in self._cache:
            v = self._cache.pop(key)
            self._cache[key] = v
            return v
        return None

    def _cache_put(self, key, tensor):
        nbytes = tensor.nbytes
        while self._cache_nbytes + nbytes > self.cache_bytes and len(self._cache) > 0:
            _, v = self._cache.popitem(last=False)
            self._cache_nbytes -= v.nbytes
        self._cache[key] = tensor
        self._cache_nbytes += nbytes

    # --- CHANGED: add seg_idx to shard cache/load ---
    def _load_shard_frames(self, task_idx: int, seg_idx: int, shard_idx: int) -> torch.Tensor:
        key = (task_idx, seg_idx, shard_idx)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        path = self.shard_lists[task_idx][seg_idx][shard_idx]
        td = torch.load(path, map_location="cpu", weights_only=False)
        frames = td["frames"]

        # Normalize to (N,3,H,W)
        if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()

        # Ensure uint8 storage (robust to float in [0,1] or [0,255])
        if frames.dtype != torch.uint8:
            frames_f = frames.to(torch.float32)
            mx = float(frames_f.max().item()) if frames_f.numel() > 0 else 0.0
            if mx > 1.5:
                frames = frames_f.clamp(0, 255).to(torch.uint8)
            else:
                frames = (frames_f.clamp(0, 1) * 255.0).to(torch.uint8)

        if frames.shape[-2] != self.H or frames.shape[-1] != self.W:
            raise RuntimeError(f"Shard frame size {tuple(frames.shape[-2:])} != {(self.H, self.W)} in {path}")

        self._cache_put(key, frames)
        return frames

    # --- CHANGED: map global frame idx -> segment -> shard within segment ---
    def _get_frames(self, task_idx: int, start: int, length: int) -> torch.Tensor:
        out = []
        idx = int(start)
        remaining = int(length)

        seg_cum = self.seg_cum_frames[task_idx]

        while remaining > 0:
            seg_idx = bisect.bisect_right(seg_cum, idx)
            prev_cum = 0 if seg_idx == 0 else seg_cum[seg_idx - 1]
            seg_end = seg_cum[seg_idx]
            local_idx = idx - prev_cum

            shard_idx = local_idx // self.shard_size
            off = local_idx % self.shard_size

            frames = self._load_shard_frames(task_idx, seg_idx, shard_idx)

            take = min(remaining, frames.shape[0] - off)

            # don't read past the segment's N_eff (important if segment was truncated)
            seg_remaining = seg_end - idx
            take = min(take, seg_remaining)

            if take <= 0:
                raise RuntimeError(
                    f"Frame indexing error task={self.tasks[task_idx]} idx={idx} seg_idx={seg_idx} "
                    f"local_idx={local_idx} shard_idx={shard_idx} off={off} shard_len={frames.shape[0]}"
                )

            out.append(frames[off:off + take])
            idx += take
            remaining -= take

        return torch.cat(out, dim=0)  # (length,3,H,W) uint8

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, start = self._lookup(int(idx))

        obs = self._get_frames(task_idx, start, self.T + 1)  # (T+1,3,H,W) uint8

        # Transition from obs[t] -> obs[t+1] uses action/reward stored at index (t+1)
        act = self.act[task_idx][start + 1 : start + 1 + self.T]   # (T,16) float32 (padded)
        rew = self.rew[task_idx][start + 1 : start + 1 + self.T]   # (T,) float32

        Ad = int(self._act_dims[task_idx])
        act_padded = torch.zeros(self.T, self.A, dtype=torch.float32)
        if Ad > 0:
            act_padded[:, :Ad] = torch.nan_to_num(act[:, :Ad], nan=0.0)

        act_mask = self._act_mask_1d[task_idx][None, :].expand(self.T, self.A).contiguous()

        return {
            "obs": obs,
            "act": act_padded,
            "act_mask": act_mask,
            "rew": rew,
            "lang_emb": self._lang_embs[task_idx],
            "emb_id": self._emb_ids[task_idx],
        }


def collate_batch(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
