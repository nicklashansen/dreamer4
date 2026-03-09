import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from agent import TanhNormal


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    every: int = 25
    topk: int = 3
    bc_spike_mult: float = 8.0
    bc_abs_threshold: float = 100.0
    ema_decay: float = 0.99
    jsonl_path: Optional[str] = None
    snapshot_dir: Optional[str] = None
    snapshot_topk: int = 3


def _safe_float(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.detach().float().mean().item())


def _quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float(), q).item())


def _group_stats(
    metric: torch.Tensor,
    group_ids: torch.Tensor,
    id_to_name: List[str],
    *,
    edge_frac: Optional[torch.Tensor] = None,
    std_mean: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if metric.numel() == 0:
        return out
    uniq = torch.unique(group_ids.detach().cpu(), sorted=True)
    for gid_t in uniq:
        gid = int(gid_t.item())
        mask = (group_ids == gid_t.to(group_ids.device))
        v = metric[mask]
        name = id_to_name[gid] if 0 <= gid < len(id_to_name) else f"unknown_{gid}"
        row: Dict[str, Any] = {
            "id": gid,
            "name": name,
            "count": int(mask.sum().item()),
            "bc_mean": _safe_float(v),
            "bc_p95": _quantile(v, 0.95),
            "bc_max": float(v.max().item()),
        }
        if edge_frac is not None:
            row["edge_frac_mean"] = _safe_float(edge_frac[mask])
        if std_mean is not None:
            row["std_mean"] = _safe_float(std_mean[mask])
        out.append(row)
    out.sort(key=lambda x: x["bc_mean"], reverse=True)
    return out


class AgentDiagnosticsHook:
    """Detailed per-batch diagnostics for Phase-2 training.

    Emits JSONL events with per-task/source breakdown and outlier sample details.
    On spikes, dumps top-K full samples to disk for exact repro.
    """

    def __init__(
        self,
        cfg: DiagnosticsConfig,
        *,
        task_names: List[str],
        source_names: List[str],
    ):
        self.cfg = cfg
        self.task_names = task_names
        self.source_names = source_names
        self.bc_ema = 1.0

        self.jsonl_path: Optional[Path] = None
        self.snapshot_dir: Optional[Path] = None
        self._f = None
        if self.cfg.enabled:
            if self.cfg.jsonl_path:
                self.jsonl_path = Path(self.cfg.jsonl_path)
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                self._f = self.jsonl_path.open("a", buffering=1, encoding="utf-8")
            if self.cfg.snapshot_dir:
                self.snapshot_dir = Path(self.cfg.snapshot_dir)
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def _write(self, payload: Dict[str, Any]) -> None:
        if self._f is not None:
            self._f.write(json.dumps(payload, sort_keys=True) + "\n")

    @torch.no_grad()
    def _bc_details(
        self,
        policy,
        h_t: torch.Tensor,
        actions: torch.Tensor,
        act_mask: torch.Tensor,
        mtp_length: int,
    ) -> Dict[str, torch.Tensor]:
        mu, std = policy(h_t)  # (B,T,L,A)
        B, T, L, A = mu.shape
        max_l = min(L, mtp_length)

        bc_sum = torch.zeros(B, device=mu.device, dtype=torch.float32)
        bc_cnt = torch.zeros(B, device=mu.device, dtype=torch.float32)
        edge_num = torch.zeros(B, device=mu.device, dtype=torch.float32)
        edge_den = torch.zeros(B, device=mu.device, dtype=torch.float32)
        valid_num = torch.zeros(B, device=mu.device, dtype=torch.float32)
        valid_den = torch.zeros(B, device=mu.device, dtype=torch.float32)
        mu_err_acc = torch.zeros(B, device=mu.device, dtype=torch.float32)
        z_err_acc = torch.zeros(B, device=mu.device, dtype=torch.float32)
        hz = h_t.float().norm(dim=-1).mean(dim=1)  # (B,)

        for l in range(max_l):
            valid_T = T - l
            if valid_T <= 0:
                break
            target = actions[:, l:l + valid_T, :A].clamp(-1, 1)
            mask_l = act_mask[:, l:l + valid_T, :A].float()

            dist_l = TanhNormal(mu[:, :valid_T, l, :A], std[:, :valid_T, l, :A])
            per_dim = dist_l.log_prob_per_dim(target)
            nll_t = -(per_dim * mask_l).sum(dim=-1) / mask_l.sum(dim=-1).clamp_min(1.0)
            bc_sum += nll_t.mean(dim=1)
            bc_cnt += 1.0

            edge_hits = ((target.abs() > 0.98).float() * mask_l).sum(dim=(1, 2))
            edge_all = mask_l.sum(dim=(1, 2)).clamp_min(1.0)
            edge_num += edge_hits
            edge_den += edge_all

            valid_num += mask_l.sum(dim=(1, 2))
            valid_den += torch.full_like(valid_num, float(valid_T * A))

            mu0 = mu[:, :valid_T, l, :A]
            std0 = std[:, :valid_T, l, :A].clamp_min(1e-6)
            abs_mu_err = (mu0 - target).abs() * mask_l
            mu_err_acc += abs_mu_err.sum(dim=(1, 2)) / mask_l.sum(dim=(1, 2)).clamp_min(1.0)

            at = torch.atanh(target.clamp(-0.999, 0.999))
            z_err = ((at - mu0).abs() / std0) * mask_l
            z_err_acc += z_err.sum(dim=(1, 2)) / mask_l.sum(dim=(1, 2)).clamp_min(1.0)

        bc_per = bc_sum / bc_cnt.clamp_min(1.0)
        std_per = std.mean(dim=(1, 2, 3))
        edge_frac = edge_num / edge_den.clamp_min(1.0)
        valid_frac = valid_num / valid_den.clamp_min(1.0)
        mu_err_mean = mu_err_acc / bc_cnt.clamp_min(1.0)
        z_err_mean = z_err_acc / bc_cnt.clamp_min(1.0)
        mu0 = mu[:, :, 0, :A]
        std0 = std[:, :, 0, :A]
        pred_mean = torch.tanh(mu0)
        pred_lo = torch.tanh(mu0 - 2.0 * std0)
        pred_hi = torch.tanh(mu0 + 2.0 * std0)
        return {
            "bc_per_sample": bc_per,
            "std_per_sample": std_per,
            "edge_frac": edge_frac,
            "valid_frac": valid_frac,
            "mu_err_mean": mu_err_mean,
            "z_err_mean": z_err_mean,
            "h_norm": hz,
            "pred_mean_l0": pred_mean,
            "pred_lo_l0": pred_lo,
            "pred_hi_l0": pred_hi,
        }

    def _dump_spike_samples(
        self,
        *,
        step: int,
        top_idx: torch.Tensor,
        bottom_idx: torch.Tensor,
        top_outliers: List[Dict[str, Any]],
        obs_u8: torch.Tensor,
        actions: torch.Tensor,
        act_mask: torch.Tensor,
        rewards: torch.Tensor,
        task_ids: torch.Tensor,
        source_ids: torch.Tensor,
        segment_idx: torch.Tensor,
        window_start: torch.Tensor,
        episode_id: torch.Tensor,
        sample_idx: torch.Tensor,
        pred_mean_l0: torch.Tensor,
        pred_lo_l0: torch.Tensor,
        pred_hi_l0: torch.Tensor,
    ) -> Optional[str]:
        if self.snapshot_dir is None:
            return None
        k = min(int(top_idx.numel()), max(1, int(self.cfg.snapshot_topk)))
        top_idx = top_idx[:k]
        bottom_idx = bottom_idx[:k]

        payload = {
            "step": int(step),
            "top_outliers": top_outliers[:k],
            "hard_samples": [],
            "easy_samples": [],
        }
        for bidx_t in top_idx:
            bidx = int(bidx_t.item())
            task_id = int(task_ids[bidx].item())
            source_id = int(source_ids[bidx].item())
            task_name = self.task_names[task_id] if 0 <= task_id < len(self.task_names) else f"unknown_{task_id}"
            source_name = self.source_names[source_id] if 0 <= source_id < len(self.source_names) else f"unknown_{source_id}"
            payload["hard_samples"].append({
                "batch_index": bidx,
                "sample_idx": int(sample_idx[bidx].item()),
                "task_id": task_id,
                "task": task_name,
                "source_id": source_id,
                "source": source_name,
                "segment_idx": int(segment_idx[bidx].item()),
                "window_start": int(window_start[bidx].item()),
                "episode_id": int(episode_id[bidx].item()),
                "obs_u8": obs_u8[bidx].detach().cpu(),
                "act": actions[bidx].detach().cpu(),
                "act_mask": act_mask[bidx].detach().cpu(),
                "rew": rewards[bidx].detach().cpu(),
                "pred_mean_l0": pred_mean_l0[bidx].detach().cpu(),
                "pred_lo_l0": pred_lo_l0[bidx].detach().cpu(),
                "pred_hi_l0": pred_hi_l0[bidx].detach().cpu(),
            })
        for bidx_t in bottom_idx:
            bidx = int(bidx_t.item())
            task_id = int(task_ids[bidx].item())
            source_id = int(source_ids[bidx].item())
            task_name = self.task_names[task_id] if 0 <= task_id < len(self.task_names) else f"unknown_{task_id}"
            source_name = self.source_names[source_id] if 0 <= source_id < len(self.source_names) else f"unknown_{source_id}"
            payload["easy_samples"].append({
                "batch_index": bidx,
                "sample_idx": int(sample_idx[bidx].item()),
                "task_id": task_id,
                "task": task_name,
                "source_id": source_id,
                "source": source_name,
                "segment_idx": int(segment_idx[bidx].item()),
                "window_start": int(window_start[bidx].item()),
                "episode_id": int(episode_id[bidx].item()),
                "obs_u8": obs_u8[bidx].detach().cpu(),
                "act": actions[bidx].detach().cpu(),
                "act_mask": act_mask[bidx].detach().cpu(),
                "rew": rewards[bidx].detach().cpu(),
                "pred_mean_l0": pred_mean_l0[bidx].detach().cpu(),
                "pred_lo_l0": pred_lo_l0[bidx].detach().cpu(),
                "pred_hi_l0": pred_hi_l0[bidx].detach().cpu(),
            })
        out = self.snapshot_dir / f"spike_step_{step:07d}.pt"
        torch.save(payload, out)
        return str(out)

    def maybe_record(
        self,
        *,
        step: int,
        bc_loss: float,
        dyn_loss: float,
        rew_loss: float,
        total_loss: float,
        flow_mse: float,
        boot_mse: float,
        h_t: torch.Tensor,
        policy,
        actions: torch.Tensor,
        act_mask: torch.Tensor,
        rewards: torch.Tensor,
        obs_u8: torch.Tensor,
        mtp_length: int,
        task_ids: torch.Tensor,
        source_ids: torch.Tensor,
        segment_idx: torch.Tensor,
        window_start: torch.Tensor,
        episode_id: torch.Tensor,
        sample_idx: torch.Tensor,
        reward_pred_mean: float,
        reward_target_mean: float,
        elapsed_hours: float,
    ) -> Optional[str]:
        if not self.cfg.enabled:
            return None

        self.bc_ema = self.cfg.ema_decay * self.bc_ema + (1.0 - self.cfg.ema_decay) * abs(float(bc_loss))
        spike_thr = max(self.cfg.bc_abs_threshold, self.cfg.bc_spike_mult * self.bc_ema)
        is_spike = float(bc_loss) >= spike_thr
        do_detail = is_spike or (step % max(1, int(self.cfg.every)) == 0)
        if not do_detail:
            return None

        d = self._bc_details(policy, h_t, actions, act_mask, mtp_length)
        bc_per = d["bc_per_sample"].detach().float().cpu()
        std_per = d["std_per_sample"].detach().float().cpu()
        edge_frac = d["edge_frac"].detach().float().cpu()
        valid_frac = d["valid_frac"].detach().float().cpu()
        mu_err_mean = d["mu_err_mean"].detach().float().cpu()
        z_err_mean = d["z_err_mean"].detach().float().cpu()
        h_norm = d["h_norm"].detach().float().cpu()
        pred_mean_l0 = d["pred_mean_l0"].detach().float().cpu()
        pred_lo_l0 = d["pred_lo_l0"].detach().float().cpu()
        pred_hi_l0 = d["pred_hi_l0"].detach().float().cpu()

        task_ids_cpu = task_ids.detach().long().cpu()
        source_ids_cpu = source_ids.detach().long().cpu()
        segment_idx_cpu = segment_idx.detach().long().cpu()
        window_start_cpu = window_start.detach().long().cpu()
        episode_id_cpu = episode_id.detach().long().cpu()
        sample_idx_cpu = sample_idx.detach().long().cpu()

        by_task = _group_stats(bc_per, task_ids_cpu, self.task_names, edge_frac=edge_frac, std_mean=std_per)
        by_source = _group_stats(bc_per, source_ids_cpu, self.source_names, edge_frac=edge_frac, std_mean=std_per)

        k = min(max(1, int(self.cfg.topk)), bc_per.numel())
        top_vals, top_idx = torch.topk(bc_per, k=k, largest=True)
        top_outliers: List[Dict[str, Any]] = []
        for i in range(k):
            idx = int(top_idx[i].item())
            task_id = int(task_ids_cpu[idx].item())
            src_id = int(source_ids_cpu[idx].item())
            task_name = self.task_names[task_id] if 0 <= task_id < len(self.task_names) else f"unknown_{task_id}"
            src_name = self.source_names[src_id] if 0 <= src_id < len(self.source_names) else f"unknown_{src_id}"
            top_outliers.append({
                "batch_index": idx,
                "sample_idx": int(sample_idx_cpu[idx].item()),
                "bc_nll": float(top_vals[i].item()),
                "std_mean": float(std_per[idx].item()),
                "edge_frac": float(edge_frac[idx].item()),
                "valid_frac": float(valid_frac[idx].item()),
                "mu_err_mean": float(mu_err_mean[idx].item()),
                "z_err_mean": float(z_err_mean[idx].item()),
                "h_norm": float(h_norm[idx].item()),
                "task_id": task_id,
                "task": task_name,
                "source_id": src_id,
                "source": src_name,
                "segment_idx": int(segment_idx_cpu[idx].item()),
                "window_start": int(window_start_cpu[idx].item()),
                "episode_id": int(episode_id_cpu[idx].item()),
            })

        snapshot_file = None
        if is_spike:
            k_ref = min(max(1, int(self.cfg.snapshot_topk)), bc_per.numel())
            _, bottom_idx = torch.topk(bc_per, k=k_ref, largest=False)
            snapshot_file = self._dump_spike_samples(
                step=step,
                top_idx=top_idx,
                bottom_idx=bottom_idx,
                top_outliers=top_outliers,
                obs_u8=obs_u8,
                actions=actions,
                act_mask=act_mask,
                rewards=rewards,
                task_ids=task_ids,
                source_ids=source_ids,
                segment_idx=segment_idx,
                window_start=window_start,
                episode_id=episode_id,
                sample_idx=sample_idx,
                pred_mean_l0=pred_mean_l0,
                pred_lo_l0=pred_lo_l0,
                pred_hi_l0=pred_hi_l0,
            )

        payload: Dict[str, Any] = {
            "kind": "phase2_diagnostics",
            "step": int(step),
            "is_spike": bool(is_spike),
            "spike_threshold": float(spike_thr),
            "elapsed_hours": float(elapsed_hours),
            "snapshot_file": snapshot_file,
            "loss": {
                "total": float(total_loss),
                "dyn": float(dyn_loss),
                "bc": float(bc_loss),
                "rew": float(rew_loss),
                "flow_mse": float(flow_mse),
                "boot_mse": float(boot_mse),
                "bc_ema": float(self.bc_ema),
            },
            "reward": {
                "pred_mean": float(reward_pred_mean),
                "target_mean": float(reward_target_mean),
            },
            "batch": {
                "size": int(bc_per.numel()),
                "bc_mean": _safe_float(bc_per),
                "bc_p95": _quantile(bc_per, 0.95),
                "bc_max": float(bc_per.max().item()) if bc_per.numel() > 0 else float("nan"),
                "std_mean": _safe_float(std_per),
                "std_min": float(std_per.min().item()) if std_per.numel() > 0 else float("nan"),
                "std_max": float(std_per.max().item()) if std_per.numel() > 0 else float("nan"),
                "edge_frac_mean": _safe_float(edge_frac),
                "valid_frac_mean": _safe_float(valid_frac),
                "mu_err_mean": _safe_float(mu_err_mean),
                "z_err_mean": _safe_float(z_err_mean),
                "h_norm_mean": _safe_float(h_norm),
                "h_norm_p95": _quantile(h_norm, 0.95),
            },
            "by_task": by_task,
            "by_source": by_source,
            "top_outliers": top_outliers,
        }
        self._write(payload)

        if is_spike:
            worst = top_outliers[0] if top_outliers else None
            if worst is not None:
                return (
                    f"[diag] BC spike step={step} bc={bc_loss:.2f} thr={spike_thr:.2f} "
                    f"worst(task={worst['task']}, source={worst['source']}, sample={worst['sample_idx']}, "
                    f"bc={worst['bc_nll']:.2f}, std={worst['std_mean']:.3f}, edge={worst['edge_frac']:.3f})"
                )
            return f"[diag] BC spike step={step} bc={bc_loss:.2f} thr={spike_thr:.2f}"
        return None
