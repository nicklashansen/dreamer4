# agent.py — RL components for Dreamer 4 (Phases 2 & 3)
#
# Components:
#   - symlog / symexp transforms
#   - TwoHotDist: symexp two-hot distribution for reward/value heads
#   - PolicyHead: MLP → TruncNormal over continuous actions
#   - RewardHead: MLP → TwoHotDist (with multi-token prediction)
#   - ValueHead: MLP → TwoHotDist (with multi-token prediction)
#   - compute_lambda_returns: TD(λ) target computation
#   - pmpo_loss: PMPO policy objective (Eq. 11 from paper)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# ---------------------------------------------------------------------------
# symlog / symexp (Dreamer v3, Hafner et al. 2023)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1.0)


# ---------------------------------------------------------------------------
# Two-Hot Distribution in symlog space
# ---------------------------------------------------------------------------

class TwoHotDist:
    """Discrete distribution over evenly-spaced bins in symlog space.

    Bins are centered at linspace(-low, high, num_bins) in symlog space.
    Targets are encoded as a two-hot: weight split between the two nearest bins.
    Expected value is sum(softmax(logits) * bin_centers), then symexp'd.
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        """
        Args:
            logits: (..., num_bins)
            low, high: range in symlog space
        """
        self.logits = logits
        self.num_bins = logits.shape[-1]
        self.low = float(low)
        self.high = float(high)

        # bin centers in symlog space
        bins = torch.linspace(self.low, self.high, self.num_bins,
                              device=logits.device, dtype=torch.float32)
        self.bins = bins  # (num_bins,)

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits.float(), dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Expected value in original (symexp'd) space."""
        p = self.probs  # (..., num_bins)
        symlog_mean = (p * self.bins).sum(dim=-1)  # (...)
        return symexp(symlog_mean)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """Compute log probability of target values under the two-hot encoding.

        Args:
            target: (...) real-valued targets

        Returns:
            (...) log probabilities
        """
        target_symlog = symlog(target.float())
        # clamp to bin range
        target_symlog = target_symlog.clamp(self.low, self.high)

        # find the two nearest bins
        # bin width
        width = (self.high - self.low) / (self.num_bins - 1)
        # continuous bin index
        idx_f = (target_symlog - self.low) / width
        lo = idx_f.floor().long().clamp(0, self.num_bins - 2)
        hi = lo + 1
        # weight for upper bin
        w_hi = idx_f - lo.float()
        w_lo = 1.0 - w_hi

        # two-hot log prob = log(w_lo * p[lo] + w_hi * p[hi])
        p = self.probs  # (..., num_bins)
        p_lo = p.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
        p_hi = p.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
        mixed = w_lo * p_lo + w_hi * p_hi
        return (mixed + 1e-8).log()


# ---------------------------------------------------------------------------
# MLP block (used by all heads)
# ---------------------------------------------------------------------------

class HeadMLP(nn.Module):
    """SwiGLU-gated MLP as used in JAX Dreamer 4 heads.

    Each layer: x → Dense(2H) → split(u, v) → u * silu(v) → Dense(H)
    Final: Dense(d_out)
    """

    def __init__(self, d_in: int, d_out: int, hidden: int, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.depth = depth
        self.gate_projs = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        d = d_in
        for _ in range(depth):
            self.gate_projs.append(nn.Linear(d, 2 * hidden))
            self.out_projs.append(nn.Linear(hidden, hidden))
            d = hidden
        self.head = nn.Linear(d, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for gate, out in zip(self.gate_projs, self.out_projs):
            uv = gate(x)
            u, v = uv.chunk(2, dim=-1)
            x = self.drop(out(u * F.silu(v)))
        return self.head(x)


# ---------------------------------------------------------------------------
# Policy Head — truncated Normal for continuous actions in [-1, 1]
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Predicts continuous actions from agent token outputs.

    Multi-token prediction: from h_t, predicts actions for t+1 .. t+L.
    During inference, only the first prediction (t+1) is used.
    """

    def __init__(
        self,
        d_model: int = 512,
        action_dim: int = 16,
        hidden: int = 512,
        mlp_depth: int = 2,
        mtp_length: int = 8,
        min_std: float = 0.1,
        init_std: float = 0.5,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.mtp_length = int(mtp_length)
        self.min_std = float(min_std)
        self._init_std = float(init_std)

        # One MLP outputs mean + raw_std for all L future steps
        out_dim = mtp_length * action_dim * 2  # mean and raw_std
        self.mlp = HeadMLP(d_model, out_dim, hidden, mlp_depth)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: (B, T, d_model) — agent token outputs (squeezed from n_agent=1)

        Returns:
            mean: (B, T, L, action_dim)
            std:  (B, T, L, action_dim)
        """
        B, T, D = h_t.shape
        raw = self.mlp(h_t)  # (B, T, L*A*2)
        raw = raw.view(B, T, self.mtp_length, self.action_dim, 2)
        mean_raw = raw[..., 0]  # (B, T, L, A)
        std_raw = raw[..., 1]

        mean = torch.tanh(mean_raw)
        std = F.softplus(std_raw + math.log(math.exp(self._init_std - self.min_std) - 1.0)) + self.min_std

        return mean, std

    def dist(self, h_t: torch.Tensor, step: int = 0) -> Independent:
        """Get action distribution for a specific MTP step.

        Args:
            h_t: (B, T, d_model)
            step: which MTP step (0 = next action)

        Returns:
            Independent Normal distribution over actions, shape (B, T)
        """
        mean, std = self.forward(h_t)
        return Independent(Normal(mean[:, :, step], std[:, :, step]), 1)

    def sample(self, h_t: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Sample actions, clamped to [-1, 1].

        Returns: (B, T, action_dim)
        """
        d = self.dist(h_t, step)
        return d.rsample().clamp(-1, 1)

    def log_prob(self, h_t: torch.Tensor, actions: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Log prob of actions under the policy.

        Args:
            h_t: (B, T, d_model)
            actions: (B, T, action_dim)

        Returns: (B, T)
        """
        d = self.dist(h_t, step)
        return d.log_prob(actions)


# ---------------------------------------------------------------------------
# Reward Head — SymExp TwoHot
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Predicts rewards from agent token outputs using symexp two-hot.

    Multi-token prediction: from h_t, predicts rewards for t+1 .. t+L.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden: int = 512,
        mlp_depth: int = 2,
        mtp_length: int = 8,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        super().__init__()
        self.mtp_length = int(mtp_length)
        self.num_bins = int(num_bins)
        self.low = float(low)
        self.high = float(high)

        out_dim = mtp_length * num_bins
        self.mlp = HeadMLP(d_model, out_dim, hidden, mlp_depth)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, T, d_model)

        Returns:
            logits: (B, T, L, num_bins)
        """
        B, T, D = h_t.shape
        raw = self.mlp(h_t)  # (B, T, L*num_bins)
        return raw.view(B, T, self.mtp_length, self.num_bins)

    def dist(self, h_t: torch.Tensor, step: int = 0) -> TwoHotDist:
        logits = self.forward(h_t)
        return TwoHotDist(logits[:, :, step], low=self.low, high=self.high)

    def predict(self, h_t: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Predicted reward (expected value). Returns (B, T)."""
        return self.dist(h_t, step).mean

    def loss(self, h_t: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Average NLL across all MTP steps that have targets.

        Args:
            h_t: (B, T, d_model)
            targets: (B, T) rewards — loss is computed for MTP steps 0..min(L, T)-1

        Returns:
            scalar loss
        """
        logits = self.forward(h_t)  # (B, T, L, num_bins)
        B, T, L, _ = logits.shape

        total_loss = torch.tensor(0.0, device=h_t.device)
        count = 0
        for l in range(L):
            # For MTP step l, target at time t is reward at t+l+1
            # But targets are already aligned: targets[:, t] = reward for transition at t
            # So for step l, we shift: target is targets[:, t+l] for input h_t[:, t]
            valid_T = T - l
            if valid_T <= 0:
                break
            dist_l = TwoHotDist(logits[:, :valid_T, l], low=self.low, high=self.high)
            target_l = targets[:, l:l + valid_T]
            nll = -dist_l.log_prob(target_l)  # (B, valid_T)
            total_loss = total_loss + nll.mean()
            count += 1

        return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Value Head — SymExp TwoHot (Phase 3 only)
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """Predicts state value from agent token outputs using symexp two-hot.

    Single-step prediction (no MTP for value).
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden: int = 512,
        mlp_depth: int = 2,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.low = float(low)
        self.high = float(high)

        self.mlp = HeadMLP(d_model, num_bins, hidden, mlp_depth)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, T, num_bins)."""
        return self.mlp(h_t)

    def dist(self, h_t: torch.Tensor) -> TwoHotDist:
        return TwoHotDist(self.forward(h_t), low=self.low, high=self.high)

    def predict(self, h_t: torch.Tensor) -> torch.Tensor:
        """Predicted value (expected value). Returns (B, T)."""
        return self.dist(h_t).mean

    def loss(self, h_t: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """NLL loss against lambda-return targets.

        Args:
            h_t: (B, T, d_model)
            targets: (B, T) lambda-return targets

        Returns: scalar loss
        """
        dist = self.dist(h_t)
        return -dist.log_prob(targets).mean()


# ---------------------------------------------------------------------------
# Lambda-return computation (for Phase 3)
# ---------------------------------------------------------------------------

def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.997,
    lam: float = 0.95,
    continuation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute TD(λ) returns.

    R_t^λ = r_t + γ c_t [(1 - λ) v_{t+1} + λ R_{t+1}^λ]
    R_T^λ = v_T

    Args:
        rewards: (B, T) rewards r_1 .. r_T
        values: (B, T+1) value estimates v_0 .. v_T (or (B, T) if bootstrap excluded)
        gamma: discount factor
        lam: λ for TD(λ)
        continuation: (B, T) continuation flags (1.0 = continue, 0.0 = terminal).
                      If None, assumed all 1.0.

    Returns:
        returns: (B, T) lambda-return targets for v_0 .. v_{T-1}
    """
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    if values.shape[-1] == T + 1:
        v = values  # (B, T+1)
    else:
        # assume values is (B, T) and last value is the bootstrap
        v = values

    if continuation is None:
        cont = torch.ones(B, T, device=device, dtype=dtype)
    else:
        cont = continuation

    returns = torch.zeros(B, T, device=device, dtype=dtype)
    # R_T = v_T
    last = v[:, -1]  # bootstrap value

    for t in reversed(range(T)):
        last = rewards[:, t] + gamma * cont[:, t] * ((1.0 - lam) * v[:, t + 1] + lam * last)
        returns[:, t] = last

    return returns


# ---------------------------------------------------------------------------
# PMPO Loss (Eq. 11 from paper)
# ---------------------------------------------------------------------------

def pmpo_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    log_probs_prior: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> torch.Tensor:
    """PMPO policy loss from Dreamer 4.

    L = -α * mean(log π(a|s) for s in D+) - (1-α) * mean(log π(a|s) for s in D-)
        + β * KL(π || π_prior)

    where D+ = {s : A_s > 0}, D- = {s : A_s ≤ 0}

    The KL term is approximated as: mean(log π - log π_prior) (reverse KL sample estimate).

    Args:
        log_probs: (N,) log π(a|s) under current policy
        advantages: (N,) advantage estimates A_s = R^λ_s - v_s
        log_probs_prior: (N,) log π_prior(a|s) under frozen behavioral prior, or None
        alpha: weight for positive advantages (0.5 = equal weight)
        beta: KL regularization coefficient

    Returns:
        scalar loss
    """
    pos_mask = (advantages > 0).float()
    neg_mask = 1.0 - pos_mask

    n_pos = pos_mask.sum().clamp(min=1.0)
    n_neg = neg_mask.sum().clamp(min=1.0)

    # Policy gradient: maximize log prob on D+, minimize on D-
    # The sign flip in D- means: we WANT low log_prob on bad states
    # Actually, re-reading the paper: D- uses sign(A) = -1, so the objective is
    # to increase log_prob on D+ and decrease on D-.
    # Loss = -(α * mean_D+(log_pi) + (1-α) * mean_D-(sign(A) * log_pi))
    # Since sign(A) < 0 for D-, this naturally pushes log_pi down for D-.

    loss_pos = -(pos_mask * log_probs).sum() / n_pos
    loss_neg = (neg_mask * log_probs).sum() / n_neg  # push down log_prob for negative advantages

    policy_loss = alpha * loss_pos + (1.0 - alpha) * loss_neg

    # KL to behavioral prior
    kl_loss = torch.tensor(0.0, device=log_probs.device)
    if log_probs_prior is not None and beta > 0:
        # Reverse KL: E_π[log π - log π_prior]
        kl_loss = (log_probs - log_probs_prior).mean()

    return policy_loss + beta * kl_loss
