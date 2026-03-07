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
# TanhNormal distribution — SAC-style squashed Gaussian for [-1, 1] actions
# ---------------------------------------------------------------------------

class TanhNormal:
    """Normal distribution followed by tanh squashing.

    Samples: x ~ Normal(mu, sigma), a = tanh(x)
    Log prob: log p(a) = log N(x; mu, sigma) - sum(log(1 - tanh(x)^2))

    This gives a proper density over [-1, 1] with correct Jacobian correction.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Args:
            mu: (..., action_dim) pre-tanh mean
            sigma: (..., action_dim) std in pre-tanh space
        """
        self.mu = mu
        self.sigma = sigma
        self._normal = Independent(Normal(mu, sigma), 1)

    def rsample(self) -> torch.Tensor:
        """Differentiable sample in [-1, 1]."""
        x = self._normal.rsample()
        return torch.tanh(x)

    def rsample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample and compute log_prob without atanh round-trip (TD-MPC2 style).

        This is more numerically stable than rsample() + log_prob() because it
        computes the Jacobian directly from the pre-tanh sample, avoiding the
        lossy atanh(tanh(x)) round-trip when actions saturate near ±1.

        Returns:
            actions: (..., action_dim) in [-1, 1]
            log_prob: (...) log probabilities
        """
        x = self._normal.rsample()
        actions = torch.tanh(x)
        log_p = self._normal.log_prob(x)
        # Jacobian: -sum(log(1 - tanh(x)^2)), computed from x directly
        log_p = log_p - torch.log(torch.relu(1 - actions.pow(2)) + 1e-6).sum(dim=-1)
        return actions, log_p

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log prob of external actions in [-1, 1] with Jacobian correction.

        For log_prob of freshly sampled actions, prefer rsample_with_log_prob()
        which avoids the lossy atanh round-trip.

        Args:
            actions: (..., action_dim) in [-1, 1]

        Returns:
            (...) log probabilities (summed over action dims)
        """
        return self.log_prob_per_dim(actions).sum(dim=-1)

    def log_prob_per_dim(self, actions: torch.Tensor) -> torch.Tensor:
        """Per-dimension log prob with Jacobian correction.

        Args:
            actions: (..., action_dim) in [-1, 1]

        Returns:
            (..., action_dim) per-dimension log probabilities
        """
        a = actions.clamp(-0.999, 0.999)
        x = torch.atanh(a)
        # Per-dim normal log prob (without Independent reduction)
        log_p = self._normal.base_dist.log_prob(x)
        # Per-dim Jacobian correction
        log_p = log_p - torch.log(torch.relu(1 - a.pow(2)) + 1e-6)
        return log_p

    @property
    def mean(self) -> torch.Tensor:
        """Deterministic action (tanh of the mean)."""
        return torch.tanh(self.mu)


# ---------------------------------------------------------------------------
# Policy Head — TanhNormal for continuous actions in [-1, 1]
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Predicts continuous actions from agent token outputs.

    Uses TanhNormal: MLP outputs pre-tanh mean + std, sampling applies tanh,
    log_prob includes the Jacobian correction for proper density over [-1, 1].

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
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.mtp_length = int(mtp_length)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # One MLP outputs mean + raw_std for all L future steps
        out_dim = mtp_length * action_dim * 2  # mean and raw_std
        self.mlp = HeadMLP(d_model, out_dim, hidden, mlp_depth)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: (B, T, d_model) — agent token outputs (squeezed from n_agent=1)

        Returns:
            mu:  (B, T, L, action_dim) — pre-tanh mean
            std: (B, T, L, action_dim) — std in pre-tanh space
        """
        B, T, D = h_t.shape
        raw = self.mlp(h_t)  # (B, T, L*A*2)
        raw = raw.view(B, T, self.mtp_length, self.action_dim, 2)
        mu = raw[..., 0]   # (B, T, L, A) — pre-tanh, unbounded
        std_raw = raw[..., 1]

        # Bounded log_std via tanh (TD-MPC2 style)
        # log_std ∈ [log_std_min, log_std_max] → std ∈ [exp(min), exp(max)]
        # With defaults: std ∈ [0.0067, 7.389]
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(std_raw) + 1)
        std = torch.exp(log_std)

        return mu, std

    def dist(self, h_t: torch.Tensor, step: Optional[int] = None) -> TanhNormal:
        """Get TanhNormal action distribution.

        Args:
            h_t: (B, T, d_model)
            step: if given, return dist for that MTP step only (batch shape (B, T));
                  if None, return dist for all L steps (batch shape (B, T, L))

        Returns:
            TanhNormal distribution with event shape (action_dim,)
        """
        mu, std = self.forward(h_t)  # (B, T, L, A)
        if step is not None:
            return TanhNormal(mu[:, :, step], std[:, :, step])
        return TanhNormal(mu, std)

    def sample(self, h_t: torch.Tensor, step: Optional[int] = None) -> torch.Tensor:
        """Sample actions in [-1, 1].

        Returns: (B, T, action_dim) if step given, else (B, T, L, action_dim)
        """
        return self.dist(h_t, step).rsample()

    def log_prob(self, h_t: torch.Tensor, actions: torch.Tensor,
                 step: Optional[int] = None) -> torch.Tensor:
        """Log prob of actions under the TanhNormal policy.

        Args:
            h_t: (B, T, d_model)
            actions: (B, T, action_dim) if step given, else (B, T, L, action_dim)

        Returns: (B, T) if step given, else (B, T, L)
        """
        return self.dist(h_t, step).log_prob(actions)


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
        num_bins: int = 101,
        low: float = -10.0,
        high: float = 10.0,
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

    def dist(self, h_t: torch.Tensor, step: Optional[int] = None) -> TwoHotDist:
        """Get reward distribution.

        Args:
            step: if given, return dist for that MTP step only (batch shape (B, T));
                  if None, return dist for all L steps (batch shape (B, T, L))
        """
        logits = self.forward(h_t)  # (B, T, L, num_bins)
        if step is not None:
            return TwoHotDist(logits[:, :, step], low=self.low, high=self.high)
        return TwoHotDist(logits, low=self.low, high=self.high)

    def predict(self, h_t: torch.Tensor, step: Optional[int] = None) -> torch.Tensor:
        """Predicted reward (expected value). Returns (B, T) or (B, T, L)."""
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
        num_bins: int = 101,
        low: float = -10.0,
        high: float = 10.0,
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
    action_dim: int = 1,
    entropy_coef: float = 3e-4,
) -> torch.Tensor:
    """Policy loss with advantage-weighted policy gradient + KL + entropy.

    L = -mean(normalize(A) * log π / action_dim)   [advantage-weighted PG]
        + entropy_coef * mean(log π / action_dim)   [entropy regularization]
        + β * mean(log π - log π_prior) / action_dim [KL to behavioral prior]

    Advantages are normalized to zero mean / unit std. When advantages have
    near-zero variance (all similar), they are zeroed out → no gradient.

    Args:
        log_probs: (N,) log π(a|s) under current policy
        advantages: (N,) advantage estimates A_s = R^λ_s - v_s
        log_probs_prior: (N,) log π_prior(a|s) under frozen behavioral prior, or None
        alpha: unused (kept for API compatibility)
        beta: KL regularization coefficient
        action_dim: number of action dimensions (for normalizing continuous log_probs)
        entropy_coef: coefficient for entropy regularization (prevents std collapse)

    Returns:
        scalar loss
    """
    # Normalize log_probs by action_dim for scale-invariance
    lp = log_probs / max(action_dim, 1)

    # Normalize advantages to zero mean / unit std.
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - advantages.mean()) / adv_std
    else:
        # No meaningful advantage signal — zero out to prevent runaway
        advantages = torch.zeros_like(advantages)

    # Advantage-weighted policy gradient.
    # More robust than sign-split for continuous actions: when all advantages
    # are similar, normalized advantages → 0 → no gradient (correct behavior).
    # When there's real signal, positive advantages push log_prob up,
    # negative push down, proportional to magnitude.
    policy_loss = -(advantages.detach() * lp).mean()

    # Entropy regularization: minimize mean(lp) ≈ maximize entropy.
    # Prevents std collapse for continuous actions.
    entropy_loss = entropy_coef * lp.mean()

    # KL to behavioral prior (also normalized by action_dim)
    kl_loss = torch.tensor(0.0, device=log_probs.device)
    if log_probs_prior is not None and beta > 0:
        lp_prior = log_probs_prior / max(action_dim, 1)
        kl_loss = (lp - lp_prior).mean()

    return policy_loss + entropy_loss + beta * kl_loss
