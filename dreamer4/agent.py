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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# ---------------------------------------------------------------------------
# Head config validation
# ---------------------------------------------------------------------------

def _check_head_config(head, ckpt: dict, name: str):
    """Raise ValueError if a head's config doesn't match saved ckpt metadata.

    Checks num_bins / low / high for RewardHead / ValueHead,
    mtp_length for RewardHead / PolicyHead,
    action_dim for PolicyHead.
    """
    errors = []
    for attr, key in [("num_bins", f"{name}_num_bins"),
                      ("low",      f"{name}_low"),
                      ("high",     f"{name}_high"),
                      ("mtp_length", "mtp_length"),
                      ("action_dim", "action_dim")]:
        if not hasattr(head, attr):
            continue
        saved = ckpt.get(key)
        if saved is None:
            # key wasn't saved (old checkpoint) — skip
            continue
        actual = getattr(head, attr)
        if abs(float(actual) - float(saved)) > 1e-6:
            errors.append(f"  {attr}: head={actual!r} but ckpt saved {saved!r} (key='{key}')")
    if errors:
        raise ValueError(
            f"Head config mismatch for '{name}':\n" + "\n".join(errors) +
            "\nReconstruct the head with matching parameters or update the checkpoint."
        )


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

    def __init__(self, logits: torch.Tensor, low: float = -1.0, high: float = 9.9):
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
    Log prob: log p(a) = sum_j m_j * [log N(x_j; mu_j, sigma_j) - log(1 - a_j^2)]

    where m_j is an optional per-dimension validity mask (1=valid, 0=padding).
    Masking is done by zeroing the noise eps for invalid dims before squashing,
    which forces a_j = tanh(0) = 0 for those dims and excludes them from all
    log-prob sums. log_prob_per_dim is always unmasked (callers apply the mask).
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Args:
            mu:    (..., action_dim) pre-tanh mean
            sigma: (..., action_dim) std in pre-tanh space
            mask:  (..., action_dim) or broadcastable; 1.0=valid, 0.0=padding.
                   If None, all dimensions are treated as valid.
        """
        self.mu = mu
        self.sigma = sigma
        self.mask = mask
        self._normal = Independent(Normal(mu, sigma), 1)

    def rsample(self) -> torch.Tensor:
        """Differentiable sample in [-1, 1]. Invalid dims yield exactly 0."""
        x = Normal(self.mu, self.sigma).rsample()  # x = mu + sigma * eps
        if self.mask is not None:
            x = x * self.mask   # x_j = 0 for invalid dims → tanh(0) = 0
        return torch.tanh(x)

    def rsample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample + log_prob without atanh round-trip (numerically stable).

        The mask zeroes the full pre-tanh value x for invalid dims, ensuring
        a_j = tanh(0) = 0.  The per-dim log-prob of invalid dims is then also
        zeroed before summing, so they contribute nothing to entropy / KL.

        Returns:
            actions:  (..., action_dim) in [-1, 1]; invalid dims are 0.
            log_prob: (...) summed only over valid dims.
        """
        x = Normal(self.mu, self.sigma).rsample()  # x = mu + sigma * eps
        if self.mask is not None:
            x = x * self.mask   # x_j = 0 for invalid dims
        actions = torch.tanh(x)

        # Per-dim log prob + Jacobian correction computed from the masked x.
        # Invalid dims: x_j = 0, a_j = 0, log N(0; mu_j, sigma_j) - 0, then * mask = 0.
        per_dim_lp = Normal(self.mu, self.sigma).log_prob(x)
        per_dim_lp = per_dim_lp - torch.log(torch.relu(1 - actions.pow(2)) + 1e-6)
        if self.mask is not None:
            per_dim_lp = per_dim_lp * self.mask
        log_p = per_dim_lp.sum(dim=-1)
        return actions, log_p

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log prob of external actions, summed over valid dims only.

        Prefer rsample_with_log_prob() for freshly sampled actions to avoid
        the lossy atanh(tanh(x)) round-trip near saturation.

        Args:
            actions: (..., action_dim) in [-1, 1]

        Returns:
            (...) log probabilities (masked sum over action dims)
        """
        per_dim = self.log_prob_per_dim(actions)  # (..., action_dim)
        if self.mask is not None:
            per_dim = per_dim * self.mask
        return per_dim.sum(dim=-1)

    def log_prob_per_dim(self, actions: torch.Tensor) -> torch.Tensor:
        """Per-dimension log prob with Jacobian correction (no masking applied).

        Callers that need masking (e.g. bc_loss) apply it externally.

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
        """Deterministic action (tanh of the mean). Invalid dims are 0."""
        m = torch.tanh(self.mu)
        if self.mask is not None:
            m = m * self.mask
        return m


# ---------------------------------------------------------------------------
# BoundedNormal — Dreamer v3 style: plain Normal centered at tanh(mean)
# ---------------------------------------------------------------------------

class BoundedNormal:
    """Normal distribution with mean squashed to [-1, 1] via tanh.

    Unlike TanhNormal, there is NO Jacobian correction — actions are sampled
    from N(tanh(mu), sigma) directly and clipped to [-1, 1].  This avoids the
    pathological log-prob increase with growing sigma that TanhNormal suffers from.

    Dreamer v3 uses this with sigma ∈ [minstd, maxstd] via sigmoid bounding.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        self.mu = mu            # pre-squash mean (raw network output)
        self.sigma = sigma      # std in action space
        self.mask = mask
        self._mean = torch.tanh(mu)  # squashed mean in [-1, 1]
        if mask is not None:
            self._mean = self._mean * mask
        self._normal = Normal(self._mean, sigma)

    def rsample(self) -> torch.Tensor:
        x = self._normal.rsample()
        if self.mask is not None:
            x = x * self.mask
        return x.clamp(-1, 1)

    def rsample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._normal.rsample()
        if self.mask is not None:
            x = x * self.mask
        actions = x.clamp(-1, 1)
        # Log prob of the unclamped sample under the Normal (no Jacobian)
        per_dim_lp = self._normal.log_prob(x)
        if self.mask is not None:
            per_dim_lp = per_dim_lp * self.mask
        log_p = per_dim_lp.sum(dim=-1)
        return actions, log_p

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        per_dim = self._normal.log_prob(actions)
        if self.mask is not None:
            per_dim = per_dim * self.mask
        return per_dim.sum(dim=-1)

    def log_prob_per_dim(self, actions: torch.Tensor) -> torch.Tensor:
        return self._normal.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        ent = self._normal.entropy()
        if self.mask is not None:
            ent = ent * self.mask
        return ent.sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean


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
        std_parameterization: str = "tanh",  # "tanh" (TD-MPC2), "softplus", or "sigmoid" (Dreamer v3)
        min_std: float = 0.1,
        max_std: float = 1.0,   # upper bound for sigmoid parameterization
        dist_type: str = "tanh_normal",  # "tanh_normal" or "bounded_normal" (Dreamer v3)
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.mtp_length = int(mtp_length)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.std_parameterization = std_parameterization
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.dist_type = dist_type

        # One MLP outputs mean + raw_std for all L future steps
        out_dim = mtp_length * action_dim * 2  # mean and raw_std
        self.mlp = HeadMLP(d_model, out_dim, hidden, mlp_depth)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: (B, T, d_model) — agent token outputs (squeezed from n_agent=1)

        Returns:
            mu:  (B, T, L, action_dim) — raw mean (interpretation depends on dist_type)
            std: (B, T, L, action_dim) — std
        """
        B, T, D = h_t.shape
        raw = self.mlp(h_t)  # (B, T, L*A*2)
        raw = raw.view(B, T, self.mtp_length, self.action_dim, 2)
        mu = 5.0 * torch.tanh(raw[..., 0] / 5.0)  # bounded to [-5, 5]
        std_raw = raw[..., 1]

        if self.std_parameterization == "sigmoid":
            # Dreamer v3 style: std ∈ [min_std, max_std] via sigmoid
            std = (self.max_std - self.min_std) * torch.sigmoid(std_raw + 2.0) + self.min_std
        elif self.std_parameterization == "softplus":
            std = torch.nn.functional.softplus(std_raw) + self.min_std
        else:
            # TD-MPC2 style: bounded log_std via tanh
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(std_raw) + 1)
            std = torch.exp(log_std)

        return mu, std

    def dist(self, h_t: torch.Tensor, step: Optional[int] = None,
             mask: Optional[torch.Tensor] = None):
        """Get action distribution.

        Args:
            h_t:  (B, T, d_model)
            step: if given, return dist for that MTP step only (batch shape (B, T));
                  if None, return dist for all L steps (batch shape (B, T, L))
            mask: (..., action_dim) broadcastable validity mask (1=valid, 0=padding).

        Returns:
            TanhNormal or BoundedNormal distribution
        """
        mu, std = self.forward(h_t)  # (B, T, L, A)
        DistCls = BoundedNormal if self.dist_type == "bounded_normal" else TanhNormal
        if step is not None:
            return DistCls(mu[:, :, step], std[:, :, step], mask=mask)
        return DistCls(mu, std, mask=mask)

    def sample(self, h_t: torch.Tensor, step: Optional[int] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample actions in [-1, 1]. Invalid dims (mask=0) yield exactly 0.

        Returns: (B, T, action_dim) if step given, else (B, T, L, action_dim)
        """
        return self.dist(h_t, step, mask=mask).rsample()

    def log_prob(self, h_t: torch.Tensor, actions: torch.Tensor,
                 step: Optional[int] = None,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Log prob of actions under the TanhNormal policy (masked sum).

        Args:
            h_t:     (B, T, d_model)
            actions: (B, T, action_dim) if step given, else (B, T, L, action_dim)
            mask:    (..., action_dim) broadcastable; if None all dims count.

        Returns: (B, T) if step given, else (B, T, L)
        """
        return self.dist(h_t, step, mask=mask).log_prob(actions)


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
        num_bins: int = 11,
        low: float = 0.0,
        high: float = 1.25,
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
        low: float = -1.0,
        high: float = 9.0,
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
# Running normalizer (Dreamer v3 style)
# ---------------------------------------------------------------------------

class RunningNorm:
    """Running mean/std normalization with EMA, matching Dreamer v3's Normalize.

    Tracks offset (mean) and scale (std) of a stream of values, used
    for normalizing value targets (valnorm) and advantages (advnorm).
    Returns (offset, scale) and optionally normalizes in place.
    """

    def __init__(self, decay: float = 0.99, max_scale: float = 1e8,
                 min_scale: float = 1e-8):
        self.decay = decay
        self.max_scale = max_scale
        self.min_scale = min_scale
        self._mean = 0.0
        self._var = 1.0
        self._initialized = False

    def __call__(self, x: torch.Tensor, update: bool = True):
        """Return (offset, scale) and optionally update stats."""
        if update:
            batch_mean = x.detach().float().mean().item()
            batch_var = x.detach().float().var().item()
            if not self._initialized:
                self._mean = batch_mean
                self._var = batch_var
                self._initialized = True
            else:
                self._mean = self.decay * self._mean + (1 - self.decay) * batch_mean
                self._var = self.decay * self._var + (1 - self.decay) * batch_var
        scale = max(self.min_scale, min(self.max_scale, self._var ** 0.5))
        return self._mean, scale

    def stats(self):
        scale = max(self.min_scale, min(self.max_scale, self._var ** 0.5))
        return self._mean, scale


# ---------------------------------------------------------------------------
# Slow (EMA) value head for target stabilization (Dreamer v3)
# ---------------------------------------------------------------------------

class SlowValueHead:
    """Exponential moving average copy of a ValueHead for stable TD targets.

    The slow copy's parameters track the main value head with a given decay.
    Dreamer v3 also uses it as a regularization target: the value head is
    trained to match the slow copy in addition to the TD(λ) targets.
    """

    def __init__(self, value_head: ValueHead, decay: float = 0.98):
        self.decay = decay
        # Deep copy the value head
        import copy
        self.slow = copy.deepcopy(value_head)
        for p in self.slow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, value_head: ValueHead):
        """EMA update: slow ← decay * slow + (1 - decay) * current."""
        for sp, fp in zip(self.slow.parameters(), value_head.parameters()):
            sp.data.mul_(self.decay).add_(fp.data, alpha=1 - self.decay)

    def predict(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.slow.predict(h_t)

    def dist(self, h_t: torch.Tensor) -> TwoHotDist:
        return self.slow.dist(h_t)


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

    assert values.shape == (B, T + 1), \
        f"values must be (B, T+1)=({B}, {T+1}), got {tuple(values.shape)}"
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
    action_dim: Union[int, torch.Tensor] = 1,
    entropy_coef: float = 3e-4,
) -> torch.Tensor:
    """PMPO policy loss (TD-MPC2 Eq. 3).

    L = (1-α)/|D-| Σ_{D-} lp  -  α/|D+| Σ_{D+} lp  +  β/N Σ KL[π ∥ π_prior]

    D+ = {i : A_i > 0}, D- = {i : A_i ≤ 0}.  With α=0.5, the total weight
    on D+ and D- is equal regardless of how many samples fall in each set,
    so even a few negative-advantage samples get proportional influence.

    Args:
        log_probs: (N,) log π(a|s) under current policy
        advantages: (N,) advantage estimates (raw, not normalized)
        log_probs_prior: (N,) log π_prior(a|s) under frozen behavioral prior
        alpha: weight on D+ (positive advantage) term
        beta: KL regularization coefficient
        action_dim: valid action dims for normalizing log_probs (int or (N,) tensor)
        entropy_coef: additional entropy regularization coefficient

    Returns:
        scalar loss
    """
    # Normalize log_probs by valid action dim count for scale-invariance.
    if isinstance(action_dim, torch.Tensor):
        lp = log_probs / action_dim.clamp(min=1).float()
    else:
        lp = log_probs / max(action_dim, 1)

    # Split into D+ (positive advantage) and D- (non-positive advantage).
    # NOTE: advantages should be pre-centered (e.g. per-timestep) by the caller.
    pos_mask = (advantages > 0).float()
    neg_mask = 1.0 - pos_mask
    n_pos = pos_mask.sum().clamp(min=1)
    n_neg = neg_mask.sum().clamp(min=1)

    # (1-α)/|D-| Σ_{D-} lp  — push lp DOWN for bad actions
    # -α/|D+| Σ_{D+} lp     — push lp UP for good actions
    policy_loss = (1 - alpha) / n_neg * (lp * neg_mask).sum() \
                - alpha / n_pos * (lp * pos_mask).sum()

    # Entropy regularization (not in original PMPO, but useful for continuous actions).
    entropy_loss = entropy_coef * lp.mean()

    # KL to behavioral prior: β/N Σ (lp - lp_prior)
    kl_loss = torch.tensor(0.0, device=log_probs.device)
    if log_probs_prior is not None and beta > 0:
        if isinstance(action_dim, torch.Tensor):
            lp_prior = log_probs_prior / action_dim.clamp(min=1).float()
        else:
            lp_prior = log_probs_prior / max(action_dim, 1)
        kl_loss = (lp - lp_prior).mean()

    return policy_loss + entropy_loss + beta * kl_loss


def ppo_loss(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
    entropy_coef: float = 3e-4,
    action_dim: Union[int, torch.Tensor] = 1,
) -> torch.Tensor:
    """Clipped PPO surrogate loss for continuous actions.

    L = -mean(min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)) + entropy_coef * mean(lp)

    Advantages are normalized to zero mean / unit std before use.

    Args:
        log_probs: (N,) log π_θ(a|s) under current (updated) policy
        log_probs_old: (N,) log π_old(a|s) fixed from rollout sampling
        advantages: (N,) A_t = R^λ_t - v_t
        eps_clip: clipping radius ε
        entropy_coef: entropy regularization coefficient
        action_dim: scalar or (N,) per-sample valid dim count for normalization
    """
    if isinstance(action_dim, torch.Tensor):
        norm = action_dim.clamp(min=1).float()
    else:
        norm = max(action_dim, 1)
    lp = log_probs / norm

    adv_std = advantages.std()
    adv_norm = (advantages - advantages.mean()) / adv_std.clamp(min=1e-8)
    adv_norm = adv_norm.detach()

    ratio = torch.exp(log_probs - log_probs_old.detach())
    surr1 = ratio * adv_norm
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = entropy_coef * lp.mean()
    return policy_loss + entropy_loss


class ReturnEMA:
    """EMA of the 5th-to-95th percentile return range (Dreamer v3, Eq. 7).

    Provides a robust normalization scale S that is insensitive to outliers
    and smoothed over time.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.S = 1.0  # initial scale (will be overwritten on first update)
        self._initialized = False

    def update(self, returns: torch.Tensor) -> float:
        """Update and return the current scale S = EMA(Per95 - Per5)."""
        r = returns.detach().float()
        per5 = torch.quantile(r, 0.05).item()
        per95 = torch.quantile(r, 0.95).item()
        raw_range = per95 - per5
        if not self._initialized:
            self.S = raw_range
            self._initialized = True
        else:
            self.S = self.decay * self.S + (1 - self.decay) * raw_range
        return max(1.0, self.S)


def reinforce_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy_coef: float = 3e-4,
    action_dim: Union[int, torch.Tensor] = 1,
    return_scale: float = 1.0,
) -> torch.Tensor:
    """REINFORCE policy gradient (Dreamer v3 style, Eq. 6).

    L = -mean(sg(advantages) / max(1, S) * log π / action_dim) + η * mean(lp)

    Advantages are normalized by a percentile-based range S (computed
    externally via ReturnEMA) rather than z-score, making the gradient
    robust to outlier returns.

    Args:
        log_probs: (N,) log π(a|s)
        advantages: (N,) A_t = R^λ_t - v_t
        entropy_coef: entropy regularization coefficient
        action_dim: scalar or (N,) per-sample valid dim count for normalization
        return_scale: S from ReturnEMA.update() — percentile range for normalization
    """
    if isinstance(action_dim, torch.Tensor):
        lp = log_probs / action_dim.clamp(min=1).float()
    else:
        lp = log_probs / max(action_dim, 1)

    adv_norm = advantages.detach() / max(1.0, return_scale)

    policy_loss = -(adv_norm * lp).mean()
    entropy_loss = entropy_coef * lp.mean()
    return policy_loss + entropy_loss
