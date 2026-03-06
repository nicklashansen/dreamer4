#!/usr/bin/env python3
"""Level 0: Smoke tests for RL components.

Run: python dreamer4/test_components.py

Tests shape correctness, gradient flow, and numerical sanity for all new
agent components (PolicyHead, RewardHead, ValueHead, TwoHotDist, lambda
returns, PMPO loss).
"""
import sys
import torch
torch.manual_seed(42)

from agent import (
    symlog, symexp, TwoHotDist,
    PolicyHead, RewardHead, ValueHead,
    compute_lambda_returns, pmpo_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, T, D = 2, 8, 512
ACTION_DIM = 6
MTP_L = 8
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    else:
        PASS += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return cond


def test_symlog_symexp():
    print("\n--- symlog / symexp ---")
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], device=DEVICE)
    roundtrip = symexp(symlog(x))
    check("roundtrip symexp(symlog(x)) ≈ x", torch.allclose(roundtrip, x, atol=1e-4),
          f"max err={( roundtrip - x).abs().max().item():.6f}")

    check("symlog(0) = 0", symlog(torch.tensor(0.0)).item() == 0.0)
    check("symexp(0) = 0", symexp(torch.tensor(0.0)).item() == 0.0)
    check("symlog monotonic", (symlog(x)[1:] >= symlog(x)[:-1]).all().item())


def test_twohot_dist():
    print("\n--- TwoHotDist ---")
    num_bins = 255
    logits = torch.randn(B, T, num_bins, device=DEVICE, requires_grad=True)
    dist = TwoHotDist(logits)

    mean = dist.mean
    check("mean shape", mean.shape == (B, T), f"got {mean.shape}")
    check("mean finite", mean.isfinite().all().item())

    targets = torch.randn(B, T, device=DEVICE) * 5.0
    lp = dist.log_prob(targets)
    check("log_prob shape", lp.shape == (B, T), f"got {lp.shape}")
    check("log_prob ≤ 0", (lp <= 0).all().item(), f"max={lp.max().item():.4f}")

    loss = -lp.mean()
    loss.backward()
    check("gradient flows to logits", logits.grad is not None and logits.grad.abs().sum() > 0)

    # check that concentrated logits give mean close to bin center
    logits2 = torch.zeros(1, 1, num_bins, device=DEVICE)
    logits2[0, 0, num_bins // 2] = 100.0  # spike at center bin (symlog=0)
    dist2 = TwoHotDist(logits2)
    check("concentrated dist mean ≈ 0", abs(dist2.mean.item()) < 0.01,
          f"got {dist2.mean.item():.4f}")


def test_policy_head():
    print("\n--- PolicyHead ---")
    head = PolicyHead(d_model=D, action_dim=ACTION_DIM, mtp_length=MTP_L).to(DEVICE)
    h_t = torch.randn(B, T, D, device=DEVICE)

    mean, std = head(h_t)
    check("mean shape", mean.shape == (B, T, MTP_L, ACTION_DIM), f"got {mean.shape}")
    check("std shape", std.shape == (B, T, MTP_L, ACTION_DIM), f"got {std.shape}")
    check("mean in [-1,1]", (mean.abs() <= 1.0 + 1e-6).all().item(),
          f"max={mean.abs().max().item():.4f}")
    check("std > 0", (std > 0).all().item(), f"min={std.min().item():.6f}")

    actions = head.sample(h_t, step=0)
    check("sample shape", actions.shape == (B, T, ACTION_DIM), f"got {actions.shape}")
    check("sample in [-1,1]", (actions.abs() <= 1.0 + 1e-6).all().item())

    lp = head.log_prob(h_t, actions, step=0)
    check("log_prob shape", lp.shape == (B, T), f"got {lp.shape}")
    check("log_prob finite", lp.isfinite().all().item())

    # gradient flow
    loss = -lp.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    check("gradient flows", has_grad)


def test_reward_head():
    print("\n--- RewardHead ---")
    head = RewardHead(d_model=D, mtp_length=MTP_L).to(DEVICE)
    h_t = torch.randn(B, T, D, device=DEVICE, requires_grad=True)

    logits = head(h_t)
    check("logits shape", logits.shape == (B, T, MTP_L, 255), f"got {logits.shape}")

    pred = head.predict(h_t, step=0)
    check("predict shape", pred.shape == (B, T), f"got {pred.shape}")
    check("predict finite", pred.isfinite().all().item())

    targets = torch.randn(B, T, device=DEVICE)
    loss = head.loss(h_t, targets)
    check("loss scalar", loss.dim() == 0)
    check("loss finite", loss.isfinite().item(), f"loss={loss.item():.4f}")

    loss.backward()
    check("gradient flows to h_t", h_t.grad is not None and h_t.grad.abs().sum() > 0)


def test_value_head():
    print("\n--- ValueHead ---")
    head = ValueHead(d_model=D).to(DEVICE)
    h_t = torch.randn(B, T, D, device=DEVICE, requires_grad=True)

    pred = head.predict(h_t)
    check("predict shape", pred.shape == (B, T), f"got {pred.shape}")

    targets = torch.randn(B, T, device=DEVICE) * 10.0
    loss = head.loss(h_t, targets)
    check("loss scalar", loss.dim() == 0)
    check("loss finite", loss.isfinite().item(), f"loss={loss.item():.4f}")

    loss.backward()
    check("gradient flows to h_t", h_t.grad is not None and h_t.grad.abs().sum() > 0)


def test_lambda_returns():
    print("\n--- compute_lambda_returns ---")
    rewards = torch.ones(B, T, device=DEVICE)
    values = torch.zeros(B, T + 1, device=DEVICE)
    gamma, lam = 0.99, 0.95

    returns = compute_lambda_returns(rewards, values, gamma=gamma, lam=lam)
    check("returns shape", returns.shape == (B, T), f"got {returns.shape}")
    check("returns finite", returns.isfinite().all().item())

    # with constant reward=1 and value=0, returns should decrease from front to back
    check("returns[0] > returns[-1]", returns[0, 0].item() > returns[0, -1].item(),
          f"first={returns[0,0].item():.4f} last={returns[0,-1].item():.4f}")

    # last return should be approximately reward (since value bootstrap = 0)
    check("last return ≈ 1.0", abs(returns[0, -1].item() - 1.0) < 0.1,
          f"got {returns[0,-1].item():.4f}")

    # with continuation = 0 at step 4, returns after that should be independent
    cont = torch.ones(B, T, device=DEVICE)
    cont[:, 4] = 0.0
    returns_term = compute_lambda_returns(rewards, values, gamma=gamma, lam=lam, continuation=cont)
    check("terminal resets returns", returns_term[0, 3].item() < returns[0, 3].item(),
          f"with_term={returns_term[0,3].item():.4f} without={returns[0,3].item():.4f}")


def test_pmpo_loss():
    print("\n--- pmpo_loss ---")
    N = 100
    log_probs = torch.randn(N, device=DEVICE, requires_grad=True)
    advantages = torch.randn(N, device=DEVICE)
    log_probs_prior = torch.randn(N, device=DEVICE)

    loss = pmpo_loss(log_probs, advantages, log_probs_prior, alpha=0.5, beta=0.3)
    check("loss scalar", loss.dim() == 0)
    check("loss finite", loss.isfinite().item(), f"loss={loss.item():.4f}")

    loss.backward()
    check("gradient flows", log_probs.grad is not None and log_probs.grad.abs().sum() > 0)

    # without prior
    log_probs2 = torch.randn(N, device=DEVICE, requires_grad=True)
    loss_no_prior = pmpo_loss(log_probs2, advantages, None, alpha=0.5, beta=0.3)
    check("loss without prior finite", loss_no_prior.isfinite().item())


def test_dynamics_integration():
    """Test that agent.py heads work with Dynamics model agent token output."""
    print("\n--- Dynamics Integration ---")
    from model import Dynamics

    dyn = Dynamics(
        d_model=D, d_bottleneck=32, d_spatial=64, n_spatial=8,
        n_register=4, n_agent=1, n_heads=8, depth=4, k_max=8,
        space_mode="wm_agent_isolated",
    ).to(DEVICE)

    packed = torch.randn(B, T, 8, 64, device=DEVICE)
    actions = torch.randn(B, T, 16, device=DEVICE).clamp(-1, 1)
    step_idxs = torch.full((B, T), 3, device=DEVICE, dtype=torch.long)
    signal_idxs = torch.full((B, T), 7, device=DEVICE, dtype=torch.long)

    x1_hat, h_t = dyn(actions, step_idxs, signal_idxs, packed)
    check("h_t not None", h_t is not None)
    check("h_t shape", h_t.shape == (B, T, 1, D), f"got {h_t.shape}")

    # squeeze agent dim and feed to heads
    h = h_t.squeeze(2)  # (B, T, D)

    policy = PolicyHead(d_model=D, action_dim=ACTION_DIM).to(DEVICE)
    reward = RewardHead(d_model=D).to(DEVICE)
    value = ValueHead(d_model=D).to(DEVICE)

    a = policy.sample(h)
    check("policy from dynamics h_t", a.shape == (B, T, ACTION_DIM))

    r = reward.predict(h)
    check("reward from dynamics h_t", r.shape == (B, T))

    v = value.predict(h)
    check("value from dynamics h_t", v.shape == (B, T))

    # full backward through everything
    loss = -policy.log_prob(h, a).mean() + reward.loss(h, r.detach()) + value.loss(h, v.detach())
    loss.backward()
    dyn_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in dyn.parameters())
    check("gradient flows through dynamics", dyn_has_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Testing with B={B}, T={T}, D={D}, ACTION_DIM={ACTION_DIM}, MTP_L={MTP_L}")

    test_symlog_symexp()
    test_twohot_dist()
    test_policy_head()
    test_reward_head()
    test_value_head()
    test_lambda_returns()
    test_pmpo_loss()
    test_dynamics_integration()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
        sys.exit(0)
