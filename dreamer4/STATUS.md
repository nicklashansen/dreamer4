# Dreamer4 RL Status — 2026-03-07

## TL;DR

Phase 2 (BC + reward) works on expert-only data but the reward model learns a
constant (~1.718) because expert data has uniform high reward. Phase 3 (imagination
RL) then explodes because the reward model gives no signal. Training Phase 2 on
mixed data fixes reward diversity but breaks BC (TanhNormal can't fit multimodal
actions, std maxes out, BC loss diverges, corrupts dynamics).

**Blocking issue**: the reward head needs diverse data, but BC needs clean expert data.
We need to decouple them — see "Next Steps" below.

---

## Project Overview

- **Repo**: `/data/dreamer4/dreamer4/` on branch `RL-dev`
- **Conda env**: `~/miniforge3/envs/dreamer4/` (Python 3.10, PyTorch 2.8.0+cu128)
- **Pre-trained checkpoints**: `../logs/tokenizer.pt`, `../logs/dynamics.pt`
- **Data**: `/public/dreamer4/` — expert, mixed-small, mixed-large (+ corresponding `-shards/` dirs)

### Key files
| File | Purpose |
|------|---------|
| `agent.py` | PolicyHead (TanhNormal), RewardHead, ValueHead (TwoHotDist), lambda returns, PMPO loss |
| `train_agent.py` | Phase 2: BC + reward prediction + dynamics finetuning |
| `train_imagination.py` | Phase 3: imagined rollouts, TD(lambda) + PMPO policy gradient |
| `train_dynamics.py` | Dynamics pre-training (unchanged, but exports helpers) |
| `model.py` | Tokenizer, Dynamics, attention masks |
| `test_components.py` | 47 smoke tests for all RL components |
| `test_action_sensitivity.py` | Verifies dynamics h_t responds to different actions |

---

## What Works

### Phase 2 on expert-only data (walker-walk-v3)
Converges well over 1000+ steps:
```
step 0000000 | bc=1435.9  rew=4.62  | std=0.019 r_pred=0.016 r_tgt=1.997
step 0000500 | bc=-3.20   rew=0.67  | std=0.637 r_pred=1.718 r_tgt=1.997
step 0001000 | bc=-3.45   rew=0.67  | std=0.500 r_pred=1.714 r_tgt=1.988
```
- BC loss decreases nicely, policy learns expert actions
- But `r_pred=1.718` and `r_tgt=1.997` are CONSTANT — reward model just learns "reward is always ~1.7"
- Checkpoints at `/data/dreamer4/runs/walker-walk-v3/phase2/step_0001000.pt`

### Component tests
All 47 tests pass:
```bash
cd /data/dreamer4/dreamer4
python test_components.py
```

---

## What Doesn't Work

### Phase 3 (imagination RL) — policy loss explodes
From v3 Phase 2 checkpoint:
```
step 0000000 | pi=-0.01   val=4.61  | adv+=1.00 r=1.716 V=-0.009
step 0000050 | pi=1.67    val=3.45  | adv+=1.00 r=1.716 V=41.161
step 0000100 | pi=213917  val=1.73  | adv+=1.00 r=1.716 V=23.087
```
- `adv+=1.00` — ALL advantages are positive (no negative signal)
- `r=1.716` — reward prediction is constant regardless of action
- Policy loss explodes to ~500K by step 225
- Root cause: **reward model predicts constant, so no signal for policy improvement**

### Phase 2 on mixed data (walker-walk-v4) — BC diverges
Adding mixed-small data for reward diversity:
```
step 0000000 | bc=1384.9  rew=4.62  | std=0.019 r_pred=0.017 r_tgt=1.619
step 0000050 | bc=2.52    rew=4.21  | std=7.389 r_pred=0.043 r_tgt=1.404  <-- std maxed out!
step 0000300 | bc=261.2   rew=10.95 | std=7.389 r_pred=1.718 r_tgt=1.307  <-- diverging!
```
- TanhNormal can't fit multimodal actions (expert + suboptimal), maxes std at exp(2)=7.389
- BC loss dominates total loss, corrupting dynamics (flow: 0.03 -> 0.20)
- Reward model also degrades (reverts to constant prediction)

---

## Data

| Source | Path | walker-walk reward stats |
|--------|------|------------------------|
| expert | `/public/dreamer4/expert` | mean=1.963, std=0.230, p10=1.990 (nearly constant ~2.0) |
| mixed-small | `/public/dreamer4/mixed-small` | mean=1.128, std=0.737, p10=0.099 (good diversity) |
| mixed-large | `/public/dreamer4/mixed-large` | Not yet tested |

Shards dirs (for frame loading): `/public/dreamer4/{expert,mixed-small,mixed-large}-shards/`

---

## Next Steps (Blocking Issue)

The reward head needs diverse data to learn good-vs-bad, but BC needs expert-only data
to avoid diverging. Two approaches (pick one):

### Option A: Two-stage Phase 2 (recommended — simplest)
1. Run Phase 2 on expert-only data (already works, v3 checkpoints exist)
2. Write a small "Phase 2b" script that loads the Phase 2 checkpoint, freezes
   dynamics + policy, and fine-tunes ONLY the reward head on mixed data

### Option B: Separate data loaders in Phase 2
Modify `train_agent.py` to accept separate `--bc_data_dirs` and `--reward_data_dirs`.
BC loss uses expert loader, reward loss uses mixed loader. More elegant but bigger change.

### After reward is fixed
Re-run Phase 3 imagination RL. With a reward model that gives varied predictions,
advantages will have real signal (not all-positive), and the policy should improve.

---

## Architecture Notes

### Attention mask: `wm_agent`
- `wm_agent` = `torch.ones((S,S))` — all tokens attend to all tokens
- The pretrained dynamics used `wm_agent_isolated` (agent tokens only see agent tokens)
- Phase 2 overrides to `wm_agent` so agent tokens can see actions + spatial state
- **Do NOT change the wm_agent mask** — the Phase 2 model was fine-tuned with it.
  We tried a JAX-reference firewall mask and it broke Phase 3 (dynamics output garbage
  with attention patterns it wasn't trained on).

### Policy: TanhNormal (continuous)
- Pre-tanh Gaussian, actions = tanh(x), log_prob includes Jacobian correction
- `rsample_with_log_prob()` avoids lossy atanh round-trip (TD-MPC2 style)
- Bounded std: log_std in [-10, 2] via tanh (std range [4.5e-5, 7.389])
- `log_prob_per_dim()` for masked BC loss (some action dims are inactive)

### Reward/Value: TwoHotDist in symlog space
- 101 bins, symlog range [-10, 10]
- Target encoded as two-hot between nearest bins
- Mean = symexp(weighted sum of bin centers)

### PMPO loss (continuous actions)
- Advantage-weighted policy gradient (not sign-split — that explodes for continuous)
- Advantages normalized to zero mean / unit std
- Entropy regularization: `entropy_coef * mean(log_prob / action_dim)`
- KL to behavioral prior: `beta * mean((log_pi - log_pi_prior) / action_dim)`

### Hyperparameters
- gamma=0.997, lambda=0.95, alpha=0.5, beta=0.3, entropy_coef=3e-4
- Phase 2: lr=3e-4 (heads), lr_dynamics=1e-4, batch=16, seq_len=32
- Phase 3: lr_policy=3e-5, lr_value=3e-4, batch=8, horizon=16, grad_clip=1.0
- MTP L=8, action_dim=16

---

## Run Commands

### Run tests
```bash
cd /data/dreamer4/dreamer4
python test_components.py
```

### Phase 2 — expert-only (WORKS)
```bash
cd /data/dreamer4/dreamer4
mkdir -p /data/dreamer4/runs/walker-walk-v5/phase2

PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_agent.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --dynamics_ckpt ../logs/dynamics.pt \
    --data_dirs /public/dreamer4/expert \
    --frame_dirs /public/dreamer4/expert-shards \
    --tasks walker-walk \
    --space_mode wm_agent \
    --batch_size 16 \
    --seq_len 32 \
    --max_steps 2000 \
    --log_every 50 \
    --save_every 500 \
    --num_workers 4 \
    --lr 3e-4 \
    --lr_dynamics 1e-4 \
    --ckpt_dir /data/dreamer4/runs/walker-walk-v5/phase2 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-v5/phase2.log
```

### Phase 2 — mixed data (BROKEN — DO NOT USE until BC/reward decoupled)
```bash
cd /data/dreamer4/dreamer4
# This adds --data_dirs and --frame_dirs for both expert + mixed-small
# BC loss diverges because TanhNormal can't fit multimodal action distribution
PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_agent.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --dynamics_ckpt ../logs/dynamics.pt \
    --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
    --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards \
    --tasks walker-walk \
    --space_mode wm_agent \
    --batch_size 16 --seq_len 32 --max_steps 2000 --log_every 50 --save_every 500 \
    --num_workers 4 --lr 3e-4 --lr_dynamics 1e-4 \
    --ckpt_dir /data/dreamer4/runs/walker-walk-vX/phase2 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-vX/phase2.log
```

### Phase 3 — imagination RL (BROKEN — needs fixed reward model first)
```bash
cd /data/dreamer4/dreamer4
mkdir -p /data/dreamer4/runs/walker-walk-v5/phase3

PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=3 python train_imagination.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --phase2_ckpt /data/dreamer4/runs/walker-walk-v5/phase2/step_0001000.pt \
    --tasks walker-walk \
    --space_mode wm_agent \
    --batch_size 8 \
    --context_len 16 \
    --horizon 16 \
    --max_steps 2000 \
    --log_every 25 \
    --save_every 500 \
    --num_workers 4 \
    --gamma 0.997 --lam 0.95 \
    --alpha 0.5 --beta 0.3 --entropy_coef 3e-4 \
    --lr_policy 3e-5 --lr_value 3e-4 \
    --ckpt_dir /data/dreamer4/runs/walker-walk-v5/phase3 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-v5/phase3.log
```

### Action sensitivity test
```bash
cd /data/dreamer4/dreamer4
CUDA_VISIBLE_DEVICES=0 python test_action_sensitivity.py \
    /data/dreamer4/runs/walker-walk-v3/phase2/step_0001000.pt \
    ../logs/tokenizer.pt
```

---

## Existing Checkpoints

| Run | Phase | Path | Status |
|-----|-------|------|--------|
| walker-walk-v3 | Phase 2 | `/data/dreamer4/runs/walker-walk-v3/phase2/step_0001000.pt` | Good BC, bad reward (constant) |
| walker-walk-v3 | Phase 3 | (none saved) | Exploded |
| walker-walk-v4 | Phase 2 | `/data/dreamer4/runs/walker-walk-v4/phase2/` | Diverged (mixed data) |

Older broken runs (used `wm_agent_isolated`, action-blind):
- `/data/dreamer4/runs/walker-walk/`
- `/data/dreamer4/runs/walker-walk-v2/`
- `/data/dreamer4/runs/cartpole-balance/`

---

## Deferred Items
- Action discretization (tried, reverted — want continuous working first)
- JAX reference corrections: MTP default L=2, searchsorted for two-hot
- mixed-large data (780 segments, much more diverse)
- Adjusting symlog bin ranges per JAX ref (reward [-3,3], value [-8,8] vs current [-10,10])
