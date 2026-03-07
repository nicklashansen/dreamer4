# Dreamer4 RL Status — 2026-03-07

## TL;DR

Phase 2 (BC + reward) is now training stably on mixed data (expert + mixed-small)
with RMS loss normalization + loss clipping. Phase 3 (imagination RL) showed
promising early results — policy std decreased, advantages had real signal, no
explosion. The current v6 Phase 2 run is the best so far. Once it checkpoints,
feed it to Phase 3.

---

## Currently Running

**Phase 2 v6** on GPU 3 — mixed data, RMS normalization + loss clipping
- Log: `tail -f /data/dreamer4/runs/walker-walk-v6/phase2.log`
- Checkpoints: `/data/dreamer4/runs/walker-walk-v6/phase2/`
- Early results look great (bc=-0.03, std=0.88, rew=0.80 at step 275)

---

## Project Overview

- **Repo**: `/data/dreamer4/dreamer4/` on branch `RL-dev`
- **Conda env**: `~/miniforge3/envs/dreamer4/` (Python 3.10, PyTorch 2.8.0+cu128)
- **Pre-trained checkpoints**: `../logs/tokenizer.pt`, `../logs/dynamics.pt`
- **Data**: `/public/dreamer4/` — expert, mixed-small, mixed-large (+ `-shards/` dirs)

### Key files
| File | Purpose |
|------|---------|
| `agent.py` | PolicyHead (TanhNormal), RewardHead, ValueHead (TwoHotDist), lambda returns, PMPO loss |
| `train_agent.py` | Phase 2: BC + reward + dynamics finetuning (RMS norm + loss clipping) |
| `train_imagination.py` | Phase 3: imagined rollouts, TD(lambda) + PMPO policy gradient |
| `model.py` | Tokenizer, Dynamics, attention masks |
| `test_components.py` | 47 smoke tests for all RL components |
| `test_action_sensitivity.py` | Verifies dynamics h_t responds to different actions |
| `plot_training.py` | Generate training curve plots from log files |

---

## Key Fixes This Session

### 1. RMS Loss Normalization (Dreamer v4 paper)
"We normalize all loss terms by running estimates of their RMS."
Each loss is divided by `max(1.0, EMA(|loss|))` so BC (which can be ~1400 at init)
doesn't dominate dynamics (~0.01) and reward (~4.6). Without this, mixed data
caused BC to dominate gradients and corrupt the dynamics model.

### 2. Loss Clipping
Each loss is clamped to `max(5.0, 5 * rms_ema)` before backprop. This prevents
single outlier batches from causing catastrophic collapse. In v5, training was
going well (bc=0.22, std=1.08 at step 850) then a single bad batch spiked bc
to 37.7 and reset std to 7.3. Loss clipping prevents this.

### 3. Mixed Data for Reward Diversity
Expert-only data has near-constant reward (~2.0), so the reward model learned
to predict constant 1.718 — useless for Phase 3. Mixed-small data has reward
range 0.006-2.0 (mean 1.13, std 0.74), giving the reward model real signal.

### 4. wm_agent Attention Mask Reverted
Tried a JAX-reference firewall mask (non-agent tokens can't see agent tokens).
Broke Phase 3 because the dynamics was fine-tuned with all-to-all attention.
Reverted to `torch.ones((S,S))` — **do NOT change this mask**.

---

## How to Read Phase 3 Metrics

| Metric | Good sign | Bad sign |
|--------|-----------|----------|
| `r` (mean reward) | Increasing over training | Flat or constant |
| `pi` (policy loss) | Negative and decreasing | Exploding positive |
| `val` (value loss) | Decreasing toward ~0.5-1.0 | Not converging |
| `adv+` (frac positive) | 0.3-0.7 (mixed signal) | Stuck at 0 or 1 |
| `std` (policy std) | Gradually decreasing to 0.3-1.0 | Stuck at max (7.4) or collapsed to 0 |
| `R` (lambda return) | Increasing | Flat |

---

## Data

| Source | Path | Shards | walker-walk reward |
|--------|------|--------|--------------------|
| expert | `/public/dreamer4/expert` | `/public/dreamer4/expert-shards` | mean=1.96, nearly constant |
| mixed-small | `/public/dreamer4/mixed-small` | `/public/dreamer4/mixed-small-shards` | mean=1.13, std=0.74, good diversity |
| mixed-large | `/public/dreamer4/mixed-large` | `/public/dreamer4/mixed-large-shards` | Not yet tested (780 segments) |

---

## Run Commands

### Run tests
```bash
cd /data/dreamer4/dreamer4
python test_components.py
```

### Phase 2 — mixed data with RMS norm + loss clipping (CURRENT BEST)
```bash
cd /data/dreamer4/dreamer4
mkdir -p /data/dreamer4/runs/walker-walk-v7/phase2

PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_agent.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --dynamics_ckpt ../logs/dynamics.pt \
    --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
    --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards \
    --tasks walker-walk \
    --space_mode wm_agent \
    --batch_size 16 \
    --seq_len 32 \
    --max_steps 2000 \
    --log_every 25 \
    --save_every 500 \
    --num_workers 4 \
    --lr 3e-4 \
    --lr_dynamics 1e-4 \
    --ckpt_dir /data/dreamer4/runs/walker-walk-v7/phase2 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-v7/phase2.log
```

### Phase 3 — imagination RL (from Phase 2 checkpoint)
```bash
cd /data/dreamer4/dreamer4
mkdir -p /data/dreamer4/runs/walker-walk-v7/phase3

PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_imagination.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --phase2_ckpt /data/dreamer4/runs/walker-walk-v7/phase2/step_0001000.pt \
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
    --ckpt_dir /data/dreamer4/runs/walker-walk-v7/phase3 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-v7/phase3.log
```

### Plot training curves
```bash
cd /data/dreamer4/dreamer4
python plot_training.py /data/dreamer4/runs/walker-walk-v6/
```

### Action sensitivity test
```bash
cd /data/dreamer4/dreamer4
CUDA_VISIBLE_DEVICES=0 python test_action_sensitivity.py \
    /data/dreamer4/runs/walker-walk-v6/phase2/step_0001000.pt \
    ../logs/tokenizer.pt
```

---

## Architecture Notes

### Attention mask: `wm_agent`
- `wm_agent` = `torch.ones((S,S))` — all tokens attend to all tokens
- Pre-trained dynamics used `wm_agent_isolated`, Phase 2 overrides to `wm_agent`
- **Do NOT change this mask** — model was fine-tuned with it

### Policy: TanhNormal (continuous)
- Pre-tanh Gaussian, actions = tanh(x), Jacobian-corrected log_prob
- `rsample_with_log_prob()` for numerically stable sampling (TD-MPC2 style)
- Bounded std: log_std in [-10, 2] via tanh (std range [4.5e-5, 7.389])

### Reward/Value: TwoHotDist in symlog space
- 101 bins, symlog range [-10, 10]

### PMPO loss
- Advantage-weighted PG (not sign-split — that explodes for continuous)
- Advantages normalized to zero mean / unit std
- Entropy reg + KL to behavioral prior

### RMS normalization + loss clipping (train_agent.py)
- Each loss divided by `max(1.0, EMA(|loss|))` with decay=0.99
- Each loss clamped to `max(5.0, 5 * EMA)` before backprop
- Prevents any single loss from dominating AND prevents outlier batches from collapsing training

### Hyperparameters
- gamma=0.997, lambda=0.95, alpha=0.5, beta=0.3, entropy_coef=3e-4
- Phase 2: lr=3e-4 (heads), lr_dynamics=1e-4, batch=16, seq_len=32
- Phase 3: lr_policy=3e-5, lr_value=3e-4, batch=8, horizon=16, grad_clip=1.0
- MTP L=8, action_dim=16

---

## Existing Checkpoints

| Run | Phase | Path | Status |
|-----|-------|------|--------|
| v6 | Phase 2 | `/data/dreamer4/runs/walker-walk-v6/phase2/` | Running (best so far) |
| v5 | Phase 2 | `/data/dreamer4/runs/walker-walk-v5/phase2/step_0000500.pt` | OK but spiked at step 875 |
| v5 | Phase 3 | `/data/dreamer4/runs/walker-walk-v5/phase3/` | Promising early results |
| v3 | Phase 2 | `/data/dreamer4/runs/walker-walk-v3/phase2/step_0001000.pt` | Good BC, constant reward |

Older broken runs: `walker-walk/`, `walker-walk-v2/`, `walker-walk-v4/`, `cartpole-balance/`

---

## Run History

| Run | What changed | Result |
|-----|-------------|--------|
| v1-v2 | `wm_agent_isolated` | Action-blind, no RL signal |
| v3 | `wm_agent`, expert-only | BC great, reward constant ~1.718 |
| v4 | Mixed data, no RMS norm | BC diverged (261+), dynamics degraded |
| v4-frozen | Mixed data, lr_dynamics=0 | BC oscillated wildly (2 to 394K) |
| v5 | Mixed data + RMS norm | Stable but BC spiked at step 875, collapsing policy |
| v6 | Mixed data + RMS norm + loss clipping | Stable, bc=-0.03 std=0.88 at step 275 (running) |

---

## Deferred Items
- Action discretization (tried, reverted — want continuous working first)
- mixed-large data (780 segments, much more diverse)
- JAX reference corrections: MTP default L=2, searchsorted for two-hot
- Adjusting symlog bin ranges per JAX ref (reward [-3,3], value [-8,8])
- RMS normalization for Phase 3 (train_imagination.py) — not yet added
