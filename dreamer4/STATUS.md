# Dreamer4 RL Status — 2026-03-07

## What's Running
- **Phase 2 (walker-walk-v2)** training on GPU 0 — should finish in ~30min
  - Checkpoint dir: `/data/dreamer4/runs/walker-walk-v2/phase2/`
  - Log: `/data/dreamer4/runs/walker-walk-v2/phase2.log`

## Critical Bug Fixed This Session
**`space_mode` must be `"wm_agent"` for RL, NOT the pretrained `"wm_agent_isolated"`.**

With `wm_agent_isolated`, agent tokens only attend to themselves — completely blind
to actions and spatial state. h_t has ZERO action sensitivity. This made all
imagination RL impossible (rewards constant regardless of policy actions).

With `wm_agent`, agent tokens attend to everything. Verified with
`test_action_sensitivity.py` — h_t now responds to actions (2.9% relative diff
after 500 steps of Phase 2 finetuning, growing).

## Other Fixes
- **PMPO → advantage-weighted PG**: sign-split PMPO explodes for continuous actions
  (unbounded log_prob maximization). Replaced with standard advantage-weighted policy
  gradient: `loss = -(normalized_advantages * log_prob).mean()`. Plus entropy
  regularization and KL to behavioral prior.
- **TanhNormal**: proper continuous action density with Jacobian correction
- **action_dim normalization**: log_probs divided by action_dim in policy loss

## What Needs to Happen Next

### 1. Wait for Phase 2 to finish
Check: `tail -f /data/dreamer4/runs/walker-walk-v2/phase2.log`

### 2. Run tests (PMPO was just changed, not yet tested)
```bash
cd /data/dreamer4/dreamer4
python test_components.py
```

### 3. Run action sensitivity test on final Phase 2 checkpoint
```bash
cd /data/dreamer4/dreamer4
CUDA_VISIBLE_DEVICES=0 python test_action_sensitivity.py \
    /data/dreamer4/runs/walker-walk-v2/phase2/final.pt \
    ../logs/tokenizer.pt
```

### 4. Run Phase 3 (imagination RL)
```bash
cd /data/dreamer4/dreamer4
PYTHONUNBUFFERED=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_imagination.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --phase2_ckpt /data/dreamer4/runs/walker-walk-v2/phase2/final.pt \
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
    --ckpt_dir /data/dreamer4/runs/walker-walk-v2/phase3 \
    2>&1 | tee /data/dreamer4/runs/walker-walk-v2/phase3.log
```

### 5. Plot training curves
```bash
cd /data/dreamer4/dreamer4
python plot_training.py /data/dreamer4/runs/walker-walk-v2/
```

## Key Files Changed
- `agent.py` — TanhNormal, pmpo_loss (now advantage-weighted PG + entropy + KL)
- `train_agent.py` — added `--space_mode` arg (default "wm_agent")
- `train_imagination.py` — added `--space_mode` and `--entropy_coef` args
- `test_action_sensitivity.py` — NEW: verifies dynamics responds to actions
- `test_components.py` — updated tests for new API

## Previous Runs (broken, used wm_agent_isolated)
- `/data/dreamer4/runs/walker-walk/` — Phase 2 + Phase 3, action-blind
- `/data/dreamer4/runs/cartpole-balance/` — Phase 2 + Phase 3, action-blind

## Open Questions
- Is the advantage-weighted PG stable? (just changed, not yet tested in training)
- Does Phase 2 with wm_agent produce enough reward sensitivity for Phase 3?
- Entropy_coef=3e-4 may need tuning for continuous actions
