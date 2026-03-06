#!/bin/bash
# Full training pipeline for cartpole-balance: Phase 2 → Phase 3 → Plot
set -euo pipefail

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="/data/dreamer4/runs/cartpole-balance"
mkdir -p "$CKPT_DIR"

echo "=========================================="
echo "Phase 2: Agent Finetuning (cartpole-balance)"
echo "=========================================="

python train_agent.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --dynamics_ckpt ../logs/dynamics.pt \
    --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
    --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards \
    --tasks_json ../tasks.json \
    --tasks cartpole-balance \
    --batch_size 16 \
    --seq_len 32 \
    --num_workers 4 \
    --max_steps 2000 \
    --log_every 50 \
    --save_every 500 \
    --lr 3e-4 \
    --lr_dynamics 1e-4 \
    --bootstrap_start 500 \
    --self_fraction 0.25 \
    --grad_clip 1.0 \
    --ckpt_dir "$CKPT_DIR/phase2" \
    --wandb_project "" \
    2>&1 | tee "$CKPT_DIR/phase2.log"

echo ""
echo "=========================================="
echo "Phase 3: Imagination RL (cartpole-balance)"
echo "=========================================="

python train_imagination.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --phase2_ckpt "$CKPT_DIR/phase2/final.pt" \
    --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
    --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards \
    --tasks_json ../tasks.json \
    --tasks cartpole-balance \
    --batch_size 8 \
    --context_len 16 \
    --horizon 16 \
    --num_workers 4 \
    --max_steps 2000 \
    --log_every 50 \
    --save_every 500 \
    --lr_policy 3e-5 \
    --lr_value 3e-4 \
    --gamma 0.997 \
    --lam 0.95 \
    --alpha 0.5 \
    --beta 0.3 \
    --grad_clip 1.0 \
    --ckpt_dir "$CKPT_DIR/phase3" \
    --wandb_project "" \
    2>&1 | tee "$CKPT_DIR/phase3.log"

echo ""
echo "=========================================="
echo "Plotting results"
echo "=========================================="

python plot_training.py "$CKPT_DIR"

echo "Done! Results in $CKPT_DIR/"
