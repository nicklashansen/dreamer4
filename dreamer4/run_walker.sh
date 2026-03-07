#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
DIR=/data/dreamer4/runs/walker-walk

# Wait for Phase 2 to finish (check for final.pt)
echo "Waiting for Phase 2 to complete..."
while [ ! -f "$DIR/phase2/final.pt" ]; do
    sleep 30
done
echo "Phase 2 done!"

echo "Starting Phase 3..."
python train_imagination.py \
    --tokenizer_ckpt ../logs/tokenizer.pt \
    --phase2_ckpt "$DIR/phase2/final.pt" \
    --data_dirs /public/dreamer4/expert /public/dreamer4/mixed-small \
    --frame_dirs /public/dreamer4/expert-shards /public/dreamer4/mixed-small-shards \
    --tasks_json ../tasks.json \
    --tasks walker-walk \
    --batch_size 8 \
    --context_len 16 \
    --horizon 16 \
    --num_workers 4 \
    --max_steps 2000 \
    --log_every 50 \
    --save_every 500 \
    --lr_policy 3e-5 \
    --lr_value 3e-4 \
    --gamma 0.997 --lam 0.95 \
    --alpha 0.5 --beta 0.3 \
    --grad_clip 1.0 \
    --ckpt_dir "$DIR/phase3" \
    --wandb_project "" \
    2>&1 | tee "$DIR/phase3.log"

echo "Phase 3 done! Plotting..."
python plot_training.py "$DIR"
echo "All done! Results in $DIR/"
