#!/bin/bash
# Launch the Dreamer4 interactive web demo.
# Access at http://localhost:7860 (or SSH tunnel if remote).
#
# With autopilot (policy plays the game):
#   ./run_demo.sh --agent_ckpt /path/to/phase3/latest.pt --task walker-walk

python interactive.py \
  --data_dir /public/dreamer4/expert \
  --frames_dir /public/dreamer4/expert-shards \
  --tasks_json /data/dreamer4/tasks.json \
  --tokenizer_ckpt /data/dreamer4/logs/tokenizer.pt \
  --dynamics_ckpt /data/dreamer4/logs/dynamics.pt \
  "$@"
