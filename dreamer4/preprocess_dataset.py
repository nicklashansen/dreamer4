# preprocess_dataset.py
import os
from pathlib import Path

import torch
from torchvision.io import read_image
import torch.nn.functional as F

from task_set import TASK_SET


FILEDIR = "/<path>/expert"        # raw data source
OUTDIR = "/<path>/expert-shards"  # preprocessed shards

TARGET_SIZE = 128   # image resolution for training
SHARD_SIZE = 2048   # number of frames per shard


def safe_save_frames(frames: torch.Tensor, out_path: Path) -> bool:
    """
    Safely save {"frames": frames} to out_path:
      - write to a temporary file
      - atomically rename to final path
      - delete temp file if anything goes wrong
    Returns True on success, False on failure.
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        # Ensure parent exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp
        torch.save({"frames": frames}, tmp_path)

        # Atomic rename (works across POSIX filesystems)
        os.replace(tmp_path, out_path)
        print(f"  [OK] Saved shard with {frames.shape[0]} frames to {out_path}")
        return True
    except Exception as e:
        print(f"  [WARN] Failed saving shard {out_path}: {e}")
        # Clean up any partial temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as e2:
            print(f"  [WARN] Failed removing temp file {tmp_path}: {e2}")
        return False


def process_task(task: str):
    task_out_dir = Path(OUTDIR) / task

    # skip if already done
    if any(task_out_dir.glob(f"{task}_shard*.pt")):
        print(f"[{task}] already processed, skipping.")
        return

    task_out_dir.mkdir(parents=True, exist_ok=True)

    shard_frames = []  # list of (N_i, 3, 128, 128) uint8
    shard_idx = 0

    i = 0
    while True:
        png_path = Path(FILEDIR) / f"{task}-{i}.png"
        if not png_path.exists():
            break

        print(f"[{task}] reading {png_path}")
        try:
            frames = read_image(str(png_path))  # (3, 224, 224 * num_frames), uint8
        except Exception as e:
            print(f"  [WARN] Skipping {png_path} (read error): {e}")
            i += 1
            continue

        C, H, W_total = frames.shape
        if H != 224 or W_total % 224 != 0:
            print(f"  [WARN] Skipping {png_path}, unexpected shape {frames.shape}")
            i += 1
            continue

        num_frames = W_total // 224
        if num_frames == 0:
            print(f"  [WARN] Skipping {png_path}, no frames detected")
            i += 1
            continue

        # Split horizontally: (num_frames, 3, 224, 224)
        frames = frames.view(C, 224, num_frames, 224)      # (3, 224, N, 224)
        frames = frames.permute(2, 0, 1, 3)                # (N, 3, 224, 224)

        # Downsample to 128x128 once
        frames_f = frames.to(torch.float32) / 255.0
        frames_f = F.interpolate(
            frames_f,
            size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        frames_u8 = (frames_f.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # (N, 3, 128, 128)

        shard_frames.append(frames_u8)

        # Flush shards as long as we have ≥ SHARD_SIZE frames
        while shard_frames and sum(f.shape[0] for f in shard_frames) >= SHARD_SIZE:
            concat = torch.cat(shard_frames, dim=0)  # (>=SHARD_SIZE, 3, 128, 128)
            to_save, remainder = concat[:SHARD_SIZE], concat[SHARD_SIZE:]
            out_path = task_out_dir / f"{task}_shard{shard_idx:04d}.pt"

            print(f"[{task}] saving shard {shard_idx} with {to_save.shape[0]} frames to {out_path}")
            ok = safe_save_frames(to_save, out_path)
            if not ok:
                # If saving failed, we keep the remainder in memory and just move on.
                # You can also choose to break here if the FS is clearly broken.
                print(f"  [WARN] Continuing after failed save of {out_path} (check disk space/FS).")

            shard_frames = []
            if remainder.shape[0] > 0:
                shard_frames.append(remainder)

            shard_idx += 1

        i += 1

    # Flush remainder at the end
    if shard_frames:
        concat = torch.cat(shard_frames, dim=0)
        out_path = task_out_dir / f"{task}_shard{shard_idx:04d}.pt"
        print(f"[{task}] saving final shard {shard_idx} with {concat.shape[0]} frames to {out_path}")
        safe_save_frames(concat, out_path)


def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    for task in TASK_SET:
        process_task(task)


if __name__ == "__main__":
    main()
