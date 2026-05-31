"""Stable-WorldModel format that writes Nicklas-style Dreamer4 data.

Nicklas/Dreamer4 raw layout::

    <root>/
      <task>.pt                  # episode/action/reward and other tabular data
      <task>-0.png               # horizontal strip of RGB frames
      <task>-1.png
      ...

The PNG files store frames concatenated horizontally. For example, if one frame is
``224x224`` and a strip contains 100 frames, then the PNG has shape
``224 x (224 * 100)``. This mirrors the raw dataset format expected by
``preprocess_dataset.py`` in Nicklas Hansen's Dreamer4 implementation.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import make_tensordict
from PIL import Image

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format, validate_write_mode
from stable_worldmodel.data.formats.utils import is_image_column
from stable_worldmodel.data.utils import get_cache_dir


DEFAULT_IMAGE_KEY = "pixels"


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensors/lists/arrays to a NumPy array without changing semantics."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_hwc_uint8(frame: Any) -> np.ndarray:
    """Return one image frame as uint8 HWC.

    Accepted input shapes are ``(H, W, C)`` and ``(C, H, W)``. Float images are
    interpreted as either ``[0, 1]`` or ``[0, 255]`` and converted to uint8.
    """
    arr = _to_numpy(frame)

    if arr.ndim != 3:
        raise ValueError(f"Expected one image frame with 3 dims, got shape {arr.shape}")

    # CHW -> HWC. Prefer this only when the first dim is a plausible channel dim.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Expected image frame with channel dimension 1, 3, or 4; "
            f"got shape {arr.shape}"
        )

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    # Nicklas data is RGB. Drop alpha if needed.
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    return np.ascontiguousarray(arr)


def _stack_tabular(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-episode arrays along the time axis."""
    if not parts:
        return np.asarray([])
    return np.concatenate(parts, axis=0)


def _episode_lengths_from_ids(episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer episode lengths and offsets from a flat episode-id vector."""
    episode = np.asarray(episode).reshape(-1)
    if episode.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int64)

    boundaries = np.flatnonzero(episode[1:] != episode[:-1]) + 1
    offsets = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, [episode.size]))
    lengths = (ends - offsets).astype(np.int32)
    return lengths, offsets


class DreamerFolderDataset(Dataset):
    """Reader for Nicklas-style Dreamer4 raw folders.

    The reader exposes a Stable-WorldModel ``Dataset`` interface on top of a
    single task stored as ``<task>.pt`` plus horizontal frame strips
    ``<task>-<i>.png``.

    Args:
        name: Dataset name under the Stable-WorldModel cache directory.
        path: Direct path to the dataset root. Takes precedence over ``name``.
        task_name: Task prefix. If omitted, the reader auto-detects the only
            ``*.pt`` file in the folder.
        image_key: Name used for the image column returned by ``_load_slice``.
        frame_size: Optional ``(height, width)`` of individual frames inside the
            PNG strips. If omitted, the reader assumes square frames and uses the
            strip height as both height and width, matching Nicklas data.
        frames_per_strip: Optional number of frames in each full PNG strip. If
            omitted, it is inferred from the first strip.
        keys_to_load: Optional subset of columns to return. Include
            ``image_key`` to load frames.
    """

    def __init__(
        self,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
        task_name: str | None = None,
        image_key: str = DEFAULT_IMAGE_KEY,
        frame_size: tuple[int, int] | None = None,
        frames_per_strip: int | None = None,
    ) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            if name is None:
                raise TypeError("DreamerFolderDataset requires either `name` or `path`")
            self.path = Path(cache_dir or get_cache_dir(sub_folder="datasets")) / name

        self.task_name = task_name or self._detect_task_name()
        self.image_key = image_key
        self.frame_size = frame_size
        self.frames_per_strip = frames_per_strip
        self._strip_cache: dict[int, torch.Tensor] = {}

        demo_path = self.path / f"{self.task_name}.pt"
        if not demo_path.exists():
            raise FileNotFoundError(f"Missing Dreamer metadata file: {demo_path}")

        demo = torch.load(demo_path, map_location="cpu", weights_only=False)
        if not isinstance(demo, dict):
            raise TypeError(f"Expected {demo_path} to contain a dict, got {type(demo)}")

        if "episode" not in demo:
            raise KeyError(f"{demo_path} must contain an 'episode' column")

        self._cache: dict[str, np.ndarray] = {
            key: _to_numpy(value) for key, value in demo.items()
        }
        lengths, offsets = _episode_lengths_from_ids(self._cache["episode"])

        if keys_to_load is None:
            keys = [image_key] + sorted(self._cache.keys())
        else:
            keys = list(keys_to_load)
        self._keys = keys

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    def _detect_task_name(self) -> str:
        pt_files = sorted(self.path.glob("*.pt"))
        if len(pt_files) != 1:
            raise ValueError(
                "Could not auto-detect task name. Expected exactly one .pt file in "
                f"{self.path}, found {[p.name for p in pt_files]}. Pass task_name=..."
            )
        return pt_files[0].stem

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _strip_path(self, strip_idx: int) -> Path:
        return self.path / f"{self.task_name}-{strip_idx}.png"

    def _load_strip(self, strip_idx: int) -> torch.Tensor:
        """Load one horizontal PNG strip as ``(N, C, H, W)`` uint8."""
        cached = self._strip_cache.get(strip_idx)
        if cached is not None:
            return cached

        path = self._strip_path(strip_idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing frame strip: {path}")

        arr = np.asarray(Image.open(path))
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        height, total_width, channels = arr.shape
        if self.frame_size is None:
            frame_h = height
            frame_w = height
        else:
            frame_h, frame_w = self.frame_size

        if height != frame_h or total_width % frame_w != 0:
            raise RuntimeError(
                f"Unexpected strip shape {arr.shape} in {path}; expected height "
                f"{frame_h} and width divisible by frame width {frame_w}."
            )

        n_frames = total_width // frame_w
        arr = arr.reshape(frame_h, n_frames, frame_w, channels)
        arr = np.transpose(arr, (1, 3, 0, 2))  # (N, C, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        self._strip_cache[strip_idx] = tensor

        if self.frames_per_strip is None:
            self.frames_per_strip = int(n_frames)
        return tensor

    def _load_frames(self, global_indices: Iterable[int]) -> torch.Tensor:
        frames = []
        for idx in global_indices:
            if self.frames_per_strip is None:
                # Load first strip once to infer frames_per_strip.
                self._load_strip(0)
            assert self.frames_per_strip is not None
            strip_idx = int(idx) // self.frames_per_strip
            local_idx = int(idx) % self.frames_per_strip
            strip = self._load_strip(strip_idx)
            if local_idx >= strip.shape[0]:
                raise IndexError(
                    f"Frame index {idx} maps to local index {local_idx}, but "
                    f"strip {strip_idx} only has {strip.shape[0]} frames"
                )
            frames.append(strip[local_idx])
        return torch.stack(frames, dim=0)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        g_end = int(self.offsets[ep_idx] + end)
        indices = range(g_start, g_end, self.frameskip)

        steps: dict[str, Any] = {}
        for key in self._keys:
            if key == self.image_key:
                steps[key] = self._load_frames(indices)
                continue

            if key not in self._cache:
                raise KeyError(f"Column '{key}' is not available in {self.task_name}.pt")

            data = self._cache[key][g_start:g_end]
            # Match the old Stable-WorldModel folder behavior: frame-like and
            # reward-like columns are frameskipped, action can stay dense because
            # some code expects all low-level actions between two sampled frames.
            if key != "action":
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[key] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[key] = torch.from_numpy(np.ascontiguousarray(data))

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col == self.image_key:
            raise KeyError("Image data is stored as PNG strips and cannot be returned as one array")
        return self._cache[col]

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {key: arr[row_idx] for key, arr in self._cache.items()}


class ImageDataset(DreamerFolderDataset):
    """Convenience reader that uses ``pixels`` as the image key."""

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        super().__init__(name=name, image_key=DEFAULT_IMAGE_KEY, **kwargs)


class DreamerFolderWriter:
    """Write episodes in Nicklas-style Dreamer4 raw format.

    The writer collects all non-image columns into ``<task>.pt`` and writes the
    selected image column as horizontal PNG strips named ``<task>-0.png``,
    ``<task>-1.png``, ... . Multiple episodes are separated by the generated
    ``episode`` tensor inside ``<task>.pt``.

    Args:
        path: Target dataset root.
        mode: ``'append'``, ``'overwrite'``, or ``'error'``.
        shard_size: Number of frames per PNG strip.
        task_name: Prefix for the output files, e.g. ``walker-walk``.
        image_key: Preferred image column. If absent, the first image-like
            column is used.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "append",
        shard_size: int = 2048,
        task_name: str = "task",
        image_key: str = DEFAULT_IMAGE_KEY,
    ) -> None:
        validate_write_mode(mode)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.task_name = task_name
        self.image_key = image_key
        self.shard_size = int(shard_size)

        self._frames: list[np.ndarray] = []
        self._columns: dict[str, list[np.ndarray]] = {}
        self._lengths: list[int] = []
        self._episode_ids: list[np.ndarray] = []
        self._episode_idx = 0
        self._shard_idx = 0
        self._validated = False
        self._appending_existing = False
        self._existing_shapes: dict[str, tuple[int, ...]] = {}

    def __enter__(self) -> "DreamerFolderWriter":
        exists = (self.path / f"{self.task_name}.pt").exists() or any(
            self.path.glob(f"{self.task_name}-*.png")
        )
        if exists and self.mode == "error":
            raise FileExistsError(
                f"DreamerFolderWriter: '{self.path}' already contains data for "
                f"task '{self.task_name}'. Use mode='overwrite' or mode='append'."
            )
        if exists and self.mode == "overwrite":
            self._clear_existing()
        elif exists and self.mode == "append":
            self._load_existing_state()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Flush pending frames and write ``<task>.pt`` metadata."""
        while self._frames:
            self._save_strip(final=True)

        if not self._columns and not self._episode_ids:
            return

        demo: dict[str, Any] = {
            key: torch.as_tensor(_stack_tabular(parts))
            for key, parts in sorted(self._columns.items())
        }
        demo["episode"] = torch.as_tensor(
            _stack_tabular(self._episode_ids), dtype=torch.int64
        )
        demo = make_tensordict(demo)

        if "action" not in demo:
            logging.warning("Writing Dreamer folder without an 'action' column")
        if "reward" not in demo:
            logging.warning("Writing Dreamer folder without a 'reward' column")

        torch.save(demo, self.path / f"{self.task_name}.pt")

    def write_episode(self, ep_data: dict[str, Any]) -> None:
        """Append one episode.

        ``ep_data`` should contain one image sequence and tabular columns such as
        ``action`` and ``reward``. The image sequence length defines the episode
        length; all tabular columns must have the same first dimension.
        """
        image_col = self._select_image_column(ep_data)
        frames = _to_numpy(ep_data[image_col])
        ep_len = int(len(frames))

        if ep_len <= 0:
            raise ValueError("Cannot write an empty episode")

        if self._appending_existing and not self._validated:
            self._validate_episode_against_existing(ep_data, image_col)
            self._validated = True

        for idx in range(ep_len):
            self._frames.append(_as_hwc_uint8(frames[idx]))
            if len(self._frames) >= self.shard_size:
                self._save_strip(final=False)

        for key, values in ep_data.items():
            if key == image_col or is_image_column(values):
                continue
            arr = _to_numpy(values)
            if len(arr) != ep_len:
                raise ValueError(
                    f"Column '{key}' has length {len(arr)}, but image column "
                    f"'{image_col}' has length {ep_len}"
                )
            if key == "episode":
                # Episode ids are generated by this writer so that they remain
                # consecutive even when appending.
                continue
            self._columns.setdefault(key, []).append(np.ascontiguousarray(arr))

        self._episode_ids.append(
            np.full((ep_len,), self._episode_idx, dtype=np.int64)
        )
        self._lengths.append(ep_len)
        self._episode_idx += 1

    def write_episodes(self, episodes: Iterable[dict[str, Any]]) -> None:
        for episode in episodes:
            self.write_episode(episode)

    def _select_image_column(self, ep_data: dict[str, Any]) -> str:
        if self.image_key in ep_data and is_image_column(ep_data[self.image_key]):
            return self.image_key

        image_cols = [key for key, value in ep_data.items() if is_image_column(value)]
        if not image_cols:
            raise ValueError(
                "No image column found. Expected a uint8 image sequence such as "
                f"'{self.image_key}' with shape (T, H, W, C) or (T, C, H, W)."
            )
        if len(image_cols) > 1:
            logging.warning(
                "Multiple image columns found %s; using '%s'. Pass image_key=... "
                "to choose explicitly.",
                image_cols,
                image_cols[0],
            )
        return image_cols[0]

    def _save_strip(self, *, final: bool) -> None:
        if not self._frames:
            return

        count = len(self._frames) if final else self.shard_size
        frames = self._frames[:count]
        self._frames = self._frames[count:]

        # HWC frames -> one horizontal H x (W * N) x C PNG strip.
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames):
            raise ValueError("All frames in a Dreamer PNG strip must have the same shape")

        strip = np.concatenate(frames, axis=1)
        out_path = self.path / f"{self.task_name}-{self._shard_idx}.png"
        Image.fromarray(strip.squeeze(-1) if strip.shape[-1] == 1 else strip).save(out_path)
        self._shard_idx += 1

    def _clear_existing(self) -> None:
        (self.path / f"{self.task_name}.pt").unlink(missing_ok=True)
        for png in self.path.glob(f"{self.task_name}-*.png"):
            png.unlink()
            
        for child in self.path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    def _load_existing_state(self) -> None:
        demo_path = self.path / f"{self.task_name}.pt"
        if not demo_path.exists():
            raise FileNotFoundError(
                f"Cannot append: missing metadata file {demo_path}. Use mode='overwrite'."
            )

        demo = torch.load(demo_path, map_location="cpu", weights_only=False)
        if "episode" not in demo:
            raise KeyError(f"Existing file {demo_path} does not contain 'episode'")

        for key, value in demo.items():
            arr = _to_numpy(value)
            if key == "episode":
                self._episode_ids = [arr.astype(np.int64)]
                self._episode_idx = int(arr.max()) + 1 if arr.size else 0
            else:
                self._columns[key] = [arr]
                self._existing_shapes[key] = tuple(arr.shape[1:])

        existing_pngs = sorted(self.path.glob(f"{self.task_name}-*.png"))
        self._shard_idx = len(existing_pngs)
        self._appending_existing = True

    def _validate_episode_against_existing(
        self, ep_data: dict[str, Any], image_col: str
    ) -> None:
        incoming_shapes = {
            key: tuple(_to_numpy(value).shape[1:])
            for key, value in ep_data.items()
            if key != image_col and not is_image_column(value) and key != "episode"
        }

        missing = set(self._existing_shapes) - set(incoming_shapes)
        extra = set(incoming_shapes) - set(self._existing_shapes)
        if missing or extra:
            raise ValueError(
                "Append failed because the tabular schema changed. "
                f"Missing columns: {sorted(missing)}; extra columns: {sorted(extra)}"
            )

        for key, shape in incoming_shapes.items():
            if shape != self._existing_shapes[key]:
                raise ValueError(
                    f"Append failed because column '{key}' changed shape: "
                    f"existing={self._existing_shapes[key]}, incoming={shape}"
                )


@register_format
class DreamerFolder(Format):
    """Stable-WorldModel format plugin for Nicklas-style Dreamer4 folders."""

    name = "dreamer_folder"

    @classmethod
    def detect(cls, path: str | Path) -> bool:
        p = Path(path)
        return p.is_dir() and any(p.glob("*.pt")) and any(p.glob("*-*.png"))

    @classmethod
    def open_reader(cls, path: str | Path, **kwargs: Any) -> DreamerFolderDataset:
        return DreamerFolderDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path: str | Path, **kwargs: Any) -> DreamerFolderWriter:
        return DreamerFolderWriter(path, **kwargs)


__all__ = [
    "DreamerFolder",
    "DreamerFolderDataset",
    "DreamerFolderWriter",
    "ImageDataset",
]