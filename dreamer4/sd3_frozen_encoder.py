"""
Frozen SD3 VAE tokenizer wrapper for Dreamer4 dynamics pretraining.

This uses the Stable Diffusion 3 VAE as a frozen image tokenizer.

Encoder path:
    frames: (B,T,3,H,W) float [0,1] or uint8 [0,255]
    z:      (B,T,n_latents,d_bottleneck)

Decoder path:
    z:      (B,T,n_latents,d_bottleneck)
    frames: (B,T,3,H,W) float [0,1]

The output z is intended to be passed into Dreamer4's packing function:

    z1 = pack_bottleneck_to_spatial(
        z,
        n_spatial=n_latents // packing_factor,
        k=packing_factor,
    )

For eval / visualization, packed dynamics predictions can be unpacked first:

    z_btLd = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)
    frames = sd3_encoder.decode(z_btLd)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from diffusers import AutoencoderKL
from einops import rearrange


class SD3FrozenEncoder(nn.Module):
    """
    Frozen Stable Diffusion 3 VAE encoder/decoder used as a tokenizer.

    Important:
      - This class has no temporal modeling; frames are encoded independently.
      - Input frames must be RGB and either float [0,1] or uint8 [0,255].
      - Encoder output is SD3-scaled latent tokens.
      - Decoder input must be in the same scaled latent space returned by forward().

    Args:
      model_name:
        HuggingFace diffusers model name containing an SD3 VAE subfolder.

      image_size:
        Training image size (H,W). Must match dataset frames.
        Both H and W must be divisible by 8.

      latent_patch_size:
        Groups p x p cells of the SD3 latent grid into one token.

        For H=W=128:
          p=1 -> latent grid 16x16 -> n_latents=256, d_bottleneck=16
          p=2 -> latent grid 8x8   -> n_latents=64,  d_bottleneck=64
          p=4 -> latent grid 4x4   -> n_latents=16,  d_bottleneck=256

        For H=W=224:
          SD3 latent grid is 28x28.
          Valid p values include 1, 2, 4, 7, 14, 28.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        *,
        image_size: Tuple[int, int] = (128, 128),
        latent_patch_size: int = 2,
        device: torch.device | str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        strict_input_range: bool = True,
    ):
        super().__init__()

        H, W = map(int, image_size)
        assert H % 8 == 0 and W % 8 == 0, (
            f"image_size must be divisible by 8, got {image_size}"
        )

        self.H = H
        self.W = W
        self.C = 3

        self.latent_h = H // 8
        self.latent_w = W // 8

        self.latent_patch_size = int(latent_patch_size)
        self.strict_input_range = bool(strict_input_range)

        assert self.latent_patch_size > 0
        assert self.latent_h % self.latent_patch_size == 0, (
            f"latent_h={self.latent_h} must be divisible by "
            f"latent_patch_size={self.latent_patch_size}"
        )
        assert self.latent_w % self.latent_patch_size == 0, (
            f"latent_w={self.latent_w} must be divisible by "
            f"latent_patch_size={self.latent_patch_size}"
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch_dtype,
        ).to(device)

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.latent_channels = int(self.vae.config.latent_channels)

        # SD3 VAE latent convention.
        self.scaling_factor = float(
            getattr(self.vae.config, "scaling_factor", 1.5305)
        )
        self.shift_factor = float(
            getattr(self.vae.config, "shift_factor", 0.0609)
        )

        p = self.latent_patch_size
        self.n_latents = (self.latent_h // p) * (self.latent_w // p)
        self.d_bottleneck = self.latent_channels * p * p

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def vae_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def feature_dim(self) -> int:
        return self.d_bottleneck

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert frames to SD3 VAE input range [-1,1].

        Accepts:
          uint8 [0,255]
          float [0,1]
        """
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        else:
            frames = frames.float()

        if self.strict_input_range:
            mn = float(frames.min().detach().cpu())
            mx = float(frames.max().detach().cpu())
            if mn < -1e-4 or mx > 1.0 + 1e-4:
                raise ValueError(
                    "SD3FrozenEncoder expects frames in [0,1] or uint8 [0,255], "
                    f"got min={mn:.4f}, max={mx:.4f}."
                )

        frames = frames.clamp(0, 1)
        frames = frames * 2.0 - 1.0
        return frames.to(device=self.device, dtype=self.vae_dtype)

    def _scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        SD3 latent scaling.

        Raw VAE latent -> scaled latent space used by SD3 diffusion.
        """
        return (latents - self.shift_factor) * self.scaling_factor

    def _unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Reverse SD3 latent scaling.

        Scaled latent space -> raw VAE latent before decoding.
        """
        return latents / self.scaling_factor + self.shift_factor

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode raw frames.

        Accepts:
          (B,T,3,H,W) -> (B,T,n_latents,d_bottleneck)
          (B,3,H,W)   -> (B,n_latents,d_bottleneck)

        Returns:
          SD3-scaled latent tokens.
        """
        is_video = frames.dim() == 5

        if is_video:
            B, T, C, H, W = frames.shape
            assert C == 3 and H == self.H and W == self.W, (
                f"Expected (B,T,3,{self.H},{self.W}), got {tuple(frames.shape)}"
            )
            x = rearrange(frames, "b t c h w -> (b t) c h w")
        else:
            B, C, H, W = frames.shape
            assert C == 3 and H == self.H and W == self.W, (
                f"Expected (B,3,{self.H},{self.W}), got {tuple(frames.shape)}"
            )
            x = frames

        x = self._prepare_frames(x)

        latents = self.vae.encode(x).latent_dist.mode()
        latents = self._scale_latents(latents)

        p = self.latent_patch_size

        tokens = rearrange(
            latents,
            "bt c (lh p1) (lw p2) -> bt (lh lw) (c p1 p2)",
            p1=p,
            p2=p,
        )

        tokens = tokens.float()

        if is_video:
            tokens = rearrange(tokens, "(b t) l d -> b t l d", b=B, t=T)

        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode SD3 latent tokens back to RGB frames.

        Accepts:
          (B,T,n_latents,d_bottleneck) -> (B,T,3,H,W)
          (B,n_latents,d_bottleneck)   -> (B,3,H,W)

        Input:
          tokens must be in the same scaled latent space returned by forward().

        Returns:
          frames float [0,1].
        """
        is_video = tokens.dim() == 4

        if is_video:
            B, T, L, D = tokens.shape
            assert L == self.n_latents and D == self.d_bottleneck, (
                f"Expected (B,T,{self.n_latents},{self.d_bottleneck}), "
                f"got {tuple(tokens.shape)}"
            )
            tokens_flat = rearrange(tokens, "b t l d -> (b t) l d")
        else:
            B, L, D = tokens.shape
            assert L == self.n_latents and D == self.d_bottleneck, (
                f"Expected (B,{self.n_latents},{self.d_bottleneck}), "
                f"got {tuple(tokens.shape)}"
            )
            tokens_flat = tokens

        p = self.latent_patch_size
        lh = self.latent_h // p
        lw = self.latent_w // p

        latents = rearrange(
            tokens_flat,
            "bt (lh lw) (c p1 p2) -> bt c (lh p1) (lw p2)",
            lh=lh,
            lw=lw,
            c=self.latent_channels,
            p1=p,
            p2=p,
        )

        latents = latents.to(device=self.device, dtype=self.vae_dtype)
        latents = self._unscale_latents(latents)

        frames = self.vae.decode(latents).sample
        frames = (frames * 0.5 + 0.5).clamp(0, 1)
        frames = frames.float()

        if is_video:
            frames = rearrange(frames, "(b t) c h w -> b t c h w", b=B, t=T)

        return frames

    @torch.no_grad()
    def reconstruct(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convenience method:

            frames -> encode -> decode -> reconstructed frames

        Useful for checking whether SD3 VAE preserves the task-relevant details.
        """
        return self.decode(self.forward(frames))

    def tokenizer_info(self) -> Dict[str, int | float | str | bool]:
        return {
            "tokenizer_type": "sd3_frozen_encoder_decoder",
            "H": self.H,
            "W": self.W,
            "C": 3,
            "n_latents": self.n_latents,
            "d_bottleneck": self.d_bottleneck,
            "latent_channels": self.latent_channels,
            "latent_patch_size": self.latent_patch_size,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "has_decoder": True,
        }


def dtype_from_string(name: str) -> torch.dtype:
    name = str(name).lower()

    if name in {"fp32", "float32"}:
        return torch.float32

    if name in {"fp16", "float16", "half"}:
        return torch.float16

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {name}")


def load_frozen_sd3_encoder(
    *,
    device: torch.device,
    model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    H: int = 128,
    W: int = 128,
    latent_patch_size: int = 2,
    torch_dtype: torch.dtype = torch.float16,
    strict_input_range: bool = True,
) -> tuple[
    SD3FrozenEncoder,
    SD3FrozenEncoder,
    Dict[str, int | float | str | bool],
]:
    """
    Loader matching the train_dynamics.py convention:

        encoder, decoder, tok_args = load_frozen_sd3_encoder(...)

    The same frozen SD3 module is returned as both encoder and decoder because:
      - call encoder(frames) for encoding
      - call decoder.decode(z) for decoding

    For Dreamer4-style code that expects decoder to be non-None, this is useful.
    """
    encoder = SD3FrozenEncoder(
        model_name=model_name,
        image_size=(H, W),
        latent_patch_size=latent_patch_size,
        device=device,
        torch_dtype=torch_dtype,
        strict_input_range=strict_input_range,
    ).eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    tok_args = encoder.tokenizer_info()

    # Keep this key matching Dreamer4 train_dynamics expectations.
    # It is not used by SD3 encoding, because SD3 consumes raw frames directly.
    tok_args.update({"patch": 4})

    return encoder, encoder, tok_args