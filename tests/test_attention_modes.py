import unittest

import torch

from dreamer4.model import Dynamics, Modality, SpaceSelfAttentionModality


class WorldModelAttentionModeTest(unittest.TestCase):
    def test_dynamics_isolates_agent_tokens(self):
        dynamics = Dynamics(
            d_model=8,
            d_bottleneck=2,
            d_spatial=4,
            n_spatial=2,
            n_register=1,
            n_agent=1,
            n_heads=2,
            depth=1,
            k_max=8,
            time_every=1,
        )
        space_attention = dynamics.transformer.layers[0].space
        modality_ids = space_attention.modality_ids
        allow = space_attention.attn_mask[0, 0]
        is_agent = modality_ids == int(Modality.AGENT)
        is_not_agent = ~is_agent

        self.assertTrue(allow[is_not_agent][:, is_not_agent].all().item())
        self.assertFalse(allow[is_not_agent][:, is_agent].any().item())
        self.assertFalse(allow[is_agent][:, is_not_agent].any().item())
        self.assertTrue(allow[is_agent][:, is_agent].all().item())

    def test_full_mixing_world_model_mode_is_not_supported(self):
        modality_ids = torch.tensor(
            [int(Modality.SPATIAL), int(Modality.AGENT)], dtype=torch.int32
        )

        with self.assertRaisesRegex(ValueError, "Unsupported mode"):
            SpaceSelfAttentionModality(
                d_model=8,
                n_heads=2,
                modality_ids=modality_ids,
                n_latents=0,
                mode="wm_agent",
                dropout=0.0,
            )


if __name__ == "__main__":
    unittest.main()
