from diffusers import UNet2DModel
import torch.nn as nn

# Lightweight model configurations.
# "tiny" is the recommended configuration for the resource-constrained
# "small" and "standard" are retained for comparison.
WIDTHS = {
    "tiny": (32, 64, 64, 64),
    "small": (64, 128, 128, 128),
    "standard": (128, 256, 256, 256),
}


class UNet(nn.Module):
    def __init__(self, variant="small"):
        super().__init__()

        if variant not in WIDTHS:
            raise ValueError(
                f"Unknown model variant '{variant}'. "
                f"Choose from: {', '.join(WIDTHS)}"
            )

        w1, w2, w3, w4 = WIDTHS[variant]

        self.net = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(w1, w2, w3, w4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t):
        return self.net(x, t).sample
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
