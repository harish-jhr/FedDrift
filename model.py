import torch.nn as nn
from diffusers import UNet2DModel

WIDTHS = {
    "small": (64, 128, 128, 128),
    "standard": (128, 256, 256, 256),
}


# DDPM U-Net. The 32x32 model is the base; 64x64 adds one level in front, and 28x28
# (MNIST) drops the lowest level because 28 cannot be halved three times.
class UNet(nn.Module):

    def __init__(self, size="small", img_size=32, channels=3):
        super().__init__()
        self.shape = (channels, img_size, img_size)
        widths = WIDTHS[size]
        down = ("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
        if img_size == 64:
            widths, down = (widths[0],) + widths, ("DownBlock2D",) + down
        elif img_size == 28:
            widths, down = widths[:3], down[:3]
        elif img_size != 32:
            raise ValueError(f"unsupported image size {img_size}")
        up = tuple(b.replace("Down", "Up") for b in reversed(down))
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=channels,
            out_channels=channels,
            layers_per_block=2,
            block_out_channels=widths,
            down_block_types=down,
            up_block_types=up,
        )

    def forward(self, x, t):
        return self.unet(x, t).sample

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
