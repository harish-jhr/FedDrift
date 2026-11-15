# Load a checkpoint and draw samples. Also builds the band-swap model used in the
# diagnostics: one model runs the reverse process except inside one timestep band,
# where a second model takes over.
import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import save_image

from data import CHANNELS, DATASETS, IMG_SIZE
from diffusion import Schedule, ddpm_sample
from model import UNet


class Swap(nn.Module):
    def __init__(self, base, other, lo, hi):
        super().__init__()
        self.base, self.other, self.lo, self.hi = base, other, lo, hi

    def forward(self, x, t):
        return self.other(x, t) if self.lo <= t[0].item() < self.hi else self.base(x, t)


def load(path, size, device, dataset="cifar10"):
    model = UNet(size, IMG_SIZE[dataset], CHANNELS[dataset]).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def build(args, device):
    dataset = getattr(args, "dataset", "cifar10")
    model = load(args.ckpt, args.model, device, dataset)
    name = os.path.splitext(args.ckpt)[0]
    if getattr(args, "swap_ckpt", None):
        lo, hi = args.swap_range
        model = Swap(model, load(args.swap_ckpt, args.model, device, dataset), lo, hi)
        name += f"_swap{lo}-{hi}"
    return model, name


def shape_of(model):
    return next(m.shape for m in model.modules() if isinstance(m, UNet))


# n samples in [0, 1].
def generate(model, n, sched, device, seed=0, batch=1000):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            x = ddpm_sample(model, (b, *shape_of(model)), sched, device, gen)
        out.append((x.clamp(-1, 1) + 1) / 2)
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--objective", default="ddpm", choices=["ddpm", "fm"])
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--swap_ckpt", default=None, help="model used inside --swap_range")
    parser.add_argument("--swap_range", type=int, nargs=2, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device, kind=args.objective)
    model, name = build(args, device)
    x = generate(model, args.n, sched, device, args.seed)
    out = args.out or name + "_samples.png"
    save_image(x, out, nrow=int(args.n ** 0.5), padding=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
