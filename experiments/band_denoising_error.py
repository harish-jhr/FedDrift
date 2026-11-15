# Denoising error of one or more checkpoints in each timestep band, on the same images and noise.
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10
from diffusion import Schedule, denoising_loss
from drift import BUCKETS
from sample import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    x_all, _ = load_cifar10(args.data_root)
    pick = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(args.seed))[:args.n]
    x = x_all[pick].to(device)

    draws = []
    for lo, hi in BUCKETS:
        t = torch.randint(lo, hi, (args.n,), device=device, generator=gen)
        eps = torch.randn(x.shape, device=device, generator=gen)
        draws.append((t, eps))

    print(f"{'checkpoint':<60}" + "".join(f"{lo}-{hi}".rjust(12) for lo, hi in BUCKETS))
    for path in args.ckpts:
        model = load(path, args.model, device)
        row = []
        for t, eps in draws:
            total = 0.0
            with torch.no_grad():
                for i in range(0, args.n, args.batch):
                    s = slice(i, i + args.batch)
                    total += denoising_loss(model, x[s], t[s], eps[s], sched).item() * len(x[s])
            row.append(total / args.n)
        print(f"{path[-60:]:<60}" + "".join(f"{v:12.5f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
