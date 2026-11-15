# Memorisation check for the shared-image runs: loss on the shared images vs unseen test images.
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10
from diffusion import Schedule, q_sample
from model import UNet


@torch.no_grad()
def band_loss(model, x, sched, device, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in range(5):
        t = torch.randint(200 * b, 200 * b + 200, (len(x),), generator=g)
        eps = torch.randn(x.shape, generator=g)
        tot = 0.0
        for i in range(0, len(x), 250):
            xb, tb, eb = x[i:i + 250].to(device), t[i:i + 250].to(device), eps[i:i + 250].to(device)
            tot += (model(q_sample(xb, tb, eb, sched), tb).float() - eb).pow(2).mean(dim=(1, 2, 3)).sum().item()
        out.append(tot / len(x))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--share_size", type=int, default=1000)
    parser.add_argument("--n_other", type=int, default=1000)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test, _ = load_cifar10(args.data_root, train=False)
    perm = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(args.seed))
    shared, rest = perm[:args.share_size], perm[args.share_size:]
    other = rest[torch.randperm(len(rest), generator=torch.Generator().manual_seed(12345))[:args.n_other]]
    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    sched = Schedule(device=device)
    ls = band_loss(model, x_test[shared], sched, device, 777)
    lo = band_loss(model, x_test[other], sched, device, 777)
    gap = " ".join(f"{100 * (a - b) / b:+.2f}%" for a, b in zip(ls, lo))
    print(f"{args.ckpt} | shared " + " ".join(f"{v:.5f}" for v in ls) + " | other " + " ".join(f"{v:.5f}" for v in lo) + f" | shared vs other {gap}", flush=True)


if __name__ == "__main__":
    main()
