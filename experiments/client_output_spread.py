# How much client models disagree in their outputs in each timestep band.
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10
from diffusion import Schedule, q_sample
from fed import fedavg
from model import UNet


def load(path, size, device):
    model = UNet(size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", required=True)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--model", default="small")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--bands", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device)
    paths = [f"{args.clients}_client{i}.pt" for i in range(args.n_clients)]
    clients = [load(p, args.model, device) for p in paths]
    avg = UNet(args.model).to(device)
    avg.load_state_dict(fedavg([c.state_dict() for c in clients], [1] * args.n_clients))
    avg.eval()

    x, y = load_cifar10(args.data_root, train=False)
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(len(x), generator=g)[:args.n]
    x, y = x[pick].to(device), torch.from_numpy(y[pick.numpy()]).to(device)

    width = sched.T // args.bands
    keys = ["weight_vs_function", "blend_vs_matching", "merged_vs_matching", "client_spread", "matching_norm"]
    out = []
    for b in range(args.bands):
        acc = {k: 0.0 for k in keys}
        for i in range(0, args.n, args.batch):
            x0, yb = x[i:i + args.batch], y[i:i + args.batch]
            t = b * width + torch.randint(0, width, (len(x0),), device=device, generator=torch.Generator(device=device).manual_seed(args.seed * 1000 + b * 97 + i))
            eps = torch.randn(x0.shape, device=device, generator=torch.Generator(device=device).manual_seed(args.seed * 7 + b * 13 + i))
            xt = q_sample(x0, t, eps, sched)
            outs = torch.stack([c(xt, t).float() for c in clients])
            bar = outs.mean(0)
            match = outs[yb, torch.arange(len(x0), device=device)]
            merged = avg(xt, t).float()
            per = lambda a, c: (a - c).pow(2).mean(dim=(1, 2, 3)).sum().item()
            acc["weight_vs_function"] += per(merged, bar)
            acc["blend_vs_matching"] += per(bar, match)
            acc["merged_vs_matching"] += per(merged, match)
            acc["client_spread"] += (outs - bar).pow(2).mean(dim=(2, 3, 4)).mean(0).sum().item()
            acc["matching_norm"] += match.pow(2).mean(dim=(1, 2, 3)).sum().item()
        row = {k: v / args.n for k, v in acc.items()}
        row["band"] = [b * width, (b + 1) * width]
        out.append(row)
        print(f"t {b * width:4d}-{(b + 1) * width:4d} | " + " | ".join(f"{k} {row[k]:.3e}" for k in keys), flush=True)

    with open(f"{args.clients}_merge_gap.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
