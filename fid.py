# FID of a checkpoint against the training set (10,000 samples by default).
# The result is appended to --log as: name, number of samples, FID.
import argparse
import time

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data import DATASETS, load_dataset
from diffusion import Schedule
from sample import build, generate


def rgb(x):
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--objective", default="ddpm", choices=["ddpm", "fm"])
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap_ckpt", default=None, help="band swap: model used inside --swap_range")
    parser.add_argument("--swap_range", type=int, nargs=2, default=None)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--log", default="fid.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device=device, kind=args.objective)
    model, name = build(args, device)
    t0 = time.time()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real, _ = load_dataset(args.dataset, args.data_root)
    for i in range(0, len(real), args.batch):
        fid.update(rgb((real[i:i + args.batch] + 1) / 2).to(device), real=True)
    print(f"real features done | {time.time() - t0:.0f}s", flush=True)

    done = 0
    while done < args.n:
        b = min(args.batch, args.n - done)
        fid.update(rgb(generate(model, b, sched, device, args.seed + done, args.batch)), real=False)
        done += b
        print(f"generated {done}/{args.n} | {time.time() - t0:.0f}s", flush=True)

    score = fid.compute().item()
    print(f"{name} | n {args.n} | FID {score:.3f} | {time.time() - t0:.0f}s")
    with open(args.log, "a") as f:
        f.write(f"{name}\t{args.n}\t{score:.4f}\n")


if __name__ == "__main__":
    main()
