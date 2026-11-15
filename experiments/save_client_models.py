# Runs one round of local training from a global checkpoint and saves every client's model
# as <out>_client<i>.pt. client_output_spread.py reads these.
import argparse
import os
import sys
from copy import deepcopy

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, parse_alpha, partition_path
from diffusion import Schedule
from fed import local_train
from model import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=parse_alpha, default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="small")
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed))]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    glob = UNet(args.model).to(device)
    glob.load_state_dict(torch.load(args.ckpt, map_location=device))
    for i, idx in enumerate(parts):
        local = deepcopy(glob)
        loss, steps, _ = local_train(local, x_all, idx, sched, args.local_epochs, args.batch_size, args.lr)
        torch.save(local.state_dict(), f"{args.out}_client{i}.pt")
        print(f"client {i} | {len(idx)} images | loss {loss:.4f} | {steps} steps -> {args.out}_client{i}.pt")


if __name__ == "__main__":
    main()
