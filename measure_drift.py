import argparse
import json
import os
import time
from copy import deepcopy

import torch

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule
from drift import measure_drift
from fed import fedavg, local_train
from model import UNet


def parse_alpha(s):
    return None if s == "iid" else float(s)


def tag(alpha):
    return "iid" if alpha is None else f"{alpha:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=parse_alpha, default=0.1)  # partition FedAvg trains on
    parser.add_argument("--measure_alphas", type=parse_alpha, nargs="+", default=[0.1, 0.5, 5.0, None])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="small", choices=["small", "standard"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--measure_rounds", type=int, nargs="+", default=[0, 1, 5, 10, 25, 50])
    parser.add_argument("--measure_batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    parser.add_argument("--out_dir", default="./runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)

    def get_parts(alpha):
        path = partition_path(args.partition_dir, args.n_clients, alpha, args.seed)
        return [torch.from_numpy(p).to(device) for p in load_partition(path)]

    train_parts = get_parts(args.alpha)
    # every partition is measured at the SAME global weights, so the IID control
    # differs from the non-IID curves only in how the data is split
    measure_parts = {tag(a): get_parts(a) for a in args.measure_alphas}

    model = UNet(args.model).to(device)
    print(f"{args.model} unet | {model.num_params() / 1e6:.1f}M params | {device}")

    out_path = os.path.join(args.out_dir, f"drift_train{tag(args.alpha)}_seed{args.seed}.json")
    log = {"args": vars(args), "train_loss": {}, "drift": []}

    def measure(rnd):
        t0 = time.time()
        for name, parts in measure_parts.items():
            # every partition starts its measurement from the same seed
            res = measure_drift(model, x_all, parts, sched, batch=args.measure_batch,
                                repeats=args.repeats, seed=args.seed + rnd)
            for r in res:
                log["drift"].append({"round": rnd, "alpha": name, **r})
            row = "  ".join(f"{r['zeta2_cross_normalized']:8.4f}" for r in res)
            print(f"  round {rnd} | alpha {name:>4} | zeta2/|g|^2 per bucket: {row}")
        with open(out_path, "w") as f:
            json.dump(log, f, indent=1)
        print(f"  measured in {time.time() - t0:.0f}s -> {out_path}")

    if 0 in args.measure_rounds:
        measure(0)

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        states, sizes, losses, n_steps = [], [], [], 0
        for idx in train_parts:
            # each client starts the round from a copy of the global model
            local = deepcopy(model)
            loss, steps = local_train(local, x_all, idx, sched, args.local_epochs, args.batch_size, args.lr)
            states.append(local.state_dict())
            sizes.append(len(idx))
            losses.append(loss)
            n_steps += steps
        model.load_state_dict(fedavg(states, sizes))

        loss = sum(l * n for l, n in zip(losses, sizes)) / sum(sizes)
        dt = time.time() - t0
        log["train_loss"][rnd] = loss
        print(f"round {rnd}/{args.rounds} | loss {loss:.4f} | {n_steps} steps in {dt:.0f}s ({n_steps / dt:.1f} it/s)")

        # measured after averaging, i.e. at the weights the next round starts from
        if rnd in args.measure_rounds:
            measure(rnd)

    torch.save(model.state_dict(), out_path.replace(".json", ".pt"))


if __name__ == "__main__":
    main()
