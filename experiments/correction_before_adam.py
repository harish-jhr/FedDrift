# One-round test of a SCAFFOLD-style correction added to the gradient before the Adam step, 2k FID.
import argparse
import json
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as p2v
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, denoising_loss
from common import cos, flat, setflat
from model import UNet
from sample import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/one_round/correction_before_adam")
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_est", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--fid_n", type=int, default=2000)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda")
    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed))]
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.float)
    pi = (sizes / sizes.sum()).tolist()
    bands = [(200 * b, 200 * b + 200) for b in range(5)]
    K = len(parts)

    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    w0 = flat(model)
    work = deepcopy(model)
    params = list(work.parameters())
    sizes_p = [p.numel() for p in params]

    def grad_at(w, x0, t, eps):
        setflat(work, w)
        work.zero_grad(set_to_none=True)
        n = len(x0)
        for i in range(0, n, args.chunk):
            sl = slice(i, i + args.chunk)
            (denoising_loss(work, x0[sl], t[sl], eps[sl], sched) * len(x0[sl]) / n).backward()
        return p2v([p.grad for p in params]).detach().clone()

    def anchors(seed):
        g = torch.Generator(device=device).manual_seed(seed)
        picks = [idx[torch.randperm(len(idx), device=device, generator=g)[:args.n_est]] for idx in parts]
        a = [[None] * 5 for _ in range(K)]
        for b, (lo, hi) in enumerate(bands):
            t = torch.randint(lo, hi, (args.n_est,), device=device, generator=g)
            eps = torch.randn((args.n_est, 3, 32, 32), device=device, generator=g)
            for i in range(K):
                a[i][b] = grad_at(w0, x_all[picks[i]], t, eps)
        abar = [sum(pi[i] * a[i][b] for i in range(K)) for b in range(5)]
        return [[abar[b] - a[i][b] for b in range(5)] for i in range(K)]

    cA = anchors(101)
    cB = anchors(202)
    rel = [[cos(cA[i][b], cB[i][b]) for b in range(5)] for i in range(K)]
    rel_band = [sum(rel[i][b] for i in range(K)) / K for b in range(5)]
    size_band = [sum(pi[i] * cA[i][b].pow(2).sum() for i in range(K)).item() for b in range(5)]
    print(f"anchor reliability, mean cos(estimate A, estimate B) by band: {[round(v, 3) for v in rel_band]}", flush=True)
    print(f"correction size by band, sum_i pi_i |c_ib|^2: {[f'{v:.2e}' for v in size_band]}", flush=True)
    del cB

    gen = torch.Generator(device=device).manual_seed(11)
    sch = []
    for idx in parts:
        bl = []
        for _ in range(args.local_epochs):
            perm = idx[torch.randperm(len(idx), device=device, generator=gen)]
            bl += [perm[i:i + args.batch_size] for i in range(0, len(perm), args.batch_size)]
        sch.append(bl)
    steps = len(sch[0])

    def own(i, k):
        idx = sch[i][k]
        g = torch.Generator(device=device).manual_seed(11 * 1000003 + i * 10007 + k)
        t = torch.randint(0, sched.T, (len(idx),), device=device, generator=g)
        eps = torch.randn((len(idx), 3, 32, 32), device=device, generator=g)
        flip = torch.rand(len(idx), device=device, generator=g) < 0.5
        return torch.where(flip.view(-1, 1, 1, 1), x_all[idx].flip(3), x_all[idx]), t, eps

    def train(batches, n_steps, corr_fn):
        setflat(work, w0)
        work.train()
        opt = torch.optim.Adam(params, lr=args.lr)
        scaler = torch.amp.GradScaler("cuda")
        for k in range(n_steps):
            x0, t, eps = batches(k)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = denoising_loss(work, x0, t, eps, sched)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if corr_fn is not None:
                c = corr_fn(t)
                if c is not None:
                    j = 0
                    for p, n in zip(params, sizes_p):
                        p.grad.add_(c[j:j + n].view_as(p))
                        j += n
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        return flat(work) - w0

    def make_corr(i, lam):
        def f(t):
            frac = [((t >= lo) & (t < hi)).float().mean().item() for lo, hi in bands]
            terms = [frac[b] * lam[b] for b in range(5)]
            if not any(terms):
                return None
            return sum(terms[b] * cA[i][b] for b in range(5) if terms[b])
        return f

    def pooled_batches(k):
        xs, ts, es = zip(*[own(i, k) for i in range(K)])
        return torch.cat(xs), torch.cat(ts), torch.cat(es)

    dp = train(pooled_batches, steps, None)
    print(f"pooled: |update| {dp.norm():.4f} (diagnostic only)", flush=True)

    variants = {
        "fedavg": [0.0] * 5,
        "uniform_lam0.5": [0.5] * 5,
        "uniform_lam1": [1.0] * 5,
        "gated_t<600_lam1": [1.0, 1.0, 1.0, 0.0, 0.0],
        "gated_t<400_lam1": [1.0, 1.0, 0.0, 0.0, 0.0],
    }

    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for i in range(0, len(x_all), 1000):
        fid.update((x_all[i:i + 1000] + 1) / 2, real=True)

    res = {"anchor_reliability_by_band": rel_band, "correction_size_by_band": size_band, "models": {}}
    for name, lam in variants.items():
        deltas = [train(lambda k, i=i: own(i, k), len(sch[i]), make_corr(i, lam) if any(lam) else None) for i in range(K)]
        u = sum(p * d for p, d in zip(pi, deltas))
        setflat(work, w0 + u)
        fid.reset()
        for j in range(0, args.fid_n, 1000):
            fid.update(generate(work.eval(), min(1000, args.fid_n - j), sched, device, seed=j), real=False)
        r = {"lambda": lam, "fid": fid.compute().item(), "norm_update": u.norm().item(),
             "mean_client_move": sum(d.norm().item() for d in deltas) / K, "cos_to_pooled": cos(u, dp)}
        res["models"][name] = r
        print(f"{name:18s} | FID{args.fid_n} {r['fid']:.2f} | |avg update| {r['norm_update']:.3f} | "
              f"avg client move {r['mean_client_move']:.3f} | cos to pooled {r['cos_to_pooled']:+.3f}", flush=True)
        json.dump(res, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
