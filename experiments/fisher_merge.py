# One-round test of Fisher-weighted merging (Adam second moments; uniform, per band and drift-weighted), 2k FID.
import argparse
import json
import os
import sys
from copy import deepcopy

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, denoising_loss
from common import cos, flat, run_steps, setflat
from model import UNet
from sample import generate


def merge(deltas, weights, pi, rel=1e-3):
    den = sum(p * w for p, w in zip(pi, weights))
    eps = rel * den.mean()
    num = sum(p * (w + eps) * d for p, w, d in zip(pi, weights, deltas))
    return num / (den + eps)


def band_fisher(model, w, x, sched, bands, micro, n_micro, gen):
    setflat(model, w)
    out = []
    for lo, hi in bands:
        acc = torch.zeros_like(w)
        for _ in range(n_micro):
            idx = torch.randint(0, len(x), (micro,), device=x.device, generator=gen)
            t = torch.randint(lo, hi, (micro,), device=x.device, generator=gen)
            eps = torch.randn((micro, 3, 32, 32), device=x.device, generator=gen)
            model.zero_grad(set_to_none=True)
            denoising_loss(model, x[idx], t, eps, sched).backward()
            acc += torch.cat([p.grad.flatten() for p in model.parameters()]).pow(2)
        out.append(acc / n_micro)
    return out


@torch.no_grad()
def band_loss(model, w, x, t_list, eps, sched, batch=512):
    setflat(model, w)
    model.eval()
    res = []
    for t in t_list:
        tot = 0.0
        for i in range(0, len(x), batch):
            tot += denoising_loss(model, x[i:i + batch], t[i:i + batch], eps[i:i + batch], sched).item() * len(x[i:i + batch])
        res.append(tot / len(x))
    return res


def ownership(fs):
    tot = sum(fs) + 1e-30
    share = torch.stack([f / tot for f in fs])
    mass = tot / tot.sum()
    return (share.max(dim=0).values * mass).sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/one_round/fisher_merge")
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--n_micro", type=int, default=32)
    parser.add_argument("--n_eval", type=int, default=2048)
    parser.add_argument("--fid_n", type=int, default=2000)
    parser.add_argument("--damage", type=float, nargs=5, default=[0.4, 0.3, 0.25, 0.025, 0.025])
    parser.add_argument("--scales", type=float, nargs="*", default=[])
    parser.add_argument("--skip_fisher", action="store_true")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    x_test, _ = load_cifar10(args.data_root, train=False)
    sched = Schedule(device=device)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed))]
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.float)
    pi = (sizes / sizes.sum()).tolist()
    bands = [(200 * b, 200 * b + 200) for b in range(5)]

    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    w0 = flat(model)
    work = deepcopy(model)

    gen = torch.Generator(device=device).manual_seed(11)
    sch = []
    for idx in parts:
        b = []
        for _ in range(args.local_epochs):
            perm = idx[torch.randperm(len(idx), device=device, generator=gen)]
            b += [perm[i:i + args.batch_size] for i in range(0, len(perm), args.batch_size)]
        sch.append(b)
    steps = len(sch[0])

    def draw(i, k, n):
        g = torch.Generator(device=device).manual_seed(11 * 1000003 + i * 10007 + k)
        return (torch.randint(0, sched.T, (n,), device=device, generator=g),
                torch.randn((n, 3, 32, 32), device=device, generator=g),
                torch.rand(n, device=device, generator=g) < 0.5)

    def batch(i, k):
        idx = sch[i][k]
        t, eps, flip = draw(i, k, len(idx))
        return torch.where(flip.view(-1, 1, 1, 1), x_all[idx].flip(3), x_all[idx]), t, eps

    def client_loss(i):
        def f(m, k):
            x0, t, eps = batch(i, k)
            return denoising_loss(m, x0, t, eps, sched)
        return f

    def pooled_loss(m, k):
        xs, ts, es = zip(*[batch(i, k) for i in range(len(parts))])
        return denoising_loss(m, torch.cat(xs), torch.cat(ts), torch.cat(es), sched)

    deltas, vs, fishers = [], [], []
    fgen = torch.Generator(device=device).manual_seed(303)
    for i in range(len(parts)):
        d, v = run_steps(work, w0, len(sch[i]), client_loss(i), args.lr, "adam", use_amp, 1.0)
        deltas.append(d)
        vs.append(v)
        if not args.skip_fisher:
            fishers.append(band_fisher(work, w0 + d, x_all[parts[i]], sched, bands, args.micro, args.n_micro, fgen))
        print(f"client {i}: |delta| {d.norm():.4f}" + ("" if args.skip_fisher else f" | band fisher mass {[round(f.sum().item(), 4) for f in fishers[-1]]}"), flush=True)
    dp, _ = run_steps(work, w0, steps, pooled_loss, args.lr, "adam", use_amp, 1.0)
    print(f"pooled: |delta| {dp.norm():.4f}", flush=True)

    fa = sum(p * d for p, d in zip(pi, deltas))
    exp_step = max(1.0, (sum(p * d.pow(2).sum() for p, d in zip(pi, deltas)) / (2 * fa.pow(2).sum())).item())
    print(f"FedExP server step {exp_step:.3f}", flush=True)
    if args.skip_fisher:
        fishers = [[torch.ones_like(w0)] * 5 for _ in parts]
    scale = [sum(fishers[i][b].mean() for i in range(len(parts))).item() / len(parts) for b in range(5)]
    try:
        log = json.load(open(args.ckpt.replace("_round100.pt", ".json")))
        zeta = [max(r["zeta2_cross"], 0.0) for r in log["drift"] if r["round"] == 100 and r["alpha"] == "class"]
    except Exception:
        zeta = [1.0] * 5
    lam = {
        "fisher_uniform": [1.0] * 5,
        "fisher_bandnorm": [1.0 / s for s in scale],
        "fisher_drift": [z / sum(zeta) / s for z, s in zip(zeta, scale)],
        "fisher_damage": [d / s for d, s in zip(args.damage, scale)],
    }
    updates = {"start": torch.zeros_like(w0), "pooled": dp, "fedavg": fa}
    for s in args.scales:
        updates[f"fedavg_x{s:g}"] = s * fa
    if args.scales:
        updates["fedexp"] = exp_step * fa
    if not args.skip_fisher:
        updates["fisher_adam"] = merge(deltas, vs, pi)
        for k, l in lam.items():
            updates[k] = merge(deltas, [sum(lb * f[b] for b, lb in enumerate(l)) for f in fishers], pi)

    eg = torch.Generator(device=device).manual_seed(404)
    xe = x_test[torch.randperm(len(x_test), generator=torch.Generator().manual_seed(404))[:args.n_eval]].to(device)
    t_list = [torch.randint(lo, hi, (args.n_eval,), device=device, generator=eg) for lo, hi in bands]
    eps_e = torch.randn(xe.shape, device=device, generator=eg)

    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    if args.fid_n:
        for i in range(0, len(x_all), 1000):
            fid.update((x_all[i:i + 1000] + 1) / 2, real=True)

    res = {"ownership_by_band": [ownership([fishers[i][b] for i in range(len(parts))]) for b in range(5)],
           "fisher_scale_by_band": scale, "lambda": lam, "zeta2_round100": zeta, "models": {}}
    print(f"fisher ownership by band (0.1 = shared evenly, 1 = one client owns): {[round(o, 3) for o in res['ownership_by_band']]}", flush=True)
    for k, u in updates.items():
        losses = band_loss(work, w0 + u, xe, t_list, eps_e, sched)
        r = {"band_loss": losses, "mean_loss": sum(losses) / 5, "norm_update": u.norm().item(),
             "dist_to_pooled": (u - dp).norm().item(), "cos_to_pooled": cos(u, dp) if u.norm() > 0 else 0.0}
        if args.fid_n:
            setflat(work, w0 + u)
            fid.reset()
            for j in range(0, args.fid_n, 1000):
                fid.update(generate(work.eval(), min(1000, args.fid_n - j), sched, device, seed=j), real=False)
            r["fid"] = fid.compute().item()
        res["models"][k] = r
        print(f"{k:16s} | loss by band {' '.join(f'{v:.5f}' for v in losses)} | mean {r['mean_loss']:.5f} | |u| {r['norm_update']:.3f} | "
              f"dist pooled {r['dist_to_pooled']:.3f} | cos pooled {r['cos_to_pooled']:+.3f}" + (f" | FID{args.fid_n} {r['fid']:.2f}" if args.fid_n else ""), flush=True)
        json.dump(res, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
