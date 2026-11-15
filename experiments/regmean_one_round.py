# One round of RegMean from a FedAvg checkpoint, with the controls in the report (shuffled Gram matrices, FedAvg scaled to RegMean's size, timestep-layer part only) and the class-owner check, 2k FID.
import argparse
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regmean import grams, layers, regmean_layer as regmean
from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, denoising_loss, q_sample
from common import cos, flat, run_steps, setflat
from model import UNet
from sample import generate


class Bands(nn.Module):
    def __init__(self, models, T):
        super().__init__()
        self.models, self.T = nn.ModuleList(models), T

    def forward(self, x, t):
        return self.models[int(t[0]) * len(self.models) // self.T](x, t)


def gram_spread(G, b):
    vals = []
    for name, gk in G.items():
        g = gk[:, b]
        bar = g.mean(0)
        vals.append(((g - bar).pow(2).sum() / (len(g) * bar.pow(2).sum() + 1e-30)).item())
    return float(np.median(vals)), float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/one_round/regmean_one_round")
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gram_n", type=int, default=1024)
    parser.add_argument("--gram_batch", type=int, default=256)
    parser.add_argument("--shrink", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--rel", type=float, default=1e-4)
    parser.add_argument("--low", type=int, default=400)
    parser.add_argument("--expert_shrink", type=float, default=0.5)
    parser.add_argument("--n_eval", type=int, default=2048)
    parser.add_argument("--fid_n", type=int, default=2000)
    parser.add_argument("--recheck", type=float, default=1.0)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    x_all, y_all = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    x_test, y_test = load_cifar10(args.data_root, train=False)
    sched = Schedule(device=device)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed))]
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.float)
    pi = (sizes / sizes.sum()).tolist()
    owner = [int(np.bincount(y_all[p.cpu().numpy()]).argmax()) for p in parts]
    bands = [(200 * b, 200 * b + 200) for b in range(5)]

    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    w0 = flat(model)
    work = deepcopy(model)
    offs, s = {}, 0
    for name, p in model.named_parameters():
        offs[name] = (s, p.numel(), p.shape)
        s += p.numel()
    lnames = [name for name, _ in layers(model)]

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

    pick = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(404))[:args.n_eval]
    xe = x_test[pick].to(device)
    ye = torch.from_numpy(y_test[pick.numpy()]).to(device)
    eg = torch.Generator(device=device).manual_seed(404)
    t_list = [torch.randint(lo, hi, (args.n_eval,), device=device, generator=eg) for lo, hi in bands]
    eps_e = torch.randn(xe.shape, device=device, generator=eg)

    @torch.no_grad()
    def outputs(w):
        setflat(work, w)
        work.eval()
        res = []
        for t in t_list:
            o = []
            for i in range(0, len(xe), 512):
                o.append(work(q_sample(xe[i:i + 512], t[i:i + 512], eps_e[i:i + 512], sched), t[i:i + 512]).float())
            res.append(torch.cat(o))
        return res

    def per_sample(o):
        return [(ob - eps_e).pow(2).mean(dim=(1, 2, 3)) for ob in o]

    t0 = time.time()
    deltas, G = [], []
    ggen = torch.Generator(device=device).manual_seed(505)
    blend = None
    match = [torch.zeros(args.n_eval, device=device) for _ in bands]
    for i in range(len(parts)):
        d, _ = run_steps(work, w0, len(sch[i]), client_loss(i), args.lr, "adam", use_amp, 1.0)
        deltas.append(d)
        G.append(grams(work, x_all[parts[i]], sched, bands, args.gram_n, args.gram_batch, ggen))
        o = outputs(w0 + d)
        blend = [ob * pi[i] for ob in o] if blend is None else [bb + ob * pi[i] for bb, ob in zip(blend, o)]
        own = ye == owner[i]
        for b, l in enumerate(per_sample(o)):
            match[b][own] = l[own]
        print(f"client {i} (class {owner[i]}): |delta| {d.norm():.4f} | {time.time() - t0:.0f}s", flush=True)
    dp, _ = run_steps(work, w0, steps, pooled_loss, args.lr, "adam", use_amp, 1.0)
    fa = sum(p * d for p, d in zip(pi, deltas))
    print(f"pooled: |delta| {dp.norm():.4f} | {time.time() - t0:.0f}s", flush=True)

    res = {"owner": owner, "fork": {}, "gram_spread": [], "models": {}}
    fork = {"start": per_sample(outputs(w0)), "fedavg": per_sample(outputs(w0 + fa)), "pooled": per_sample(outputs(w0 + dp)),
            "blend": per_sample(blend), "matching": match}
    print("FORK: loss on held-out test images, by band (matching = the client that owns the image's class)", flush=True)
    for k, v in fork.items():
        res["fork"][k] = [l.mean().item() for l in v]
        print(f"  {k:9s} | " + " ".join(f"{x:.5f}" for x in res["fork"][k]), flush=True)
    fl, ml = res["fork"]["fedavg"], res["fork"]["matching"]
    gain = [(f - m) / f * 100 for f, m in zip(fl, ml)]
    win = [(fork["matching"][b] < fork["fedavg"][b]).float().mean().item() * 100 for b in range(5)]
    res["fork"]["matching_gain_pct"] = gain
    res["fork"]["matching_wins_pct"] = win
    print("  matching vs fedavg, loss lower by % | " + " ".join(f"{g:+.2f}" for g in gain), flush=True)
    print("  images where matching beats fedavg, % | " + " ".join(f"{w:.1f}" for w in win), flush=True)
    print(f"  FORK {'OPEN' if gain[0] > 0 and gain[1] > 0 else 'CLOSED'} at t<400", flush=True)

    for b in range(5):
        res["gram_spread"].append(gram_spread({n: torch.stack([g[n] for g in G]) for n in lnames}, b))
    print("Gram spread between clients by band (median / mean over layers, 0 = identical) | " +
          " ".join(f"{m:.3f}/{a:.3f}" for m, a in res["gram_spread"]), flush=True)

    def ds_of(name):
        st, n, shp = offs[name + ".weight"]
        return [d[st:st + n].view(shp) for d in deltas]

    def merged(omega, a):
        u = fa.clone()
        for name in lnames:
            gs = [sum(w * g[name][b] for b, w in enumerate(omega) if w) for g in G]
            st, n, _ = offs[name + ".weight"]
            u[st:st + n] = regmean(ds_of(name), gs, pi, a, args.rel).flatten()
        return u

    omegas = {"unif": [1.0] * 5, "low": [1.0 if hi <= args.low else 0.0 for _, hi in bands]}
    updates = {"fedavg": fa, "pooled": dp}
    for a in args.shrink:
        for k, om in omegas.items():
            updates[f"regmean_{k}_a{a:g}"] = merged(om, a)
    experts = [w0 + merged([1.0 if j == b else 0.0 for j in range(5)], args.expert_shrink) for b in range(5)]
    print(f"merges built | {time.time() - t0:.0f}s", flush=True)

    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for i in range(0, len(x_all), 1000):
        fid.update((x_all[i:i + 1000] + 1) / 2, real=True)
    ex_models = []
    for w in experts:
        m = deepcopy(model)
        setflat(m, w)
        ex_models.append(m.eval())
    ex_net = Bands(ex_models, sched.T)

    def score(net, off):
        fid.reset()
        for j in range(0, args.fid_n, 1000):
            fid.update(generate(net, min(1000, args.fid_n - j), sched, device, seed=off + j), real=False)
        return fid.compute().item()

    def report(k, r):
        line = f"{k:22s} | loss by band {' '.join(f'{v:.5f}' for v in r['band_loss'])} | cos pooled {r['cos_to_pooled']:+.3f} | FID{args.fid_n} {r['fid']:.2f}"
        if "fid_b" in r:
            line += f" / {r['fid_b']:.2f}"
        print(line + f" | {time.time() - t0:.0f}s", flush=True)

    for k, u in updates.items():
        r = {"band_loss": [l.mean().item() for l in per_sample(outputs(w0 + u))], "cos_to_pooled": cos(u, dp),
             "dist_to_pooled": (u - dp).norm().item(), "norm_update": u.norm().item()}
        setflat(work, w0 + u)
        r["fid"] = score(work.eval(), 0)
        res["models"][k] = r
        report(k, r)
        json.dump(res, open(args.out + ".json", "w"), indent=1)
    r = {"band_loss": [], "cos_to_pooled": float("nan")}
    for b, m in enumerate(ex_models):
        r["band_loss"].append(per_sample(outputs(flat(m)))[b].mean().item())
    r["fid"] = score(ex_net, 0)
    res["models"]["regmean_band_experts"] = r
    report("regmean_band_experts", r)
    json.dump(res, open(args.out + ".json", "w"), indent=1)

    base = res["models"]["fedavg"]["fid"]
    again = ["fedavg"] + [k for k, r in res["models"].items() if k.startswith("regmean") and r["fid"] < base - args.recheck]
    print(f"second sampling seed for: {again}", flush=True)
    for k in again:
        if k == "regmean_band_experts":
            net = ex_net
        else:
            setflat(work, w0 + updates[k])
            net = work.eval()
        res["models"][k]["fid_b"] = score(net, 100000)
        report(k, res["models"][k])
        json.dump(res, open(args.out + ".json", "w"), indent=1)

    fb = res["models"]["fedavg"]
    passed = [k for k in again if k != "fedavg" and res["models"][k]["fid"] <= fb["fid"] - 2 and res["models"][k]["fid_b"] <= fb["fid_b"] - 2]
    print(f"PASS (>= 2 FID better than fedavg on both seeds): {passed if passed else 'none'}", flush=True)
    res["passed"] = passed
    json.dump(res, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
