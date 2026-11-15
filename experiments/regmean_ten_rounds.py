# Ten rounds of RegMean continued from a FedAvg checkpoint, to check the one-round gain holds over several rounds.
import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, denoising_loss, q_sample
from common import flat, run_steps, setflat
from regmean import grams, layers
from model import UNet
from sample import generate


def shrink(g, a):
    return a * g + (1 - a) * torch.diag(torch.diagonal(g))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="+", default=["fedavg", "unif:0.9", "low:0.5"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--fid_rounds", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.8)
    parser.add_argument("--gram_n", type=int, default=1024)
    parser.add_argument("--gram_batch", type=int, default=256)
    parser.add_argument("--low", type=int, default=400)
    parser.add_argument("--rel", type=float, default=1e-4)
    parser.add_argument("--n_eval", type=int, default=2048)
    parser.add_argument("--fid_n", type=int, default=2000)
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
    w_start = flat(model)
    model.load_state_dict(torch.load(args.ema, map_location=device))
    e_start = flat(model)
    work = deepcopy(model)
    offs, s = {}, 0
    for name, p in model.named_parameters():
        offs[name] = (s, p.numel(), p.shape)
        s += p.numel()
    lnames = [name for name, _ in layers(model) if "time" not in name]

    pick = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(404))[:args.n_eval]
    xe = x_test[pick].to(device)
    eg = torch.Generator(device=device).manual_seed(404)
    t_list = [torch.randint(lo, hi, (args.n_eval,), device=device, generator=eg) for lo, hi in bands]
    eps_e = torch.randn(xe.shape, device=device, generator=eg)

    @torch.no_grad()
    def band_loss(w):
        setflat(work, w)
        work.eval()
        res = []
        for t in t_list:
            tot = 0.0
            for i in range(0, len(xe), 512):
                o = work(q_sample(xe[i:i + 512], t[i:i + 512], eps_e[i:i + 512], sched), t[i:i + 512]).float()
                tot += (o - eps_e[i:i + 512]).pow(2).mean(dim=(1, 2, 3)).sum().item()
            res.append(tot / len(xe))
        return res

    def schedule(rnd, i):
        g = torch.Generator(device=device).manual_seed((args.seed * 1000 + rnd) * 1009 + i)
        idx = parts[i]
        b = []
        for _ in range(args.local_epochs):
            perm = idx[torch.randperm(len(idx), device=device, generator=g)]
            b += [perm[j:j + args.batch_size] for j in range(0, len(perm), args.batch_size)]
        return b

    def client_loss(rnd, i, sch):
        def f(m, k):
            idx = sch[k]
            g = torch.Generator(device=device).manual_seed(((args.seed * 1000 + rnd) * 1009 + i) * 100003 + k)
            t = torch.randint(0, sched.T, (len(idx),), device=device, generator=g)
            eps = torch.randn((len(idx), 3, 32, 32), device=device, generator=g)
            flip = torch.rand(len(idx), device=device, generator=g) < 0.5
            x0 = torch.where(flip.view(-1, 1, 1, 1), x_all[idx].flip(3), x_all[idx])
            return denoising_loss(m, x0, t, eps, sched)
        return f

    def merge(w, deltas, G, a):
        fa = sum(p * d for p, d in zip(pi, deltas))
        if G is None:
            return fa, None
        u = fa.clone()
        worst = float("inf")
        for name in lnames:
            st, n, shp = offs[name + ".weight"]
            gs = [shrink(g[name], a) for g in G]
            S = sum(p * g for p, g in zip(pi, gs))
            ridge = args.rel * S.diagonal().mean()
            worst = min(worst, (torch.linalg.eigvalsh(S)[0] / ridge).item())
            S = S + ridge * torch.eye(len(S), device=S.device)
            ds = [d[st:st + n].view(shp) for d in deltas]
            N = sum(p * (d.movedim(1, -1) @ g) for p, d, g in zip(pi, ds, gs))
            X = torch.linalg.solve(S, N.reshape(-1, S.shape[0]).T).T
            u[st:st + n] = X.reshape(N.shape).movedim(-1, 1).flatten()
        return u, worst

    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for i in range(0, len(x_all), 1000):
        fid.update((x_all[i:i + 1000] + 1) / 2, real=True)

    def score(w, off):
        setflat(work, w)
        fid.reset()
        for j in range(0, args.fid_n, 1000):
            fid.update(generate(work.eval(), min(1000, args.fid_n - j), sched, device, seed=off + j), real=False)
        return fid.compute().item()

    t0 = time.time()
    res = {"args": vars(args), "round0": {}, "arms": {}}
    res["round0"] = {"band_loss": band_loss(w_start), "fid": score(w_start, 0), "fid_ema": score(e_start, 0)}
    print(f"round 0 | FID {res['round0']['fid']:.2f} | EMA FID {res['round0']['fid_ema']:.2f} | loss {' '.join(f'{v:.5f}' for v in res['round0']['band_loss'])} | {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(args.out + ".json", "w"), indent=1)

    state = {arm: {"w": w_start.clone(), "e": e_start.clone()} for arm in args.arms}
    for arm in args.arms:
        res["arms"][arm] = {"rounds": {}}

    for rnd in range(1, args.rounds + 1):
        for arm in args.arms:
            kind, a = (arm.split(":")[0], float(arm.split(":")[1])) if ":" in arm else (arm, None)
            w = state[arm]["w"]
            deltas, G = [], ([] if kind != "fedavg" else None)
            for i in range(len(parts)):
                sch = schedule(rnd, i)
                d, _ = run_steps(work, w, len(sch), client_loss(rnd, i, sch), args.lr, "adam", use_amp, 1.0)
                deltas.append(d)
                if G is not None:
                    keep = [b for b in range(5) if kind == "unif" or bands[b][1] <= args.low]
                    gg = torch.Generator(device=device).manual_seed((args.seed * 1000 + rnd) * 7717 + i)
                    full = grams(work, x_all[parts[i]], sched, [bands[b] for b in keep], args.gram_n, args.gram_batch, gg)
                    G.append({k: v.sum(0) for k, v in full.items()})
            u, worst = merge(w, deltas, G, a)
            fav = sum(p * d for p, d in zip(pi, deltas))
            fa_norm = fav.norm().item()
            sl = lambda v, n: v[offs[n + ".weight"][0]:offs[n + ".weight"][0] + offs[n + ".weight"][1]]
            max_ratio = max(sl(u, n).norm().item() / (sl(fav, n).norm().item() + 1e-12) for n in lnames)
            del fav
            del deltas, G
            w = w + u
            e = state[arm]["e"].lerp_(w, 1 - args.ema_decay)
            state[arm]["w"] = w
            r = {"update_norm": u.norm().item(), "fedavg_update_norm": fa_norm, "weight_norm": w.norm().item(),
                 "layer_weight_norm": {n: w[offs[n + ".weight"][0]:offs[n + ".weight"][0] + offs[n + ".weight"][1]].norm().item() for n in lnames},
                 "min_eig_over_ridge": worst, "max_layer_ratio": max_ratio, "band_loss": band_loss(w)}
            res["arms"][arm]["rounds"][rnd] = r
            line = (f"round {rnd} | {arm:9s} | |u| {r['update_norm']:.3f} (plain avg {fa_norm:.3f}) | |w| {r['weight_norm']:.3f} | "
                    f"loss {' '.join(f'{v:.5f}' for v in r['band_loss'])}")
            if worst is not None:
                line += f" | min eig / ridge {worst:.2e} | max layer ratio {max_ratio:.1f}"
            print(line + f" | {time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(args.out + ".json", "w"), indent=1)
        if rnd in args.fid_rounds:
            last = rnd == max(args.fid_rounds)
            for arm in args.arms:
                r = res["arms"][arm]["rounds"][rnd]
                r["fid"] = score(state[arm]["w"], 0)
                line = f"FID round {rnd} | {arm:9s} | {r['fid']:.2f}"
                if last:
                    r["fid_b"] = score(state[arm]["w"], 100000)
                    r["fid_ema"] = score(state[arm]["e"], 0)
                    r["fid_ema_b"] = score(state[arm]["e"], 100000)
                    line += f" / {r['fid_b']:.2f} | EMA {r['fid_ema']:.2f} / {r['fid_ema_b']:.2f}"
                print(line + f" | {time.time() - t0:.0f}s", flush=True)
                json.dump(res, open(args.out + ".json", "w"), indent=1)
            if last:
                for arm in args.arms:
                    torch.save(state[arm]["w"], f"{args.out}_{arm.replace(':', '_a')}_w.pt")
    print(f"done | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
