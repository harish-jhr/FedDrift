# One-round tests of per-band strengths for RegMean (gates and band weights), 2k FID.
import argparse
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, denoising_loss, q_sample
from common import cos, flat, run_steps, setflat
from regmean import grams, layers, regmean_layer as regmean
from model import UNet
from sample import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--out", default="runs/one_round/band_strength_tests")
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gram_n", type=int, default=1024)
    parser.add_argument("--gram_batch", type=int, default=256)
    parser.add_argument("--rel", type=float, default=1e-4)
    parser.add_argument("--n_eval", type=int, default=2048)
    parser.add_argument("--fid_n", type=int, default=2000)
    parser.add_argument("--prev", default=None)
    parser.add_argument("--merges", nargs="+", default=["notime_unif_a0.9", "notime_low_a0.5", "fedavg_same_env", "notime_unif_a0.5", "notime_low_a0.9", "timeonly_unif_a0.9"])
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
    bands = [(200 * b, 200 * b + 200) for b in range(5)]
    ref = json.load(open(args.ref))["models"] if args.ref else None

    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    w0 = flat(model)
    work = deepcopy(model)
    offs, s = {}, 0
    for name, p in model.named_parameters():
        offs[name] = (s, p.numel(), p.shape)
        s += p.numel()
    lnames = [name for name, _ in layers(model)]
    tnames = [n for n in lnames if "time" in n]
    onames = [n for n in lnames if "time" not in n]

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
    owner = [int(np.bincount(y_all[p.cpu().numpy()]).argmax()) for p in parts]
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

    t0 = time.time()
    deltas, G = [], []
    ggen = torch.Generator(device=device).manual_seed(505)
    for i in range(len(parts)):
        d, _ = run_steps(work, w0, len(sch[i]), client_loss(i), args.lr, "adam", use_amp, 1.0)
        deltas.append(d)
        G.append(grams(work, x_all[parts[i]], sched, bands, args.gram_n, args.gram_batch, ggen))
        print(f"client {i}: |delta| {d.norm():.4f} | {time.time() - t0:.0f}s", flush=True)
    dp, _ = run_steps(work, w0, steps, pooled_loss, args.lr, "adam", use_amp, 1.0)
    fa = sum(p * d for p, d in zip(pi, deltas))
    torch.cuda.empty_cache()
    print(f"pooled: |delta| {dp.norm():.4f} | {time.time() - t0:.0f}s", flush=True)

    def merged(omega, a, which, rel, perm=None):
        u = fa.clone()
        src = G if perm is None else [G[j] for j in perm]
        for name in which:
            st, n, shp = offs[name + ".weight"]
            ds = [d[st:st + n].view(shp) for d in deltas]
            gs = [sum(w * g[name][b] for b, w in enumerate(omega) if w) for g in src]
            u[st:st + n] = regmean(ds, gs, pi, a, rel).flatten()
        return u

    def band_weighted(rho, a):
        u = fa.clone()
        for name in onames:
            st, n, shp = offs[name + ".weight"]
            ds = [d[st:st + n].view(shp).movedim(1, -1) for d in deltas]
            gs = [[a * g[name][b] + (1 - a) * torch.diag(torch.diagonal(g[name][b])) for b in range(5)] for g in G]
            bar = [sum(p * gk[b] for p, gk in zip(pi, gs)) for b in range(5)]
            S = sum(bar)
            Sr = S + args.rel * S.diagonal().mean() * torch.eye(len(S), device=S.device)
            dbar = sum(p * d for p, d in zip(pi, ds))
            N = dbar @ S
            for b in range(5):
                N = N + rho[b] * (sum(p * (d @ gk[b]) for p, d, gk in zip(pi, ds, gs)) - dbar @ bar[b])
            X = torch.linalg.solve(Sr, N.reshape(-1, Sr.shape[0]).T).T
            u[st:st + n] = X.reshape(N.shape).movedim(-1, 1).flatten()
        return u

    def part_norm(u, names):
        return sum(u[offs[n + ".weight"][0]:offs[n + ".weight"][0] + offs[n + ".weight"][1]].pow(2).sum().item() for n in names) ** 0.5

    U, LOW = [1.0] * 5, [1.0, 1.0, 0.0, 0.0, 0.0]
    cache = {}

    def u9():
        if "u9" not in cache:
            cache["u9"] = merged(U, 0.9, onames, args.rel)
        return cache["u9"]
    res = {"breakdown": {}, "models": {}}
    print("NORMS: update norm in timestep-embedding layers vs all other conv/linear layers", flush=True)
    full = {"fedavg": fa, "pooled": dp, "regmean_unif_a0.5": merged(U, 0.5, lnames, args.rel),
            "regmean_low_a0.5": merged(LOW, 0.5, lnames, args.rel), "regmean_unif_a0.9": merged(U, 0.9, lnames, args.rel)}
    for k, u in full.items():
        tn, on = part_norm(u, tnames), part_norm(u, onames)
        res["breakdown"][k] = {"total": u.norm().item(), "time_layers": tn, "other_layers": on, "time_share_sq": tn ** 2 / (u.norm().item() ** 2)}
        print(f"  {k:20s} | total {u.norm():8.3f} | time layers {tn:8.3f} | other layers {on:7.3f} | share of squared norm in time layers {res['breakdown'][k]['time_share_sq']:.3f}", flush=True)
    json.dump(res, open(args.out + ".json", "w"), indent=1)

    for label, names in [("other layers", onames), ("time layers", tnames)]:
        sh = np.array([[sum(g[n][b].diagonal().sum().item() for g in G) for b in range(5)] for n in names])
        sh = sh / sh.sum(1, keepdims=True)
        res.setdefault("trace_share_median", {})[label] = np.median(sh, axis=0).tolist()
        print(f"STAGE 0b: share of summed Gram trace per band, median over {label} | " + " ".join(f"{v:.3f}" for v in np.median(sh, axis=0)), flush=True)
    json.dump(res, open(args.out + ".json", "w"), indent=1)

    torch.cuda.empty_cache()
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for i in range(0, len(x_all), 250):
        fid.update((x_all[i:i + 250] + 1) / 2, real=True)
    torch.cuda.empty_cache()

    def score(w, off):
        setflat(work, w)
        fid.reset()
        for j in range(0, args.fid_n, 1000):
            x = generate(work.eval(), min(1000, args.fid_n - j), sched, device, seed=off + j)
            for c in range(0, len(x), 250):
                fid.update(x[c:c + 250], real=False)
        torch.cuda.empty_cache()
        return fid.compute().item()

    if ref:
        print(f"CONTROLS: FID on two seeds; reference fedavg {ref['fedavg']['fid']:.2f} / {ref['fedavg']['fid_b']:.2f}, regmean_unif_a0.9 {ref['regmean_unif_a0.9']['fid']:.2f} / {ref['regmean_unif_a0.9']['fid_b']:.2f}, "
              f"regmean_low_a0.5 {ref['regmean_low_a0.5']['fid']:.2f} / {ref['regmean_low_a0.5']['fid_b']:.2f}", flush=True)

    def gated(rho, a):
        u = fa.clone()
        for name in onames:
            st, n, shp = offs[name + ".weight"]
            ds = [d[st:st + n].view(shp) for d in deltas]
            bar = [sum(p * g[name][b] for p, g in zip(pi, G)) for b in range(5)]
            gs = [sum(rho[b] * g[name][b] + (1 - rho[b]) * bar[b] for b in range(5)) for g in G]
            u[st:st + n] = regmean(ds, gs, pi, a, args.rel).flatten()
        return u
    ML, FK = [1.0, 1.0, 0.25, 0.25, 0.25], [0.25, 1.0, 1.0, 0.25, 0.25]
    specs = {"notime_unif_a0.9": lambda: merged(U, 0.9, onames, args.rel),
             "notime_low_a0.5": lambda: merged(LOW, 0.5, onames, args.rel),
             "fedavg_same_env": lambda: fa.clone(),
             "notime_unif_a0.5": lambda: merged(U, 0.5, onames, args.rel),
             "notime_low_a0.9": lambda: merged(LOW, 0.9, onames, args.rel),
             "timeonly_unif_a0.9": lambda: merged(U, 0.9, tnames, args.rel),
             "notime_mixedlow_a0.9": lambda: merged(ML, 0.9, onames, args.rel),
             "notime_fork_a0.9": lambda: merged(FK, 0.9, onames, args.rel),
             "notime_mixedlow_a0.5": lambda: merged(ML, 0.5, onames, args.rel),
             "notime_fork_a0.5": lambda: merged(FK, 0.5, onames, args.rel),
             "notime_shuffled_unif_a0.9": lambda: merged(U, 0.9, onames, args.rel, perm=[(j + 1) % len(parts) for j in range(len(parts))]),
             "notime_unif_a0.7": lambda: merged(U, 0.7, onames, args.rel),
             "notime_mixedlow_a0.7": lambda: merged(ML, 0.7, onames, args.rel),
             "notime_fork_a0.7": lambda: merged(FK, 0.7, onames, args.rel),
             "notime_scaled_fedavg": lambda: fa * (u9().norm() / fa.norm()),
             "notime_interp0.5": lambda: fa + 0.5 * (u9() - fa),
             "notime_interp1.5": lambda: fa + 1.5 * (u9() - fa),
             "gate_11100_a0.9": lambda: gated([1, 1, 1, 0, 0], 0.9),
             "gate_01100_a0.9": lambda: gated([0, 1, 1, 0, 0], 0.9),
             "gate_11000_a0.9": lambda: gated([1, 1, 0, 0, 0], 0.9)}
    shapes = {}

    def shape(kind):
        if not shapes:
            sp = []
            for b in range(5):
                vals = []
                for name in onames:
                    g = torch.stack([gk[name][b] for gk in G])
                    bar = g.mean(0)
                    vals.append(((g - bar).pow(2).sum() / (len(g) * bar.pow(2).sum() + 1e-30)).item())
                sp.append(float(np.median(vals)))
            shapes["spread_raw"] = sp
            shapes["unif"] = [1.0] * 5
            shapes["spread"] = [v / (sum(sp) / 5) for v in sp]
            shapes["spread2"] = [v * v / (sum(x * x for x in sp) / 5) for v in sp]
            shapes["hand"] = [1.5, 1.5, 1.5, 0.25, 0.25]

            @torch.no_grad()
            def per_sample(w):
                setflat(work, w)
                work.eval()
                out = []
                for t in t_list:
                    o = torch.cat([work(q_sample(xe[i:i + 512], t[i:i + 512], eps_e[i:i + 512], sched), t[i:i + 512]).float() for i in range(0, len(xe), 512)])
                    out.append((o - eps_e).pow(2).mean(dim=(1, 2, 3)))
                return out

            l0 = per_sample(w0)
            gen = torch.zeros(5)
            for k, d in enumerate(deltas):
                own = ye == owner[k]
                lk = per_sample(w0 + d)
                for b in range(5):
                    gen[b] += ((l0[b][own].mean() - lk[b][own].mean()) / l0[b][own].mean()).item() / len(deltas)
            raw = [max(0.2, float(torch.exp(gen[b] * 100 / 5))) for b in range(5)]
            m = sum(raw) / 5
            shapes["gen"] = [max(0.2, v / m) for v in raw]
            shapes["gen_signal"] = gen.tolist()
            print("GENERALISATION per band (client loss drop on held-out images of its own class, vs start, %): " + " ".join(f"{100 * v:+.2f}" for v in gen.tolist()), flush=True)
            res["band_shapes"] = dict(shapes)
            print("BAND SHAPES (mean 1): Gram spread per band " + " ".join(f"{v:.3f}" for v in sp) + " | " +
                  " | ".join(f"{k} " + " ".join(f"{v:.2f}" for v in shapes[k]) for k in ["spread", "spread2", "hand", "gen"]), flush=True)
        return shapes[kind]

    for kind in ["unif", "spread", "spread2", "hand", "gen"]:
        for rb in [1.0, 1.5]:
            specs[f"bw_{kind}_r{rb:g}"] = (lambda kind=kind, rb=rb: band_weighted([rb * v for v in shape(kind)], 0.9))
    prev = json.load(open(args.prev))["models"] if args.prev else {}
    done_verdict = [False]

    def verdict():
        allres = {**prev, **res["models"]}
        need = [f"notime_{w}_a{a:g}" for w in ["unif", "mixedlow", "fork"] for a in [0.5, 0.7, 0.9]] + ["fedavg_same_env"]
        if done_verdict[0] or not all(k in allres for k in need):
            return
        done_verdict[0] = True
        fl = allres["fedavg_same_env"]["band_loss"]
        out = {}
        print("STAGE 1 VERDICT (win = beats uniform at the same a by >= 1.0 FID on both seeds, at >= 2 of the 3 a values, with t>600 loss within 5% of FedAvg)", flush=True)
        for w in ["mixedlow", "fork"]:
            rows = []
            for a in [0.5, 0.7, 0.9]:
                m, uref = allres[f"notime_{w}_a{a:g}"], allres[f"notime_unif_a{a:g}"]
                guard = all(m["band_loss"][b] <= 1.05 * fl[b] for b in (3, 4))
                win = m["fid"] <= uref["fid"] - 1.0 and m["fid_b"] <= uref["fid_b"] - 1.0 and guard
                rows.append((a, m["fid"], m["fid_b"], uref["fid"], uref["fid_b"], guard, win))
                print(f"  {w:8s} a={a:g} | {m['fid']:.2f} / {m['fid_b']:.2f} vs uniform {uref['fid']:.2f} / {uref['fid_b']:.2f} | t>600 guard {'ok' if guard else 'FAIL'} | {'WIN' if win else 'no'}", flush=True)
            n = sum(r[-1] for r in rows)
            out[w] = {"rows": rows, "wins": n, "verdict": "WIN" if n >= 2 else "TIE"}
            print(f"  {w}: {n} of 3 -> {out[w]['verdict']}", flush=True)
        res["stage1_verdict"] = out
        json.dump(res, open(args.out + ".json", "w"), indent=1)
    done_gate = [False]

    def gate_verdict():
        m = res["models"]
        gates = [k for k in args.merges if k.startswith("gate_")]
        if done_gate[0] or not gates or not all(k in m for k in gates + ["notime_unif_a0.9", "fedavg_same_env"]):
            return
        done_gate[0] = True
        fl, ur = m["fedavg_same_env"]["band_loss"], m["notime_unif_a0.9"]
        print(f"GATE VERDICT (win = beats uniform a=0.9 by >= 1.0 FID on both seeds, t>600 loss within 5% of FedAvg) | fedavg {m['fedavg_same_env']['fid']:.2f} / {m['fedavg_same_env']['fid_b']:.2f} | "
              f"uniform a=0.9 {ur['fid']:.2f} / {ur['fid_b']:.2f}, |u| {ur['norm']:.3f}", flush=True)
        out = {}
        for k in gates:
            g = m[k]
            guard = all(g["band_loss"][b] <= 1.05 * fl[b] for b in (3, 4))
            win = g["fid"] <= ur["fid"] - 1.0 and g["fid_b"] <= ur["fid_b"] - 1.0 and guard
            out[k] = {"win": win, "guard": guard}
            print(f"  {k:16s} | {g['fid']:.2f} / {g['fid_b']:.2f} | |u| {g['norm']:.3f} (uniform {ur['norm']:.3f}) | t>600 guard {'ok' if guard else 'FAIL'} | {'WIN' if win else 'no'}", flush=True)
        res["gate_verdict"] = out
        json.dump(res, open(args.out + ".json", "w"), indent=1)

    done_bw = [False]

    def bw_verdict():
        m = {**prev, **res["models"]}
        need = [k for k in args.merges if k.startswith("bw_")]
        ref1 = "bw_unif_r1" if "bw_unif_r1" in m else "notime_unif_a0.9"
        if done_bw[0] or not need or not all(k in m for k in need + [ref1, "bw_unif_r1.5"]):
            return
        done_bw[0] = True
        fl = band_loss(w0 + fa)
        print("BAND-WEIGHT VERDICT (win = beats uniform at the same mean strength by >= 1.0 FID on both seeds, t>600 loss within 5% of FedAvg)", flush=True)
        out = {}
        for k in need:
            if k in ("bw_unif_r1", "bw_unif_r1.5"):
                continue
            ref = m[ref1] if k.endswith("_r1") else m["bw_unif_r1.5"]
            g = m[k]
            guard = all(g["band_loss"][b] <= 1.05 * fl[b] for b in (3, 4))
            win = g["fid"] <= ref["fid"] - 1.0 and g["fid_b"] <= ref["fid_b"] - 1.0 and guard
            out[k] = {"win": win, "guard": guard}
            print(f"  {k:16s} | {g['fid']:.2f} / {g['fid_b']:.2f} vs uniform {ref['fid']:.2f} / {ref['fid_b']:.2f} | |u| {g['norm']:.3f} vs {ref['norm']:.3f} | t>600 guard {'ok' if guard else 'FAIL'} | {'WIN' if win else 'no'}", flush=True)
        res["band_weight_verdict"] = out
        json.dump(res, open(args.out + ".json", "w"), indent=1)

    if any(k.startswith("bw_") for k in args.merges):
        chk = band_weighted([1.0] * 5, 0.9) - u9()
        print(f"SANITY: band decomposition at rho = 1 vs uniform RegMean a=0.9 | max |diff| {chk.abs().max().item():.2e} | relative {chk.norm().item() / (u9() - fa).norm().item():.2e}", flush=True)
        shape("unif")

    runs = [(k, specs[k]) for k in args.merges]
    for k, f in runs:
        u = f()
        ratios = {n: part_norm(u, [n]) / (part_norm(fa, [n]) + 1e-12) for n in lnames}
        worst = max(ratios, key=ratios.get)
        r = {"band_loss": band_loss(w0 + u), "norm": u.norm().item(), "time_layers": part_norm(u, tnames), "other_layers": part_norm(u, onames),
             "max_layer_ratio": ratios[worst], "max_layer": worst, "layers_over_10x": [n for n in lnames if ratios[n] > 10],
             "cos_pooled": cos(u, dp), "fid": score(w0 + u, 0)}
        r["fid_b"] = score(w0 + u, 100000)
        res["models"][k] = r
        print(f"  {k:20s} | FID {r['fid']:.2f} / {r['fid_b']:.2f} | |u| {r['norm']:.3f} (time {r['time_layers']:.3f}, other {r['other_layers']:.3f}) | "
              f"max layer ratio {ratios[worst]:.1f} ({worst}), {len(r['layers_over_10x'])} layers over 10x | cos pooled {r['cos_pooled']:+.3f} | "
              f"loss {' '.join(f'{v:.5f}' for v in r['band_loss'])} | {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(args.out + ".json", "w"), indent=1)
        verdict()
        gate_verdict()
        bw_verdict()
    print(f"done | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
