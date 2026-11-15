# Tries to predict FedAvg's error relative to pooled training from curvature (Hessian-vector) terms, one round from a FedAvg checkpoint.
import argparse
import json
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as p2v

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10, load_partition, partition_path, random_flip
from diffusion import Schedule, denoising_loss
from model import UNet
from common import cos, fedavg_round, flat, hvp, run_steps, setflat


def base_predictors(grad_fn, w0, pi, batch_a, band_batches):
    g = [grad_fn(w0, batch_a[i]) for i in range(len(pi))]
    gbar = sum(p * gi for p, gi in zip(pi, g))
    d = [gi - gbar for gi in g]
    cb = []
    for bb in band_batches:
        cb.append(sum(p * hvp(grad_fn, w0, di, bb[i]) for i, (p, di) in enumerate(zip(pi, d))))
    return {"C": sum(cb) / len(cb), "Cb": cb, "gbar": gbar}


def update_predictors(grad_fn, w0, pi, batch_b, deltas, vs):
    dbar = sum(p * di for p, di in zip(pi, deltas))
    hu = [hvp(grad_fn, w0, di - dbar, batch_b[i]) for i, di in enumerate(deltas)]
    cu = sum(p * h for p, h in zip(pi, hu))
    cpu = None
    if vs[0] is not None:
        cpu = sum(p * h / (v.sqrt() + 1e-8) for p, h, v in zip(pi, hu, vs))
    return {"Cu": cu, "CPu": cpu, "dbar": dbar}


def predictors(grad_fn, w0, pi, batch_a, batch_b, deltas, vs, band_batches):
    return {**base_predictors(grad_fn, w0, pi, batch_a, band_batches), **update_predictors(grad_fn, w0, pi, batch_b, deltas, vs)}


def evaluate(eA, eB, PA, PB):
    out = {"norm_eA": eA.norm().item(), "norm_eB": eB.norm().item(), "cos_eA_eB": cos(eA, eB)}
    for k in ["C", "Cu", "CPu", "dbar", "gbar"]:
        if PA[k] is None:
            continue
        a, b = PA[k], PB[k]
        beta = (eA @ a / (a @ a)).item()
        out[k] = {"cos_A": cos(eA, a), "cos_B": cos(eB, b), "cos_estA_estB": cos(a, b), "beta_fit_A": beta,
                  "heldout_residual_B": ((eB - beta * b).norm() / eB.norm()).item()}
    return out


def band_report(e, cb):
    total = e @ sum(cb)
    return [{"band": f"{200 * b}-{200 * b + 200}", "norm_Cb": c.norm().item(), "cos_e_Cb": cos(e, c),
             "share_of_e_dot_C": (e @ c / total).item()} for b, c in enumerate(cb)]


def verdict(res):
    best = None
    for k in ["C", "Cu", "CPu"]:
        if k in res and (best is None or res[k]["heldout_residual_B"] < res[best]["heldout_residual_B"]):
            best = k
    r = res[best]["heldout_residual_B"]
    trivial = min(res[k]["heldout_residual_B"] for k in ["dbar", "gbar"])
    beats = r < trivial
    if r <= 0.9 and beats:
        v = "PASS"
    elif r <= 0.98 and beats:
        v = "WEAK"
    else:
        v = "FAIL"
    return f"{v} | best predictor {best}: held-out residual {r:.3f} | best trivial residual {trivial:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/one_round/curvature_prediction")
    parser.add_argument("--alpha", default="class")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.03, 0.3])
    parser.add_argument("--n_est", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed))]
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.float)
    pi = (sizes / sizes.sum()).tolist()

    model = UNet("small").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    w0 = flat(model)
    gm = deepcopy(model)

    def grad_fn(w, batch):
        x0, t, eps = batch
        setflat(gm, w)
        gm.zero_grad(set_to_none=True)
        n = len(x0)
        for i in range(0, n, args.chunk):
            loss = denoising_loss(gm, x0[i:i + args.chunk], t[i:i + args.chunk], eps[i:i + args.chunk], sched) * len(x0[i:i + args.chunk]) / n
            loss.backward()
        return p2v([p.grad for p in gm.parameters()]).detach().clone()

    def schedules(seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        out = []
        for idx in parts:
            batches = []
            for _ in range(args.local_epochs):
                perm = idx[torch.randperm(len(idx), device=device, generator=gen)]
                batches += [perm[i:i + args.batch_size] for i in range(0, len(perm), args.batch_size)]
            out.append(batches)
        return out, gen

    def make_runs(seed):
        sch, _ = schedules(seed)

        def draw(key, n):
            g = torch.Generator(device=device).manual_seed(seed * 1000003 + key[0] * 10007 + key[1])
            t = torch.randint(0, sched.T, (n,), device=device, generator=g)
            eps = torch.randn((n, 3, 32, 32), device=device, generator=g)
            flip = torch.rand(n, device=device, generator=g) < 0.5
            return t, eps, flip

        def client_loss(i):
            def f(m, k):
                idx = sch[i][k]
                t, eps, flip = draw((i, k), len(idx))
                x0 = torch.where(flip.view(-1, 1, 1, 1), x_all[idx].flip(3), x_all[idx])
                return denoising_loss(m, x0, t, eps, sched)
            return f

        def pooled_loss(m, k):
            xs, ts, es = [], [], []
            for i in range(len(parts)):
                idx = sch[i][k]
                t, eps, flip = draw((i, k), len(idx))
                xs.append(torch.where(flip.view(-1, 1, 1, 1), x_all[idx].flip(3), x_all[idx]))
                ts.append(t)
                es.append(eps)
            return denoising_loss(m, torch.cat(xs), torch.cat(ts), torch.cat(es), sched)

        steps = len(sch[0])
        return [(client_loss(i), len(sch[i])) for i in range(len(parts))], pooled_loss, steps

    def est_batches(seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        pick = [p[torch.randperm(len(p), device=device, generator=gen)[:2 * args.n_est]] for p in parts]
        n = args.n_est

        def mk(lo, hi, half):
            t = torch.randint(lo, hi, (n,), device=device, generator=gen)
            eps = torch.randn((n, 3, 32, 32), device=device, generator=gen)
            return [(x_all[p[half * n:(half + 1) * n]], t, eps) for p in pick]

        a = mk(0, sched.T, 0)
        b = mk(0, sched.T, 1)
        bands = [mk(200 * k, 200 * k + 200, 1) for k in range(5)]
        return a, b, bands

    base, est_b = {}, {}
    for tag, seed in [("A", 101), ("B", 202)]:
        a, b, bands = est_batches(seed)
        base[tag] = base_predictors(grad_fn, w0, pi, a, bands)
        est_b[tag] = b
        print(f"C estimate {tag} done | |gbar| {base[tag]['gbar'].norm():.4f} |C| {base[tag]['C'].norm():.4f}", flush=True)

    settings = [("adam", args.lr, 1.0)] if args.opt == "adam" else [("sgd", lr, None) for lr in args.lrs]
    tau = make_runs(11)[2]
    out = {"cos_C_A_C_B": cos(base["A"]["C"], base["B"]["C"])}
    for kind, lr, clip in settings:
        key = f"{kind}_lr{lr:g}"
        runs = {}
        for tag, seed in [("A", 11), ("B", 22)]:
            clients, pooled_loss, steps = make_runs(seed)
            fa, deltas, vs = fedavg_round(model, w0, clients, pi, lr, kind, use_amp, clip)
            dp, _ = run_steps(model, w0, steps, pooled_loss, lr, kind, use_amp, clip)
            runs[tag] = (fa - dp, deltas, vs, dp, fa)
            print(f"{key} run {tag}: |fedavg| {fa.norm():.4f} |pooled| {dp.norm():.4f} |error| {(fa - dp).norm():.4f} | {steps} steps", flush=True)
        if not all(torch.isfinite(runs[t][0]).all() for t in runs):
            out[key] = {"verdict": "DIVERGED"}
            print(f"{key}: diverged", flush=True)
            continue
        P = {t: {**base[t], **update_predictors(grad_fn, w0, pi, est_b[t], runs[t][1], runs[t][2])} for t in runs}
        eA, eB = runs["A"][0], runs["B"][0]
        res = evaluate(eA, eB, P["A"], P["B"])
        res["cos_pooledA_pooledB"] = cos(runs["A"][3], runs["B"][3])
        res["cos_fedavgA_fedavgB"] = cos(runs["A"][4], runs["B"][4])
        res["norm_pooled"] = runs["A"][3].norm().item()
        res["error_over_pooled"] = (eA.norm() / runs["A"][3].norm()).item()
        res["kappa_theory_C"] = lr ** 2 * tau * (tau - 1) / 2
        res["kappa_theory_Cu"] = -lr * (tau - 1) / 2
        res["bands"] = band_report(eA, P["A"]["Cb"])
        res["verdict"] = verdict(res)
        out[key] = res
        print(f"{key}: |pooled| {res['norm_pooled']:.3f} | error/pooled {res['error_over_pooled']:.3f} | cos(eA,eB) {res['cos_eA_eB']:.3f} | "
              f"cos(e,C) {res['C']['cos_A']:+.3f}/{res['C']['cos_B']:+.3f} | cos(e,Cu) {res['Cu']['cos_A']:+.3f}/{res['Cu']['cos_B']:+.3f} | "
              f"beta C {res['C']['beta_fit_A']:.2e} vs theory {res['kappa_theory_C']:.2e} | {res['verdict']}", flush=True)
        json.dump(out, open(args.out + ".json", "w"), indent=1)
    try:
        log = json.load(open(args.ckpt.replace("_round100.pt", ".json")))
        out["zeta2_round100"] = [r["zeta2_cross"] for r in log["drift"] if r["round"] == 100 and r["alpha"] == "class"]
    except Exception:
        pass
    json.dump(out, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
