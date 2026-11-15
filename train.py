# Federated training of a diffusion model, with drift measured per timestep band.
#
# One script runs every experiment in the repo: FedAvg, function-space FedProx,
# distillation, shared images, pooled-band training and RegMean aggregation.
# See README.md for the commands used for each result.
import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid, save_image

import data as data_mod
from data import CHANNELS, DATASETS, IMG_SIZE, alpha_tag, load_cifar100, load_dataset, load_partition, parse_alpha, partition_path
from diffusion import Schedule, q_sample, target
from drift import measure_drift
from fed import control_variate, fedavg, local_train
from model import UNet
from regmean import band_strengths, grams, merge
from sample import generate


def to_wandb(x):
    return wandb.Image(make_grid(x[:64], nrow=8, padding=2).permute(1, 2, 0).cpu().numpy())


def get_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group("data and model")
    g.add_argument("--dataset", default="cifar10", choices=DATASETS)
    g.add_argument("--objective", default="ddpm", choices=["ddpm", "fm"], help="ddpm predicts noise, fm is flow matching")
    g.add_argument("--alpha", type=parse_alpha, default="class", help="client split: class, iid or a Dirichlet alpha")
    g.add_argument("--n_clients", type=int, default=10)
    g.add_argument("--model", default="small", choices=["small", "standard"])
    g.add_argument("--data_root", default="./data")
    g.add_argument("--partition_dir", default="./partitions")

    g = p.add_argument_group("training")
    g.add_argument("--rounds", type=int, default=100)
    g.add_argument("--local_epochs", type=int, default=5)
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--ema_decay", type=float, default=0.8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--init", default=None, help="start from these weights")

    g = p.add_argument_group("drift measurement and logging")
    g.add_argument("--measure_alphas", type=parse_alpha, nargs="+", default=["class"])
    g.add_argument("--measure_rounds", type=int, nargs="+", default=[25, 50, 100])
    g.add_argument("--measure_batch", type=int, default=256)
    g.add_argument("--repeats", type=int, default=8)
    g.add_argument("--n_samples", type=int, default=64)
    g.add_argument("--sample_every", type=int, default=0)
    g.add_argument("--bands", type=int, default=5, help="number of equal timestep bands")
    g.add_argument("--out_dir", default="./runs")
    g.add_argument("--name", default=None)
    g.add_argument("--wandb_project", default="fed-diffusion")
    g.add_argument("--no_wandb", action="store_true")
    g.add_argument("--resume_round", type=int, default=0, help="continue from the checkpoint saved at this round")
    g.add_argument("--resume_last", action="store_true", help="continue from the latest checkpoint")

    g = p.add_argument_group("FedProx and distillation (penalty on predictions vs the global model)")
    g.add_argument("--prox", type=float, default=0.0, help="FedProx strength")
    g.add_argument("--prox_range", type=int, nargs=2, default=None, help="apply only in this timestep range")
    g.add_argument("--prox_adapt", action="store_true", help="strength per band proportional to measured drift")
    g.add_argument("--prox_ramp", type=int, default=0)
    g.add_argument("--prox_clip", type=float, nargs=2, default=[0.2, 3.0])
    g.add_argument("--distill", type=float, default=0.0, help="distillation strength on server-generated images")
    g.add_argument("--distill_n", type=int, default=1000)
    g.add_argument("--distill_every", type=int, default=10)
    g.add_argument("--distill_batch", type=int, default=32)
    g.add_argument("--distill_start", type=int, default=0)

    g = p.add_argument_group("shared and pooled data (references, not data-free)")
    g.add_argument("--share_size", type=int, default=0, help="size of the shared image set given to every client")
    g.add_argument("--share_source", default="cifar10_test", choices=["cifar10_test", "cifar100"])
    g.add_argument("--share_range", type=int, nargs=2, default=[0, 1000], help="timesteps at which shared images are used")
    g.add_argument("--share_batch", type=int, default=32)
    g.add_argument("--pool_range", type=int, nargs=2, default=None, help="pooled-band training in this timestep range")
    g.add_argument("--pool_size", type=int, default=0)

    g = p.add_argument_group("SCAFFOLD-style correction")
    g.add_argument("--scaffold", action="store_true")
    g.add_argument("--scaffold_range", type=int, nargs=2, default=None)
    g.add_argument("--scaffold_scale", type=float, default=1.0)
    g.add_argument("--cv_batches", type=int, default=8)

    g = p.add_argument_group("RegMean aggregation")
    g.add_argument("--regmean", action="store_true", help="merge with RegMean instead of FedAvg")
    g.add_argument("--regmean_rho", default="uniform", choices=["uniform", "adaptive"])
    g.add_argument("--regmean_a", type=float, default=0.9, help="off-diagonal shrinkage of the Gram matrices")
    g.add_argument("--regmean_n", type=int, default=1024, help="noisy images per band for the Gram matrices")
    g.add_argument("--regmean_rel", type=float, default=1e-4, help="ridge as a fraction of the mean diagonal")
    g.add_argument("--regmean_beta", type=float, default=1.0, help="memory coefficient; 1 turns memory off")
    g.add_argument("--regmean_strength", type=float, default=1.0)
    g.add_argument("--regmean_warmup", type=int, default=10, help="first round at which adaptive strengths are measured")
    g.add_argument("--regmean_every", type=int, default=10)
    g.add_argument("--regmean_tau", type=float, default=0.05)
    g.add_argument("--regmean_clip", type=float, nargs=2, default=[0.2, 2.0])
    g.add_argument("--regmean_val", type=int, default=500, help="held-out images per client for adaptive strengths")
    g.add_argument("--heldout_n", type=int, default=1000)
    g.add_argument("--ref_norm", default="", help="FedAvg checkpoint pattern, e.g. runs/fedavg_s0_round{}.pt, for weight-norm logging")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    nb = args.bands

    name = args.name or f"{args.dataset}_{alpha_tag(args.alpha)}_s{args.seed}"
    resuming = args.resume_last or args.resume_round
    run_id = f"{args.name}_r" if (resuming and args.name) else args.name
    wandb.init(project=args.wandb_project, name=run_id or name, id=run_id, config=vars(args),
               mode="disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "online"),
               resume="allow" if resuming else None)

    data_mod.FLIP = args.dataset != "mnist"
    x_all, y_all = load_dataset(args.dataset, args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device, kind=args.objective)
    if args.share_source == "cifar100" and args.share_size and args.dataset != "cifar10":
        raise ValueError("--share_source cifar100 is only meant for cifar10")

    def get_parts(alpha):
        path = partition_path(args.partition_dir, args.n_clients, alpha, args.seed, args.dataset)
        return [torch.from_numpy(p).to(device) for p in load_partition(path)]

    train_parts = get_parts(args.alpha)
    measure_parts = {alpha_tag(a): get_parts(a) for a in args.measure_alphas}
    pool_idx = None
    if args.pool_size:
        pool_idx = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(args.seed))[:args.pool_size].to(device)
    share = None
    if args.share_size:
        x_s, _ = load_cifar100(args.data_root) if args.share_source == "cifar100" else load_dataset(args.dataset, args.data_root, train=False)
        pick = torch.randperm(len(x_s), generator=torch.Generator().manual_seed(args.seed))[:args.share_size]
        share = x_s[pick].to(device)
        del x_s

    model = UNet(args.model, IMG_SIZE[args.dataset], CHANNELS[args.dataset]).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    ema = deepcopy(model).eval().requires_grad_(False)
    print(f"{name} | {args.dataset} | {args.objective} | {model.num_params() / 1e6:.1f}M params | {device}")

    base = os.path.join(args.out_dir, name)
    log = {"args": vars(args), "train_loss": {}, "drift": []}
    prox_rel = torch.ones(nb, device=device) if args.prox_adapt else None
    probes = None

    # Measures drift per band on every requested split, saves checkpoints and sample grids.
    # For drift-adaptive FedProx / distillation it also sets the new strength per band from
    # the measured drift (strength proportional to drift, clipped, mean kept fixed).
    def measure(rnd):
        nonlocal prox_rel
        t0 = time.time()
        scalars = {}
        for split, parts in measure_parts.items():
            res = measure_drift(model, x_all, parts, sched, batch=args.measure_batch, repeats=args.repeats, seed=args.seed + rnd)
            if args.prox_adapt and rnd > 0 and split == alpha_tag(args.alpha) and len(res) == nb:
                z = torch.tensor([max(r["zeta2_cross"], 0.0) for r in res], device=device)
                if z.sum() > 0:
                    rel = (z / z.mean()).clamp(*args.prox_clip)
                    prox_rel = rel / rel.mean()
                    log.setdefault("prox_rel", {})[rnd] = prox_rel.tolist()
                    lam = args.distill or args.prox
                    print(f"  round {rnd} | strength per band from drift: " + "  ".join(f"{lam * v:.3f}" for v in prox_rel.tolist()))
            for r in res:
                log["drift"].append({"round": rnd, "alpha": split, **r})
                lo, hi = r["bucket"]
                scalars[f"drift/{split}/t{lo}-{hi}"] = r["zeta2_cross"]
                scalars[f"drift_over_grad/{split}/t{lo}-{hi}"] = r["zeta2_cross_normalized"]
            print(f"  round {rnd} | alpha {split:>5} | drift per band: " + "  ".join(f"{r['zeta2_cross']:9.2e}" for r in res))
        with open(base + ".json", "w") as f:
            json.dump(log, f, indent=1)
        if rnd > 0:
            torch.save(model.state_dict(), f"{base}_round{rnd}.pt")
            torch.save(ema.state_dict(), f"{base}_round{rnd}_ema.pt")
            x = generate(model.eval(), args.n_samples, sched, device, args.seed)
            x_ema = generate(ema, args.n_samples, sched, device, args.seed)
            save_image(x, f"{base}_round{rnd}_samples.png", nrow=8, padding=2)
            save_image(x_ema, f"{base}_round{rnd}_ema_samples.png", nrow=8, padding=2)
            scalars["samples"], scalars["samples_ema"] = to_wandb(x), to_wandb(x_ema)
        wandb.log(scalars, step=rnd)
        print(f"  measured in {time.time() - t0:.0f}s -> {base}.json")

    # denoising loss of a model in each band on the images x (same noise every call for a seed)
    @torch.no_grad()
    def band_losses(net, x, seed):
        net.eval()
        g = torch.Generator(device=device).manual_seed(seed)
        out = []
        for b in range(nb):
            lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
            t = torch.randint(lo, hi, (len(x),), device=device, generator=g)
            eps = torch.randn(x.shape, device=device, generator=g)
            tot = 0.0
            for i in range(0, len(x), 250):
                o = net(q_sample(x[i:i + 250], t[i:i + 250], eps[i:i + 250], sched), t[i:i + 250]).float()
                tot += (o - target(x[i:i + 250], eps[i:i + 250], sched)).pow(2).mean(dim=(1, 2, 3)).sum().item()
            out.append(tot / len(x))
        return out

    # RegMean setup. For adaptive strengths every client needs a small held-out set that looks
    # like its own data (test images of its own class), and we keep 1000 random test images to
    # log the held-out loss per band every round.
    rg = None
    if args.regmean:
        x_te, y_te = load_dataset(args.dataset, args.data_root, train=False)
        gsel = torch.Generator().manual_seed(args.seed + 4242)
        nc = int(max(y_all.max(), y_te.max())) + 1
        te_count = np.bincount(y_te, minlength=nc).astype(np.float64)
        val = []
        for idx in train_parts:
            mix = np.bincount(y_all[idx.cpu().numpy()], minlength=nc).astype(np.float64)
            if (mix > 0).sum() == 1:
                ids = np.where(y_te == mix.argmax())[0]
                ids = ids[torch.randperm(len(ids), generator=gsel)[:args.regmean_val].numpy()]
            else:
                w = torch.from_numpy(mix[y_te] / np.maximum(te_count[y_te], 1))
                ids = np.sort(torch.multinomial(w, min(args.regmean_val, int((w > 0).sum())), replacement=False, generator=gsel).numpy())
            val.append(x_te[ids].to(device))
        hold = x_te[torch.randperm(len(x_te), generator=gsel)[:args.heldout_n]].to(device)
        del x_te
        rg = {"val": val, "hold": hold, "rho": [args.regmean_strength] * nb, "signal": None, "C": {}}

    # resuming: --resume_last picks up from the latest checkpoint after a crash, --resume_round
    # starts from a saved round (this is how the warm start continues from FedAvg's round 25)
    start_round = 1
    if args.resume_last:
        st = torch.load(base + "_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        log = st["log"]
        if rg is not None and st.get("regmean"):
            rg.update(st["regmean"])
        if args.prox_adapt:
            prox_rel = st["prox_rel"].to(device)
        torch.set_rng_state(st["rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in st:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        start_round = st["round"] + 1
    elif args.resume_round:
        r = args.resume_round
        model.load_state_dict(torch.load(f"{base}_round{r}.pt", map_location=device))
        ema.load_state_dict(torch.load(f"{base}_round{r}_ema.pt", map_location=device))
        with open(base + ".json") as f:
            log = json.load(f)
        torch.manual_seed(args.seed * 1000 + r)
        start_round = r + 1
    if start_round > 1:
        print(f"resumed at round {start_round}")
    if 0 in args.measure_rounds and start_round == 1:
        measure(0)

    for rnd in range(start_round, args.rounds + 1):
        t0 = time.time()

        # SCAFFOLD-style correction: every client gets (average gradient - its own gradient)
        corrs = [None] * len(train_parts)
        if args.scaffold:
            lo, hi = args.scaffold_range or (0, sched.T)
            cs = [control_variate(model, x_all, idx, sched, args.scaffold_range, args.cv_batches, args.batch_size) for idx in train_parts]
            w = [len(idx) / sum(len(i) for i in train_parts) for idx in train_parts]
            cbar = [sum(wk * c[j] for wk, c in zip(w, cs)) for j in range(len(cs[0]))]
            corrs = [[args.scaffold_scale * (hi - lo) / sched.T * (cb - ci) for cb, ci in zip(cbar, c)] for c in cs]
            del cs, cbar

        # FedProx / distillation strength for this round (ramped up over the first rounds), and for
        # distillation, fresh images from the global model every distill_every rounds
        k = rnd - args.distill_start if args.distill else rnd
        ramp = 0.0 if k <= 0 else (min(1.0, k / args.prox_ramp) if args.prox_ramp else 1.0)
        lam = (args.distill or args.prox) * ramp
        if args.distill and k > 0 and (probes is None or (k - 1) % args.distill_every == 0):
            probes = generate(ema, args.distill_n, sched, device, seed=args.seed * 7919 + rnd, batch=250) * 2 - 1
            save_image((probes[:64] + 1) / 2, f"{base}_probes_round{rnd}.png", nrow=8, padding=2)

        # local training: every client starts from the global model and trains on its own images
        states, sizes, losses, ratios, devs, n_steps = [], [], [], [], [], 0
        for idx, corr in zip(train_parts, corrs):
            local = deepcopy(model)
            ds = [torch.zeros(nb, device=device), torch.zeros(nb, device=device)] if (args.prox or args.distill) else None
            loss, steps, ratio = local_train(local, x_all, idx, sched, args.local_epochs, args.batch_size, args.lr,
                                             pool_range=args.pool_range, pool_idx=pool_idx, ref=model,
                                             prox=0.0 if args.distill else lam, prox_range=args.prox_range, prox_rel=prox_rel,
                                             corr=corr, share=share, share_range=args.share_range, share_batch=args.share_batch,
                                             dev_stats=ds, probes=probes, distill=lam if args.distill else 0.0,
                                             distill_batch=args.distill_batch)
            if ds is not None:
                devs.append(ds[0] / ds[1].clamp(min=1))
            states.append(local.state_dict())
            sizes.append(len(idx))
            losses.append(loss)
            ratios.append(ratio)
            n_steps += steps

        # how different the client updates are from each other (just for logging), then FedAvg
        start = {k_: v.float() for k_, v in model.state_dict().items()}
        ups = [torch.cat([(st[k_].float() - start[k_]).flatten() for k_ in start]) for st in states]
        ubar = sum(ups) / len(ups)
        spread = (sum((u - ubar).pow(2).sum() for u in ups) / sum(u.pow(2).sum() for u in ups)).item()
        del start, ups, ubar
        prev = {k_: v.clone() for k_, v in model.state_dict().items()} if rg is not None else None
        model.load_state_dict(fedavg(states, sizes))

        # RegMean: replace the FedAvg result for the conv and linear layers.
        # Every regmean_every rounds (adaptive only) we first measure, per band, how much each
        # client's local training lowered the loss on its held-out images, and turn that into
        # new band strengths. Then every client's Gram matrices are collected and merged.
        extra = ""
        if rg is not None:
            tg = time.time()
            bands = [(b * sched.T // nb, (b + 1) * sched.T // nb) for b in range(nb)]
            gg = torch.Generator(device=device).manual_seed(args.seed * 7717 + rnd)
            due = (args.regmean_rho == "adaptive" and rnd > args.regmean_warmup
                   and (rnd - args.regmean_warmup - 1) % args.regmean_every == 0)
            net = deepcopy(model)
            if due:
                net.load_state_dict(prev)
                l0 = [band_losses(net, v, args.seed * 31 + i) for i, v in enumerate(rg["val"])]
                gain = [0.0] * nb
            G = []
            for i, (st, idx) in enumerate(zip(states, train_parts)):
                net.load_state_dict(st)
                G.append(grams(net, x_all[idx], sched, bands, args.regmean_n, 256, gg))
                if due:
                    li = band_losses(net, rg["val"][i], args.seed * 31 + i)
                    for b in range(nb):
                        gain[b] += (l0[i][b] - li[b]) / l0[i][b] / len(states)
            del net
            if due:
                rg["rho"] = band_strengths(gain, args.regmean_tau, args.regmean_clip, args.regmean_strength)
                rg["signal"] = gain
            pi = [n_ / sum(sizes) for n_ in sizes]
            merged, rec = merge(prev, states, G, pi, rg["rho"], args.regmean_a, args.regmean_rel, args.regmean_beta, rg["C"])
            sd = model.state_dict()
            sd.update(merged)
            model.load_state_dict(sd)
            del G, prev, sd, merged
            rec["weight_norm"] = sum(p.detach().float().pow(2).sum().item() for p in model.parameters()) ** 0.5
            rec["rho"], rec["signal"] = list(rg["rho"]), rg["signal"]
            rec["heldout"] = band_losses(model, rg["hold"], args.seed * 53)
            ref_path = args.ref_norm.format(rnd) if args.ref_norm else ""
            if ref_path and os.path.exists(ref_path):
                ref_sd = torch.load(ref_path, map_location=device)
                rec["weight_norm_vs_ref"] = rec["weight_norm"] / sum(ref_sd[n_].float().pow(2).sum().item() for n_, _ in model.named_parameters()) ** 0.5
                del ref_sd
            log.setdefault("regmean", {})[rnd] = rec
            extra = (f" | regmean |u| {rec['u']:.3f} vs fedavg {rec['fedavg_u']:.3f}"
                     + (f" | applied {rec['applied_u']:.3f} | |C| {rec['C_norm']:.3f}" if args.regmean_beta < 1 else "")
                     + f" | |w| {rec['weight_norm']:.2f} | rho " + " ".join(f"{v:.2f}" for v in rec["rho"])
                     + " | heldout " + " ".join(f"{v:.5f}" for v in rec["heldout"]) + f" | {time.time() - tg:.0f}s")

        # EMA of the global model, this is the model we sample from for FID
        if rnd == 1:
            ema.load_state_dict(model.state_dict())
        else:
            for e, p_ in zip(ema.parameters(), model.parameters()):
                e.data.lerp_(p_.data, 1 - args.ema_decay)

        loss = sum(l_ * n_ for l_, n_ in zip(losses, sizes)) / sum(sizes)
        dt = time.time() - t0
        log["train_loss"][rnd] = loss
        log.setdefault("update_spread", {})[rnd] = spread
        scal = {"loss": loss, "it_per_s": n_steps / dt, "corr_over_grad": sum(ratios) / len(ratios), "update_spread": spread}
        if rg is not None:
            r_ = log["regmean"][rnd]
            for key in ("u", "fedavg_u", "applied_u", "C_norm", "weight_norm", "max_layer_ratio", "weight_norm_vs_ref"):
                if key in r_:
                    scal[f"regmean/{key}"] = r_[key]
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"regmean/rho/t{lo}-{hi}"] = r_["rho"][b]
                scal[f"heldout_loss/t{lo}-{hi}"] = r_["heldout"][b]
        if devs:
            dev = sum(n_ * d for n_, d in zip(sizes, devs)) / sum(sizes)
            used = prox_rel if prox_rel is not None else torch.ones(nb, device=device)
            log.setdefault("prox_dev", {})[rnd] = dev.tolist()
            log.setdefault("prox_strength", {})[rnd] = (lam * used).tolist()
            for b in range(nb):
                lo, hi = b * sched.T // nb, (b + 1) * sched.T // nb
                scal[f"prox_dev/t{lo}-{hi}"] = dev[b].item()
                scal[f"prox_strength/t{lo}-{hi}"] = (lam * used[b]).item()
            extra += " | strength " + " ".join(f"{lam * v:.3f}" for v in used.tolist())
        if args.sample_every and rnd % args.sample_every == 0 and rnd not in args.measure_rounds:
            xs = generate(ema, 64, sched, device, seed=args.seed, batch=64)
            save_image(xs, f"{base}_round{rnd}_ema_samples.png", nrow=8, padding=2)
            scal["samples_ema"] = to_wandb(xs)
        wandb.log(scal, step=rnd)
        print(f"round {rnd}/{args.rounds} | loss {loss:.4f} | update spread {spread:.3f} | {n_steps} steps in {dt:.0f}s{extra}")

        if rnd in args.measure_rounds:
            measure(rnd)

        # save everything needed to resume after a crash
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "round": rnd, "log": log,
                 "rng": torch.get_rng_state(), "prox_rel": prox_rel,
                 "regmean": {"rho": rg["rho"], "signal": rg["signal"], "C": rg["C"]} if rg is not None else None}
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, base + "_last.tmp")
        os.replace(base + "_last.tmp", base + "_last.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
