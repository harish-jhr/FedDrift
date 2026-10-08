"""FedAvg DDPM on CIFAR-10 with FID evaluated at checkpoints. One run = one client partition.

The only thing that should differ between the iid / alpha0.5 / alpha0.1 runs is the partition:
same init weights, same optimizer / lr / epochs / batch size, same per-(round, client) training seeds,
same FID reference set, same evaluation noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import gc
import glob
import json
import os
import re
import time
from copy import deepcopy

import numpy as np
import torch
from torchvision.utils import save_image

from data import load_cifar10, load_partition, partition_path
from diffusion import Schedule, ddpm_sample
from fed import fedavg, local_train
from fid import FIDScorer, to_uint8
from model import UNet


def parse_alpha(s):
    return None if s == "iid" else float(s)


def tag(alpha):
    return "iid" if alpha is None else f"alpha{alpha:g}"   # same tag as the partition files


def client_seed(seed, rnd, k):
    # every (round, client) gets its own deterministic seed. the three runs then see the same
    # t / eps / flip streams per client index, and a resumed run is identical to an uninterrupted one
    return seed * 1_000_003 + rnd * 1009 + k


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _release_device_cache(device):
    # Apple Silicon uses unified memory: release cached MPS blocks between clients.
    gc.collect()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_round(model, x_all, parts, sched, args, rnd):
    # Keep each client's trained weights on CPU before moving to the next client.
    # The old code retained all client state_dict() tensors on MPS/unified memory.
    states, sizes, losses, n_steps = [], [], [], 0
    for k, idx in enumerate(parts):
        torch.manual_seed(client_seed(args.seed, rnd, k))
        local = deepcopy(model)
        loss, steps = local_train(local, x_all, idx, sched, args.local_epochs, args.batch_size, args.lr,
                                  max_steps=args.max_steps_per_client)

        cpu_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in local.state_dict().items()
        }
        states.append(cpu_state)
        sizes.append(len(idx))
        losses.append(loss)
        n_steps += steps

        del local, cpu_state
        _release_device_cache(x_all.device)

    averaged = fedavg(states, sizes)
    model.load_state_dict(averaged)
    del states, averaged
    _release_device_cache(x_all.device)

    return sum(l * n for l, n in zip(losses, sizes)) / sum(sizes), n_steps


@torch.inference_mode()
def generate(model, n, sched, device, seed, batch, amp):
    """Generate DDPM samples without retaining an autograd graph."""
    model.eval()
    devs = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    out, bad = [], 0
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        for i in range(0, n, batch):
            b = min(batch, n - i)
            with torch.autocast(device.type, enabled=amp):
                x = ddpm_sample(model, (b, 3, 32, 32), sched, device)
            bad += int((~torch.isfinite(x)).flatten(1).any(1).sum())
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            out.append(to_uint8(x).cpu())
            del x
            _release_device_cache(device)
    return torch.cat(out), bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=parse_alpha, required=True, help="Dirichlet alpha, or 'iid'")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--model", default="small", choices=["tiny","small", "standard"])
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps_per_client", type=int, default=None,
                        help="cap local training batches per client; useful for smoke tests")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_rounds", type=int, nargs="+", default=[0, 10, 25, 50, 75, 100])
    parser.add_argument("--fid_samples", type=int, default=5000)
    parser.add_argument("--sample_batch", type=int, default=8, help="DDPM sampling batch; kept small for MPS memory")
    parser.add_argument("--eval_seed", type=int, default=1234, help="noise seed for sampling, fixed across checkpoints and runs")
    parser.add_argument("--ref_split", default="train", choices=["train", "test"], help="real images FID is measured against")
    parser.add_argument("--fp32_sampling", action="store_true", help="no autocast while sampling (slower)")
    parser.add_argument("--save_samples", action="store_true", help="also save every generated sample (uint8 npz) per checkpoint")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in the run dir")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--partition_dir", default="./partitions")
    parser.add_argument("--out_dir", default="./runs_fid")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fid_device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="device for Inception/FID; auto uses CPU on MPS to protect unified memory")
    parser.add_argument("--fid_batch", type=int, default=32,
                        help="Inception batch size for FID feature extraction")
    parser.add_argument("--skip_fid", action="store_true",
                        help="skip image generation/FID entirely; useful for a smoke test")
    args = parser.parse_args()

    eval_rounds = sorted(set(args.eval_rounds))
    if eval_rounds[0] < 0 or eval_rounds[-1] > args.rounds:
        parser.error(f"--eval_rounds must lie in [0, {args.rounds}]")
    if args.fid_samples <= 2048:
        print("WARNING: fid_samples <= 2048, the 2048-d feature covariance is singular, FID will be unreliable")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = os.path.join(args.out_dir, tag(args.alpha))
    os.makedirs(run_dir, exist_ok=True)
    fid_path = os.path.join(run_dir, "fid.json")

    # same order as measure_drift.py, so the init weights are identical across the three runs
    torch.manual_seed(args.seed)
    x_all, _ = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)
    part_file = partition_path(args.partition_dir, args.n_clients, args.alpha, args.seed)
    parts = [torch.from_numpy(p).to(device) for p in load_partition(part_file)]
    assert sum(len(p) for p in parts) == len(x_all) and len(parts) == args.n_clients, "partition does not cover the dataset"
    model = UNet(args.model).to(device)
    print(f"{args.model} unet | {model.num_params() / 1e6:.1f}M params | {device} | {tag(args.alpha)} | "
          f"client sizes {[len(p) for p in parts]}")

    # Do not even import/initialize FID when --skip_fid is requested.
    # In particular, FIDScorer would otherwise compute Inception statistics for
    # all 50,000 CIFAR-10 reference images before the training loop starts.
    pytorch_fid_version = "skipped"
    if not args.skip_fid:
        import pytorch_fid
        pytorch_fid_version = getattr(pytorch_fid, "__version__", "?")
    config = {
        "alpha": "iid" if args.alpha is None else args.alpha, "seed": args.seed, "n_clients": args.n_clients,
        "rounds": args.rounds, "local_epochs": args.local_epochs, "batch_size": args.batch_size, "lr": args.lr,
        "model": args.model, "fid_samples": args.fid_samples, "sample_batch": args.sample_batch,
        "eval_rounds": eval_rounds, "eval_seed": args.eval_seed, "ref_split": args.ref_split,
        "amp_sampling": (device.type == "cuda" and not args.fp32_sampling),
        "partition_file": part_file, "sampler": "ddpm ancestral, T=1000, sigma^2=beta, no EMA",
        "fid_impl": "pytorch-fid InceptionV3 (FID weights, pool3 2048-d), own Frechet distance",
        "versions": {"torch": torch.__version__, "pytorch_fid": pytorch_fid_version},
        "skip_fid": args.skip_fid,
        "max_steps_per_client": args.max_steps_per_client,
    }
    same_keys = ["alpha", "seed", "n_clients", "local_epochs", "batch_size", "lr", "model",
                 "fid_samples", "sample_batch", "fid_batch", "fid_device", "eval_seed", "ref_split", "amp_sampling", "partition_file"]

    log = {"config": config, "client_sizes": [len(p) for p in parts], "train_loss": {}, "fid": {}, "eval": {}}
    start = 0
    if os.path.exists(fid_path):
        if not args.resume:
            raise SystemExit(f"{fid_path} exists. use --resume to continue it, or a different --out_dir")
        old = json.load(open(fid_path))
        diff = [k for k in same_keys if old["config"].get(k) != config[k]]
        if diff:
            raise SystemExit(f"cannot resume, config differs in {diff}")
        log["train_loss"], log["fid"], log["eval"] = old["train_loss"], old["fid"], old["eval"]
        ckpts = {int(re.search(r"round_(\d+)", p).group(1)): p
                 for p in glob.glob(os.path.join(run_dir, "checkpoint_round_*.pt"))}
        ckpts = {r: p for r, p in ckpts.items() if r <= args.rounds}
        if ckpts:
            start = max(ckpts)
            model.load_state_dict(torch.load(ckpts[start], map_location=device)["model"])
            print(f"resumed from {ckpts[start]}")
    elif args.resume:
        print("--resume given but nothing to resume, starting fresh")

    if args.fid_device == "auto":
        fid_device = torch.device("cpu") if device.type == "mps" else device
    else:
        fid_device = torch.device(args.fid_device)

    amp = config["amp_sampling"]
    fid = None
    if not args.skip_fid:
        print(f"FID device: {fid_device} | FID batch: {args.fid_batch} | sampling batch: {args.sample_batch}")
        fid = FIDScorer(
            fid_device, args.data_root, args.ref_split,
            cache_dir=args.out_dir, batch_size=args.fid_batch
        )
        log["config"]["n_ref_images"] = fid.n_ref
        log["config"]["fid_device"] = str(fid_device)
    else:
        print("FID skipped: no Inception model or 50,000-image reference pass will run.")

    for rnd in range(start, args.rounds + 1):
        if rnd > start:
            t0 = time.time()
            loss, n_steps = train_round(model, x_all, parts, sched, args, rnd)
            dt = time.time() - t0
            log["train_loss"][str(rnd)] = loss
            print(f"round {rnd}/{args.rounds} | loss {loss:.4f} | {n_steps} steps in {dt:.0f}s ({n_steps / dt:.1f} it/s)")

        if rnd in eval_rounds:
            ckpt = os.path.join(run_dir, f"checkpoint_round_{rnd}.pt")
            if not os.path.exists(ckpt):
                torch.save({"round": rnd, "alpha": config["alpha"], "seed": args.seed, "model": model.state_dict()}, ckpt)
            if args.skip_fid:
                # Keep checkpointing/round training available without launching the
                # 1000-step DDPM sampler. This is the safe smoke-test mode.
                write_json(fid_path, log)
                continue
            if str(rnd) not in log["fid"]:
                t0 = time.time()
                imgs, bad = generate(model, args.fid_samples, sched, device, args.eval_seed, args.sample_batch, amp)
                score = fid.score(imgs)
                log["fid"][str(rnd)] = score
                log["eval"][str(rnd)] = {"seconds": time.time() - t0, "nonfinite_samples": bad,
                                         "pixel_mean": imgs.float().mean().item(), "pixel_std": imgs.float().std().item()}
                save_image(imgs[:64].float() / 255.0, os.path.join(run_dir, f"samples_round_{rnd}.png"), nrow=8)
                if args.save_samples:
                    np.savez_compressed(os.path.join(run_dir, f"samples_round_{rnd}.npz"), imgs=imgs.numpy())
                print(f"  FID @ round {rnd}: {score:.2f} | {args.fid_samples} samples | nonfinite {bad} | "
                      f"{time.time() - t0:.0f}s")
        write_json(fid_path, log)

    print(f"done -> {fid_path}")


if __name__ == "__main__":
    main()