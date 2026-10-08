import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# run dir name, legend label, color, marker (colors match plot.py)
RUNS = [
    ("iid", "IID", "#eda100", "D"),
    ("alpha0.5", "alpha=0.5", "#eb6834", "s"),
    ("alpha0.1", "alpha=0.1", "#2a78d6", "o"),
]
SAME = ["seed", "n_clients", "local_epochs", "batch_size", "lr", "model", "fid_samples", "eval_seed", "ref_split"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./runs_fid")
    parser.add_argument("--out", default=None)
    parser.add_argument("--linear", action="store_true", help="linear y axis (default is log, round 0 is huge)")
    args = parser.parse_args()

    runs = []
    for name, label, color, marker in RUNS:
        path = os.path.join(args.root, name, "fid.json")
        if not os.path.exists(path):
            print(f"missing {path}, skipping")
            continue
        with open(path) as f:
            log = json.load(f)
        pts = sorted((int(r), v) for r, v in log["fid"].items())
        if pts:
            runs.append((label, color, marker, log["config"], pts))
    if not runs:
        raise SystemExit("no fid.json found")

    # the comparison is only fair if everything but the partition matches
    cfg0 = runs[0][3]
    mismatch = sorted({k for _, _, _, c, _ in runs for k in SAME if c.get(k) != cfg0.get(k)})
    if mismatch:
        print(f"WARNING: runs differ in {mismatch}, comparison is not controlled")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    for label, color, marker, _, pts in runs:
        ax.plot([r for r, _ in pts], [v for _, v in pts], color=color, marker=marker, lw=2, ms=7, label=label)
    ax.set_xlabel("communication round")
    ax.set_ylabel("FID ↓")
    if not args.linear:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("FID vs communication round", fontsize=11)

    # final-round comparison, at the last round every run has
    common = set.intersection(*[{r for r, _ in pts} for *_, pts in runs])
    last = max(common) if common else None
    if last is not None:
        vals = [dict(pts)[last] for *_, pts in runs]
        bars = bx.bar([r[0] for r in runs], vals, color=[r[1] for r in runs])
        for b, v in zip(bars, vals):
            bx.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
        bx.set_title(f"FID at round {last}", fontsize=11)
        bx.set_ylabel("FID ↓")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"FedAvg DDPM, CIFAR-10, {cfg0['n_clients']} clients | seed {cfg0['seed']}, single run | "
                 f"FID from {cfg0['fid_samples']} samples vs {cfg0['ref_split']} set"
                 + (" | CONFIGS DIFFER" if mismatch else ""), fontsize=11)
    fig.tight_layout()
    out = args.out or os.path.join(args.root, "fid_vs_round.png")
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")

    print("\nround " + "".join(f"{r[0]:>12}" for r in runs))
    for rnd in sorted({r for *_, pts in runs for r, _ in pts}):
        print(f"{rnd:>5} " + "".join(f"{dict(pts).get(rnd, float('nan')):>12.2f}" for *_, pts in runs))


if __name__ == "__main__":
    main()