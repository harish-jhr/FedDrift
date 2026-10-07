import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "0.1": ("#2a78d6", "o"),
    "0.5": ("#eb6834", "s"),
    "5": ("#1baf7a", "^"),
    "iid": ("#eda100", "D"),
}

PANELS = [
    ("zeta2_cross_normalized", "dissimilarity / |mean grad|^2", "linear"),
    ("drift_over_noise", "dissimilarity / within-client variance", "linear"),
    ("zeta2_corrected", "dissimilarity (half-split corrected)", "log"),
    ("mean_pairwise_cos", "mean pairwise cosine", "linear"),
    ("within_client_var", "within-client variance", "log"),
    ("mean_grad_norm2", "|mean grad|^2", "log"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--round", type=int, default=None) 
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.json) as f:
        log = json.load(f)
    rows = log["drift"]
    rnd = args.round if args.round is not None else max(r["round"] for r in rows)
    rows = [r for r in rows if r["round"] == rnd]
    for r in rows:
        r["drift_over_noise"] = r["zeta2_cross"] / r["within_client_var"]
    alphas = [a for a in STYLE if any(r["alpha"] == a for r in rows)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, scale) in zip(axes.flat, PANELS):
        for a in alphas:
            sel = [r for r in rows if r["alpha"] == a]
            xs = [f"{r['bucket'][0]}-{r['bucket'][1]}" for r in sel]
            color, marker = STYLE[a]
            ax.plot(xs, [r[key] for r in sel], color=color, marker=marker, lw=2, ms=7,
                    label="IID" if a == "iid" else f"alpha={a}")
        ax.set_yscale(scale)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("timestep bucket (low noise -> high noise)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)

    a = log["args"]
    fig.suptitle(f"client gradient dissimilarity per timestep bucket | round {rnd} | "
                 f"FedAvg trained on alpha={a['alpha'] or 'iid'}, {a['n_clients']} clients")
    fig.tight_layout()
    out = args.out or args.json.replace(".json", f"_round{rnd}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
