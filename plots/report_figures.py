# Figures of the final report, drawn from the logged numbers written into this file.
# Run from the repo root: python plots/report_figures.py. Output goes to plots/figures/.
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUES = ["#b9d4f3", "#7fb0ea", "#2a78d6", "#18508f"]
BANDS = ["0–200", "200–400", "400–600", "600–800", "800–1k"]
FS = 7
WIDE = 7.0
COL = 3.35

plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                     "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "pdf.fonttype": 42})


def base(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=2)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def title(ax, s):
    ax.set_title(s, color=INK, loc="left", pad=4)


def bands(ax, rot=False):
    ax.set_xticks(np.arange(5))
    if rot:
        ax.set_xticklabels(BANDS, rotation=30, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(BANDS)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def line(ax, x, y, c, label=None, ls="-", m="o"):
    ax.plot(x, y, color=c, linewidth=1.4, linestyle=ls, marker=m, markersize=3.6, markeredgecolor=SURFACE, markeredgewidth=0.6, label=label)


def diagnosis():
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.1), gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})
    x = np.arange(5)
    ax = axes[0]
    base(ax)
    drift = {10: [4.05e-3, 2.77e-3, 1.12e-3, 2.14e-4, 1.19e-5], 25: [1.35e-3, 2.64e-3, 3.83e-3, 1.42e-3, 1.04e-4],
             50: [9.30e-4, 2.71e-3, 4.58e-3, 2.38e-3, 2.80e-4], 100: [6.33e-4, 2.22e-3, 4.19e-3, 2.41e-3, 2.77e-4]}
    for (r, v), c in zip(drift.items(), BLUES):
        line(ax, x, v, c, f"round {r}")
    ax.set_yscale("log")
    bands(ax, True)
    ax.set_xlabel("Timestep band", color=INK2)
    ax.set_ylabel(r"Client drift $\hat\zeta^2_b$", color=INK2)
    title(ax, "(a) Where clients disagree")
    ax.legend(frameon=False, loc="lower left", handlelength=1.4)

    ax = axes[1]
    base(ax)
    s1 = [-2.45, -1.75, -1.81, 0.22, 0.53]
    ax.bar(x, s1, width=0.6, color=[BLUE if v < 0 else MUTED for v in s1], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(s1):
        ax.text(i, v - 0.12 if v < 0 else v + 0.08, f"{v:+.1f}", ha="center", va="top" if v < 0 else "bottom", color=INK, fontsize=FS - 1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    bands(ax, True)
    ax.set_ylim(-3.1, 1.0)
    ax.set_xlabel("Band taken from the IID model", color=INK2)
    ax.set_ylabel("Change in FID", color=INK2)
    title(ax, "(b) Where it costs: band swap")

    ax = axes[2]
    base(ax)
    names = ["FedAvg", "Pool\n400–800", "Pool\n0–400", "IID"]
    vals = [43.35, 37.91, 29.15, 23.51]
    gaps = ["", "27%", "72%", ""]
    ax.bar(range(4), vals, width=0.62, color=[INK2, BLUE, ORANGE, MUTED], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=FS - 1)
        if gaps[i]:
            ax.text(i, v / 2, gaps[i], ha="center", va="center", color=SURFACE, fontsize=FS - 0.5, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 50)
    ax.set_ylabel("FID, round 100", color=INK2)
    title(ax, "(c) Pooled-band oracle")
    fig.tight_layout(w_pad=1.2)
    save(fig, "r_diagnosis")


def diagnosis_extra():
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0))
    x = np.arange(5)
    ax = axes[0]
    base(ax)
    line(ax, x, [3.56e-3, 1.10e-3, 5.06e-4, 3.25e-4, 3.06e-4], BLUES[1], "round 25")
    line(ax, x, [2.62e-3, 9.59e-4, 4.33e-4, 2.42e-4, 2.27e-4], BLUES[3], "round 100")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("Client output spread", color=INK2)
    title(ax, "(a) Client output disagreement")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    base(ax)
    fork = [-0.45, 1.58, 1.36, -9.6, -109.0]
    shown = [max(v, -12) for v in fork]
    ax.bar(x, shown, width=0.6, color=[ORANGE if v > 0 else BLUE for v in fork], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(fork):
        lab = f"{v:+.1f}%" if abs(v) < 100 else f"{v:+.0f}%"
        ax.text(i, (shown[i] - 0.4) if v < 0 else (v + 0.3), lab, ha="center", va="top" if v < 0 else "bottom", color=INK, fontsize=FS - 1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylim(-14.5, 3.5)
    ax.set_ylabel("Owner's loss vs FedAvg (%)", color=INK2)
    title(ax, "(b) Class-owning client vs FedAvg")

    ax = axes[2]
    base(ax)
    vals = [55.29, 54.89, 45.36]
    ax.bar(range(3), vals, width=0.55, color=[MUTED, INK2, ORANGE], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=FS - 1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Start", "FedAvg\n(1 round)", "Pooled\n(1 round)"])
    ax.set_ylim(0, 64)
    ax.set_ylabel("FID (2k)", color=INK2)
    title(ax, "(c) One round from round 100")
    fig.tight_layout(w_pad=1.2)
    save(fig, "r_diagnosis_extra")


def families():
    fed = ([25, 50, 100], [141.2, 52.0, 43.65])
    panels = [
        ("(a) Function-space FedProx", [("Uniform", [131.7, 52.1, 43.92], BLUE), ("Drift-adaptive", [112.4, 53.3, 43.84], ORANGE)]),
        ("(b) Distillation on generated images", [("Uniform", [143.0, 75.8, 51.9], BLUE), ("Drift-adaptive", [125.4, 74.8, 57.8], ORANGE)]),
        ("(c) Shared images (not data-free)", [("13/batch, all $t$", [116.49, 45.48, 31.88], BLUE), ("13/batch, $t<400$", [136.00, 47.93, 28.75], ORANGE)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=True)
    for ax, (t, rows) in zip(axes, panels):
        base(ax)
        ax.axhline(23.51, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
        line(ax, *fed, INK2, "FedAvg")
        for name, y, c in rows:
            line(ax, fed[0], y, c, name)
        ax.set_yscale("log")
        ax.set_yticks([25, 30, 40, 50, 70, 100, 150])
        ax.set_yticklabels(["25", "30", "40", "50", "70", "100", "150"])
        ax.minorticks_off()
        ax.set_ylim(21, 175)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_xlim(18, 106)
        ax.set_xlabel("Round", color=INK2)
        title(ax, t)
        ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    axes[0].text(60, 24.6, "IID, round 100", color=INK2, fontsize=FS - 1.5, va="bottom")
    axes[0].set_ylabel("FID (log scale)", color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "r_families")


def one_round():
    rows = [
        ("Pooled round (oracle)", 45.36, "ref"),
        ("RegMean", 51.72, "regmean"),
        ("FedAvg", 54.89, "ref"),
        ("Timestep-layer part only", 55.00, "control"),
        ("Gram matrices shuffled", 55.80, "control"),
        ("Fisher merge, drift-weighted", 60.20, "prior"),
        ("FedAvg scaled to RegMean size", 69.72, "control"),
    ]
    col = {"ref": INK2, "regmean": ORANGE, "control": BLUE, "prior": MUTED}
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    base(ax, ygrid=False)
    y = np.arange(len(rows))[::-1]
    ax.axvline(54.89, color=INK2, linewidth=0.8, linestyle=(0, (4, 2)))
    for yi, (name, v, k) in zip(y, rows):
        ax.hlines(yi, 44, v, color=GRID, linewidth=0.8)
        ax.plot(v, yi, "o", color=col[k], markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.text(v + 0.6, yi, f"{v:.2f}", va="center", color=INK, fontsize=FS - 1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(44, 74)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("FID (2k), one round from round 100", color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=4, color=col[k], label=n)
                       for k, n in [("regmean", "RegMean"), ("control", "Control"), ("prior", "Earlier merge"), ("ref", "Reference")]],
              frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=4, handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "r_one_round")


def main_result():
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.1), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axes[0]
    base(ax)
    ax.axhline(43.65, color=INK2, linewidth=0.8, linestyle=(0, (4, 2)))
    line(ax, [50, 75, 100], [52.0, 45.0, 43.65], INK2, "FedAvg")
    line(ax, [50, 75, 100], [56.04, 44.24, 42.54], BLUE, "Uniform RegMean")
    line(ax, [50, 75, 100], [51.79, 42.98, 41.27], ORANGE, "Adaptive RegMean")
    ax.set_xticks([50, 75, 100])
    ax.set_xlim(46, 104)
    ax.set_ylim(39.5, 57.5)
    ax.set_xlabel("Round", color=INK2)
    ax.set_ylabel("FID", color=INK2)
    title(ax, "(a) CIFAR-10")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)

    ax = axes[1]
    base(ax)
    ax.axhline(7.28, color=INK2, linewidth=0.8, linestyle=(0, (4, 2)))
    line(ax, [25, 50, 75], [12.07, 8.88, 7.28], INK2, "FedAvg")
    line(ax, [25, 50, 75], [12.07, 8.09, 7.02], ORANGE, "Adaptive RegMean")
    ax.set_xticks([25, 50, 75])
    ax.set_xlim(21, 79)
    ax.set_ylim(6.5, 12.8)
    ax.set_xlabel("Round", color=INK2)
    title(ax, "(b) MNIST")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)

    ax = axes[2]
    base(ax)
    line(ax, [25, 50, 75, 100], [1.136, 1.290, 1.451, 1.613], BLUE, "Uniform, from scratch", ls=(0, (3, 1.5)))
    line(ax, [25, 50, 75, 100], [1.161, 1.363, 1.575, 1.786], ORANGE, "Adaptive, from scratch", ls=(0, (3, 1.5)))
    line(ax, [25, 50, 100], [1.0, 1.029, 1.031], BLUE, "Uniform, warm + memory", m="s")
    line(ax, [25, 50, 100], [1.0, 1.036, 1.037], ORANGE, "Adaptive, warm + memory", m="s")
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlim(20, 105)
    ax.set_ylim(0.95, 1.85)
    ax.set_xlabel("Round", color=INK2)
    ax.set_ylabel("Weight norm / FedAvg", color=INK2)
    title(ax, "(c) CIFAR-10 weight growth")
    ax.legend(handles=[Line2D([], [], color=INK2, linewidth=1.4, linestyle=(0, (3, 1.5)), marker="o", markersize=3.6, label="from scratch"),
                       Line2D([], [], color=INK2, linewidth=1.4, marker="s", markersize=3.6, label="warm start + memory")],
              frameon=False, loc="upper left", handlelength=2.2)
    ax.text(102, 1.786, "adaptive", color=ORANGE, fontsize=FS - 1, va="center", ha="right", position=(100, 1.83))
    ax.text(100, 1.47, "uniform", color=BLUE, fontsize=FS - 1, va="center", ha="right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "r_main")


def drift_settings():
    sets = [
        ("(a) CIFAR-10, DDPM", ("round 25", [1.35e-3, 2.64e-3, 3.83e-3, 1.42e-3, 1.04e-4]), ("round 100", [6.33e-4, 2.22e-3, 4.19e-3, 2.41e-3, 2.77e-4])),
        ("(b) MNIST, DDPM", ("round 25", [1.673e-3, 3.365e-3, 1.640e-2, 7.945e-3, 4.322e-4]), ("round 75", [7.130e-4, 1.054e-3, 1.167e-2, 7.826e-3, 4.562e-4])),
        ("(c) CIFAR-10, flow matching", ("round 25", [4.71e-3, 4.46e-3, 9.70e-3, 4.34e-2, 3.52e-1]), ("round 100", [1.50e-3, 2.02e-3, 4.93e-3, 2.30e-2, 1.78e-1])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.9), sharey=True)
    x = np.arange(5)
    for ax, (t, (n1, a), (n2, b)) in zip(axes, sets):
        base(ax)
        a, b = np.array(a) / sum(a) * 100, np.array(b) / sum(b) * 100
        ax.bar(x - 0.19, a, width=0.36, color=BLUES[1], edgecolor=SURFACE, linewidth=0.8, label=n1)
        ax.bar(x + 0.19, b, width=0.36, color=BLUES[3], edgecolor=SURFACE, linewidth=0.8, label=n2)
        k = int(np.argmax(b))
        ax.text(k + 0.19, b[k] + 1.5, f"{b[k]:.0f}%", ha="center", va="bottom", color=INK, fontsize=FS - 1)
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Timestep band", color=INK2)
        title(ax, t)
        ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    axes[0].set_ylabel("Share of total drift (%)", color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "r_drift_settings")


def samples():
    import matplotlib.image as mpimg
    root = os.path.join(HERE, "samples")
    pairs = [("CIFAR-10, FedAvg (round 100)", "e25_fedavg_s0_round100_ema_samples.png"),
             ("CIFAR-10, adaptive RegMean (round 100)", "e29_wgen_s0_round100_ema_samples.png"),
             ("MNIST, FedAvg (round 75)", "mn_fedavg_s0_round75_ema_samples.png"),
             ("MNIST, adaptive RegMean (round 75)", "mn_wgen_s0_round75_ema_samples.png")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 7.2))
    for ax, (t, f) in zip(axes.flat, pairs):
        ax.imshow(mpimg.imread(os.path.join(root, f)))
        ax.axis("off")
        title(ax, t)
    fig.tight_layout()
    save(fig, "r_samples")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figs = [diagnosis, diagnosis_extra, families, one_round, main_result, drift_settings]
    if os.path.isdir(os.path.join(HERE, "samples")):
        figs.append(samples)
    for f in figs:
        f()
    print("saved to", os.path.abspath(OUT))
