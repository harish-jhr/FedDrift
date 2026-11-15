# Appendix figure for the sketched-Gram variant, from its round-50 measurements.
import matplotlib.pyplot as plt
import numpy as np

from report_figures import BANDS, BLUE, COL, GRID, INK, INK2, MUTED, ORANGE, SURFACE, WIDE, FS, base, line, save, title

GREEN = "#1baf7a"


def sketch():
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.1), gridspec_kw={"width_ratios": [1.05, 0.95, 1.05]})

    ax = axes[0]
    base(ax)
    red = [11.9, 25.1, 18.1, 5.9, 7.1]
    x = np.arange(5)
    ax.bar(x, red, width=0.6, color=[GREEN if i < 2 else MUTED for i in range(5)], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(red):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=FS - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 30)
    ax.set_xlabel("Timestep band", color=INK2)
    ax.set_ylabel("Drift reduction vs FedAvg (%)", color=INK2)
    title(ax, "(a) Drift at round 50")

    ax = axes[1]
    base(ax)
    line(ax, [25, 50], [128.93, 156.21], ORANGE, ls=(0, (3, 1.5)))
    line(ax, [25, 50], [111.1, 118.66], ORANGE, m="s")
    line(ax, [25, 50], [111.1, 114.6], INK2)
    line(ax, [25, 50], [111.1, 114.98], GREEN, m="^")
    for y, t, c in [(156.21, "RegMean, scratch", ORANGE), (120.2, "RegMean, warm + mem.", ORANGE),
                    (115.6, "Sketched, $t<400$", GREEN), (111.4, "FedAvg", INK2)]:
        ax.text(51.5, y, t, color=c, fontsize=FS - 1, va="center")
    ax.set_xticks([25, 50])
    ax.set_xlim(21, 72)
    ax.set_ylim(105, 162)
    ax.set_xlabel("Round", color=INK2)
    ax.set_ylabel("Weight norm", color=INK2)
    title(ax, "(b) Weight norm")

    ax = axes[2]
    base(ax)
    names = ["Full\nGram", "Channel\nGram", "Sketch\n5 bands", "Sketch\n$t<400$"]
    vals = [1820.33, 43.5, 3.63, 1.45]
    ax.bar(range(4), vals, width=0.6, color=[MUTED, ORANGE, BLUE, GREEN], edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:,.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:,.0f}"), ha="center", va="bottom", color=INK, fontsize=FS - 1)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 8000)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("Upload per client per round (MB)", color=INK2)
    title(ax, "(c) Statistics upload")

    fig.tight_layout(w_pad=1.0)
    save(fig, "r_sketch")


if __name__ == "__main__":
    import os
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"), exist_ok=True)
    sketch()
