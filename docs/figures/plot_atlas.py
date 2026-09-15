"""Draw the connectome in the animal's coordinates, with the deciding cells lit.

    python docs/figures/plot_atlas.py [--data-root ~/connectome_data_prep]

Writes ``atlas.png``. The activation is cached in ``atlas/lateralised.csv`` so
re-styling the figure does not re-run the rate model.

**Read the caveat on the figure, not just the picture.** These are cell bodies,
one dot each. The prepared files carry no skeletons and the services that serve
them are unreachable from here, so this is not the rainbow morphology render it
superficially resembles: somata sit on the rind around the neuropil, so the
brain appears as a shell rather than a solid mass of neurites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "atlas"
BEARING = 35.0

BG = "#07070b"
FG = "#c8c8d4"
#: Cyan for cells that prefer the target on the left, magenta for the right --
#: a diverging pair that stays legible on black, unlike red/blue.
SIDES = LinearSegmentedColormap.from_list(
    "sides", ["#2ad4ff", "#1b6a80", BG, "#7a2a6a", "#ff43c8"]
)

#: The stages the visual decision passes through, outermost first.
STAGES = [
    ("optic lobe", ("ol_intrinsic", "ol_sensory")),
    ("visual projection", ("visual_projection", "visual_centrifugal")),
    ("central brain", ("cb_intrinsic", "cb_sensory")),
    ("descending", ("descending_neuron",)),
    ("nerve cord", ("vnc_intrinsic", "vnc_motor")),
]


def load(data_root: str) -> tuple:
    from flyloop.connectome.data_prep import load_dataset
    from flyloop.viz import Atlas

    c = load_dataset(data_root, "malecns", matrix="inprop")
    atlas = Atlas.from_connectome(c)
    cache = CACHE / "lateralised.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        from flyloop.viz import lateralised_activation

        df = lateralised_activation(c, bearing_deg=BEARING)
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    print(atlas.report())
    return c, atlas, df


def main(data_root: str) -> int:
    c, atlas, df = load(data_root)
    lat = atlas.values(df["lateralised"].to_numpy())
    both = atlas.values(df["both"].to_numpy())

    fig = plt.figure(figsize=(15.5, 10.0), facecolor=BG)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.60], hspace=0.20, wspace=0.05,
        left=0.055, right=0.985, top=0.935, bottom=0.095,
    )
    _context(fig.add_subplot(gs[0, 0]), atlas)
    _highlight(fig.add_subplot(gs[0, 1]), atlas, lat, both)
    _funnel(fig.add_subplot(gs[1, :]), c, df)

    fig.suptitle(
        "MaleCNS v1.0 -- 138,496 located cell bodies, and the ones"
        " that decide which way to turn",
        color="#f0f0f5", fontsize=15, y=0.975,
    )
    fig.text(
        0.012, 0.012,
        "Cell bodies, one dot per neuron (somaLocation). These files contain no"
        " skeletons, so this is not a morphology render:\nsomata sit on the rind"
        " around the neuropil. Highlight = activation(target 35 deg right) -"
        " activation(35 deg left), the\nper-neuron form of Run 10's mirror pair:"
        " cells that respond to both sides equally report that an object exists,"
        " not which way to go.",
        color="#7a7a8a", fontsize=8.5, va="bottom",
    )
    out = HERE / "atlas.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out}")

    # The whole CNS is about twice as long as it is tall, so it only looks
    # right with a panel of its own; squeezed into a row above it shrank to a
    # thumbnail with dead space either side.
    fig2 = plt.figure(figsize=(15.5, 8.6), facecolor=BG)
    ax = fig2.add_axes((0.01, 0.06, 0.98, 0.86))
    _sagittal(ax, atlas, lat, both)
    fig2.text(
        0.012, 0.012,
        "Same highlight, seen from the side. The signal is loud in the optic"
        " lobes and all but gone by the nerve cord:\nof 15,173 located cells"
        " behind the neck, 2,298 respond at all and their mean lateralisation is"
        " 0.021.",
        color="#7a7a8a", fontsize=9, va="bottom",
    )
    out2 = HERE / "atlas_body.png"
    fig2.savefig(out2, dpi=150, facecolor=BG)
    print(f"wrote {out2}")
    return 0


# ------------------------------------------------------------------- panels


def _style(ax, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color="#e8e8f0", fontsize=11.5, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _frame(ax, x, y, pad: float = 0.03) -> None:
    """Crop to the data, so the animal fills the panel instead of the margins."""
    for lo, hi, setter in (
        (x.min(), x.max(), ax.set_xlim),
        (y.min(), y.max(), ax.set_ylim),
    ):
        m = (hi - lo) * pad
        setter(lo - m, hi + m)


def _frontal(atlas):
    """Frontal view of the brain: fly-right on the viewer's right, dorsal up."""
    m = atlas.brain
    # x increases toward the fly's left, so negate it to face the animal.
    return -atlas.xyz[m, 0], -atlas.xyz[m, 1], m


def _context(ax, atlas) -> None:
    x, y, m = _frontal(atlas)
    ax.scatter(x, y, s=0.6, c=np.asarray(atlas.colours())[m], alpha=0.55, lw=0)
    _style(ax, f"the atlas: {m.sum():,} cell bodies, coloured by class")
    _frame(ax, x, y)
    from flyloop.viz.atlas import PALETTE

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=5, color=v, label=k.replace("_", " "))
            for k, v in PALETTE.items()
            if k in set(atlas.super_class)
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False,
        fontsize=8, labelcolor="#a8a8b8", ncol=4, handletextpad=0.3,
        columnspacing=1.2,
    )


def _highlight(ax, atlas, lat, both) -> None:
    x, y, m = _frontal(atlas)
    lat_b, both_b = lat[m], both[m]
    quiet = both_b <= 0
    ax.scatter(x[quiet], y[quiet], s=0.5, c="#23232e", alpha=0.65, lw=0)
    live = ~quiet
    order = np.argsort(np.abs(lat_b[live]))  # strongest drawn last, on top
    xs, ys, vs = x[live][order], y[live][order], lat_b[live][order]
    lim = float(np.percentile(np.abs(vs), 99.5)) or 1.0
    ax.scatter(
        xs, ys, s=0.6 + 11.0 * (np.abs(vs) / lim) ** 2, c=vs,
        cmap=SIDES, norm=TwoSlopeNorm(0.0, -lim, lim), alpha=0.9, lw=0,
    )
    _style(ax, f"lit up: {live.sum():,} cells respond, coloured by the side they prefer")
    _frame(ax, x, y)
    ax.text(0.03, 0.93, "prefers target\non the LEFT", color="#2ad4ff",
            transform=ax.transAxes, fontsize=10, linespacing=1.4)
    ax.text(0.97, 0.93, "prefers target\non the RIGHT", color="#ff43c8",
            transform=ax.transAxes, fontsize=10, ha="right", linespacing=1.4)
    ax.text(0.5, 0.015, "the fly's own left and right -- you are facing it",
            color="#6a6a7a", transform=ax.transAxes, fontsize=8.5, ha="center")


def _sagittal(ax, atlas, lat, both) -> None:
    """Side view of the whole CNS: the decision leaving the brain for the legs."""
    z, y = atlas.xyz[:, 2], -atlas.xyz[:, 1]
    quiet = both <= 0
    # Bright enough that the nerve cord reads as present-but-quiet rather than
    # as empty canvas: "the signal does not get here" is the point of the panel.
    ax.scatter(z[quiet], y[quiet], s=0.45, c="#3a3a4a", alpha=0.65, lw=0)
    live = ~quiet
    order = np.argsort(np.abs(lat[live]))
    lim = float(np.percentile(np.abs(lat[live]), 99.5)) or 1.0
    ax.scatter(
        z[live][order], y[live][order],
        s=0.4 + 9.0 * (np.abs(lat[live][order]) / lim) ** 2,
        c=lat[live][order], cmap=SIDES, norm=TwoSlopeNorm(0.0, -lim, lim),
        alpha=0.9, lw=0,
    )
    from flyloop.viz.atlas import BRAIN_Z_MAX

    ax.axvline(BRAIN_Z_MAX, color="#44445a", ls=":", lw=1.0)
    _style(ax, "side view of the whole animal -- the decision has to reach the legs")
    _frame(ax, z, y)
    lo, hi = ax.get_ylim()
    ax.text(BRAIN_Z_MAX - 2500, lo + 0.08 * (hi - lo), "brain",
            color="#8a8a9a", fontsize=10, ha="right")
    ax.text(BRAIN_Z_MAX + 2500, lo + 0.08 * (hi - lo), "nerve cord",
            color="#8a8a9a", fontsize=10, ha="left")


def _funnel(ax, c, df) -> None:
    """How much side information survives each stage."""
    ax.set_facecolor(BG)
    sc = c.neurons["super_class"].to_numpy()
    lat = df["lateralised"].to_numpy()
    resp = df["both"].to_numpy() > 0
    names, means, counts = [], [], []
    for label, classes in STAGES:
        m = np.isin(sc, classes) & resp
        if not m.any():
            continue
        names.append(label)
        means.append(float(np.abs(lat[m]).mean()))
        counts.append(int(m.sum()))

    x = np.arange(len(names))
    ax.bar(x, means, color="#ff43c8", alpha=0.85, width=0.55)
    for xi, v, n in zip(x, means, counts, strict=True):
        ax.text(xi, v + max(means) * 0.035, f"{v:.3f}", ha="center",
                color="#f0f0f5", fontsize=11)
        ax.text(xi, v + max(means) * 0.135, f"{n:,} cells", ha="center",
                color="#8a8a9a", fontsize=8.5)
    ax.plot(x, means, color="#5a5a70", lw=1.0, ls=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#e8e8f0", fontsize=10.5)
    ax.set_ylabel("mean |lateralisation|", color="#a8a8b8", fontsize=9.5)
    ax.set_title(
        "the decision narrows as it crosses the animal -- 22,792 optic lobe cells"
        " carry side information, 631 descending neurons deliver it",
        color="#e8e8f0", fontsize=11.5, pad=8,
    )
    ax.tick_params(colors="#7a7a8a", labelsize=8.5)
    for side, sp in ax.spines.items():
        sp.set_visible(side in ("left", "bottom"))
        sp.set_color("#33333f")
    ax.set_ylim(0, max(means) * 1.28)
    ax.grid(axis="y", alpha=0.12, color="#8888aa")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/user/yijieyin/connectome_data_prep")
    raise SystemExit(main(ap.parse_args().data_root))
