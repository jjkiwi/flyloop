"""Draw Run 10: NeuroMechFly trajectories under the connectome readout.

Reads ``embodied/trajectories.csv`` and ``embodied/summary.csv`` as written by
``flyloop embodied --out`` (or the Run 10 script) and writes ``embodied.png``.

There is no rendered video: this machine has neither EGL nor OSMesa, so MuJoCo
runs with ``MUJOCO_GL=disable`` -- full physics, no pixels. The trajectory is the
picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET_RADIUS = 2.5

COLOUR = {"malecns": "#1f7a5c", "rewired": "#b0553a"}
LABEL = {"malecns": "MaleCNS", "rewired": "rewired control"}


def main(src: Path = HERE / "embodied") -> int:
    traj = pd.read_csv(src / "trajectories.csv")
    summary = pd.read_csv(src / "summary.csv")

    graphs = [g for g in ("malecns", "rewired") if g in set(traj["graph"])]
    fig, axes = plt.subplots(1, len(graphs) + 1, figsize=(5.0 * (len(graphs) + 1), 4.6))

    for ax, graph in zip(axes, graphs, strict=False):
        sub = traj[traj["graph"] == graph]
        for (seed, side), g in sub.groupby(["seed", "side"]):
            ax.plot(
                g["x"], g["y"], color=COLOUR[graph], alpha=0.85, lw=1.6,
                label=f"seed {seed}" if side == "right" else None,
            )
            ax.plot(g["x"].iloc[-1], g["y"].iloc[-1], "o", ms=4, color=COLOUR[graph])
        # Both mirrored targets, same for every seed.
        for sign in (-1, 1):
            tx, ty = _target_of(sub, sign)
            ax.add_patch(plt.Circle((tx, ty), TARGET_RADIUS, color="#444", alpha=0.25))
            ax.plot([0, tx], [0, ty], ":", color="#888", lw=0.9)
        ax.plot(0, 0, "k^", ms=7)
        ax.set_title(f"{LABEL[graph]}\ntrajectories, {sub['t'].max():.1f} s of walking")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    ax = axes[-1]
    width = 0.35
    for i, graph in enumerate(graphs):
        vals = summary[summary["graph"] == graph]["fixation_deg"].to_numpy()
        x = np.arange(len(vals)) + (i - 0.5 * (len(graphs) - 1)) * width
        ax.bar(x, vals, width, color=COLOUR[graph], label=LABEL[graph])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(summary["seed"].nunique()))
    ax.set_xticklabels([f"seed {s}" for s in sorted(summary["seed"].unique())])
    ax.set_ylabel("fixation (deg)   positive = turned toward the target")
    ax.set_title("mirror-pair statistic\n(left turn minus right turn, halved)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = HERE / "embodied.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


def _target_of(sub: pd.DataFrame, sign: int) -> tuple[float, float]:
    """Recover a target position from the logged bearing and distance."""
    row = sub[np.sign(sub["bearing0"]) == sign].iloc[0]
    a = np.radians(-row["bearing0"])
    return float(row["distance"] * np.cos(a)), float(row["distance"] * np.sin(a))


if __name__ == "__main__":
    sys.exit(main(*(Path(a) for a in sys.argv[1:])))
