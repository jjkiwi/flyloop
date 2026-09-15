"""Draw Run 10: NeuroMechFly trajectories under the connectome readout.

Reads ``embodied/trajectories.csv`` and ``embodied/summary.csv`` as written by
``flyloop embodied --out`` (or the Run 10 script) and writes ``embodied.png``.

There is no rendered video: this machine has neither EGL nor OSMesa, so MuJoCo
runs with ``MUJOCO_GL=disable`` -- full physics, no pixels. The trajectory is the
picture.

**The bearing panel folds the mirror pairs.** Episodes with the target on the
left are reflected onto the right before averaging, which is legitimate only
because the measurement is a mirror pair by construction: the two episodes share
a body seed and differ solely in the sign of the stimulus. Folding without that
symmetry would manufacture a result.
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

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))

    for ax, graph in zip(axes[0], graphs, strict=False):
        _trajectories(ax, traj[traj["graph"] == graph], graph)
    _bearing(axes[1][0], traj, graphs)
    _fixation(axes[1][1], summary, graphs)

    fig.suptitle(
        "Run 10 -- the connectome readout walking a physical fly\n"
        "NeuroMechFly v2 in MuJoCo, steered by the left-right difference of DNa02",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "embodied.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


def _trajectories(ax, sub: pd.DataFrame, graph: str) -> None:
    for _, g in sub.groupby(["seed", "side"]):
        ax.plot(g["x"], g["y"], color=COLOUR[graph], alpha=0.8, lw=1.6)
        ax.plot(g["x"].iloc[-1], g["y"].iloc[-1], "o", ms=4.5, color=COLOUR[graph])
    for sign in (-1, 1):
        tx, ty = _target_of(sub, sign)
        ax.add_patch(plt.Circle((tx, ty), TARGET_RADIUS, color="#444", alpha=0.22))
        ax.plot([0, tx], [0, ty], ":", color="#999", lw=0.9)
    ax.plot(0, 0, "k^", ms=8)
    ax.set_title(
        f"{LABEL[graph]} -- {sub['t'].max():.1f} s of walking, 3 body seeds x 2 sides"
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def _bearing(ax, traj: pd.DataFrame, graphs: list[str]) -> None:
    """Angle to the target over time, mirror pairs folded onto the right."""
    for graph in graphs:
        sub = traj[traj["graph"] == graph].copy()
        sub["folded"] = sub["bearing"] * np.sign(sub["bearing0"])
        m = sub.groupby("t")["folded"]
        ax.plot(m.mean().index, m.mean(), color=COLOUR[graph], lw=2.2, label=LABEL[graph])
        ax.fill_between(
            m.mean().index, m.mean() - m.std(), m.mean() + m.std(),
            color=COLOUR[graph], alpha=0.18, lw=0,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(35.0, color="#999", ls=":", lw=1.0)
    ax.text(0.02, 35.6, "target at 35 deg off-axis", fontsize=8, color="#666")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle to target (deg)   0 = facing it")
    ax.set_title("Aiming: mirror pairs folded, mean +/- sd over 6 episodes")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)


def _fixation(ax, summary: pd.DataFrame, graphs: list[str]) -> None:
    seeds = sorted(summary["seed"].unique())
    width = 0.35
    for i, graph in enumerate(graphs):
        vals = summary[summary["graph"] == graph].sort_values("seed")["fixation_deg"]
        x = np.arange(len(seeds)) + (i - 0.5 * (len(graphs) - 1)) * width
        ax.bar(x, vals, width, color=COLOUR[graph], label=LABEL[graph])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("fixation (deg)   positive = turned toward it")
    ax.set_title(
        "Mirror-pair statistic: (left turn - right turn) / 2\n"
        "gait drift cancels, only the stimulus differs"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)


def _target_of(sub: pd.DataFrame, sign: int) -> tuple[float, float]:
    """Recover a target position from the first logged bearing and distance."""
    row = sub[np.sign(sub["bearing0"]) == sign].iloc[0]
    a = np.radians(-row["bearing0"])
    return float(row["distance"] * np.cos(a)), float(row["distance"] * np.sin(a))


if __name__ == "__main__":
    sys.exit(main(*(Path(a) for a in sys.argv[1:])))
