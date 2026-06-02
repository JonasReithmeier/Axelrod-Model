"""
mobility_plot.py
----------------
Loads the mobility trajectory DB produced by mobility_compute.py and
generates a structured set of plots.

Plot types
----------
1. Individual panels  — one figure per (h, F, L, T), lines coloured by q/N.
   Saved as:  plots/schelling/mobility/individual/mob_h{h}_F{F}_L{L}_T{T}.png

2. L-comparison panels — one figure per (h, F, T), one subplot per q/N,
   lines coloured by L.  Shows finite-size effects directly.
   Saved as:  plots/schelling/mobility/L_compare/mob_h{h}_F{F}_T{T}.png

3. T-comparison panels — one figure per (h, F, L), one subplot per q/N,
   lines coloured by T.  Shows how intolerance changes dynamics.
   Saved as:  plots/schelling/mobility/T_compare/mob_h{h}_F{F}_L{L}.png

Usage:
    python mobility_plot.py                         # all plot types
    python mobility_plot.py --type individual
    python mobility_plot.py --type L_compare
    python mobility_plot.py --type T_compare
    python mobility_plot.py --config path/to/config.yaml
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------

def load_db(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    df = pd.read_parquet(db_path)
    # Deserialise the stored numpy arrays
    df["mcs_axis"] = df["mcs_axis"].apply(pickle.loads)
    df["m_values"] = df["m_values"].apply(pickle.loads)
    return df


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Qualitative palette — up to 10 distinct colours
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]
_MARKERS = ["o", "s", "D", "^", "v", ">", "<", "p", "h", "8"]


def _apply_style():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.labelsize":  13,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top":       True,
        "ytick.right":     True,
        "axes.grid":       True,
        "grid.alpha":      0.25,
        "grid.linestyle":  "--",
        "figure.dpi":      100,
    })


def _color_map(values, palette=_PALETTE):
    """Map a sorted list of values → colour dict."""
    uniq = sorted(set(values))
    return {v: palette[i % len(palette)] for i, v in enumerate(uniq)}


def _marker_map(values):
    uniq = sorted(set(values))
    return {v: _MARKERS[i % len(_MARKERS)] for i, v in enumerate(uniq)}


def _markevery(n_pts: int, target: int = 20) -> int:
    return max(1, n_pts // target)


def _sci_x(ax):
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot type 1: individual panels
# ---------------------------------------------------------------------------

def plot_individual(df: pd.DataFrame, plot_dir: Path):
    """One figure per (h, F, L, T). Lines = q/N values."""
    out = plot_dir / "individual"
    cmap = _color_map(df["q_N"])
    mmap = _marker_map(df["q_N"])

    for (h, F, L, T), grp in df.groupby(["h", "F", "L", "T"]):
        fig, ax = plt.subplots(figsize=(7, 5))

        for q_N, sub in grp.groupby("q_N"):
            # Average over realizations if M > 1
            mcs_ref = sub.iloc[0]["mcs_axis"]
            m_mean  = np.mean(np.vstack(sub["m_values"].values), axis=0)

            ax.plot(
                mcs_ref, m_mean,
                color=cmap[q_N],
                marker=mmap[q_N],
                markevery=_markevery(len(mcs_ref)),
                markersize=4,
                linewidth=1.6,
                label=f"q/N = {q_N:.3g}",
            )

        ax.set_xlabel("t  (MCS)")
        ax.set_ylabel("m(t)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"h = {h},  F = {F},  L = {L},  T = {T}")
        _sci_x(ax)
        ax.legend(loc="upper right", framealpha=1.0, edgecolor="black",
                  ncol=1 + len(grp["q_N"].unique()) // 8)

        _save(fig, out / f"mob_h{h}_F{F}_L{L}_T{T}.png")


# ---------------------------------------------------------------------------
# Plot type 2: L-comparison — finite-size effects
# ---------------------------------------------------------------------------

def plot_L_compare(df: pd.DataFrame, plot_dir: Path):
    """
    One figure per (h, F, T).
    Subplots arranged in a grid, one per q/N value.
    Within each subplot, lines are coloured by L.
    """
    out  = plot_dir / "L_compare"

    for (h, F, T), grp in df.groupby(["h", "F", "T"]):
        q_N_vals = sorted(grp["q_N"].unique())
        L_vals   = sorted(grp["L"].unique())
        cmap     = _color_map(L_vals)

        n     = len(q_N_vals)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            sharey=True, squeeze=False,
        )

        for idx, q_N in enumerate(q_N_vals):
            ax  = axes[idx // ncols][idx % ncols]
            sub = grp[grp["q_N"] == q_N]

            for L in L_vals:
                rows = sub[sub["L"] == L]
                if rows.empty:
                    continue
                mcs_ref = rows.iloc[0]["mcs_axis"]
                m_mean  = np.mean(np.vstack(rows["m_values"].values), axis=0)

                ax.plot(
                    mcs_ref, m_mean,
                    color=cmap[L],
                    linewidth=1.5,
                    label=f"L = {L}",
                )

            ax.set_title(f"q/N = {q_N:.3g}", fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            _sci_x(ax)
            if idx % ncols == 0:
                ax.set_ylabel("m(t)")
            if idx // ncols == nrows - 1:
                ax.set_xlabel("t  (MCS)")

        # Hide unused subplots
        for idx in range(len(q_N_vals), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # Shared legend
        handles = [
            plt.Line2D([0], [0], color=cmap[L], linewidth=2, label=f"L = {L}")
            for L in L_vals
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(L_vals), framealpha=1.0,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"h = {h},  F = {F},  T = {T}  — finite-size comparison",
                     fontsize=13, y=1.01)
        fig.tight_layout()

        _save(fig, out / f"mob_h{h}_F{F}_T{T}.png")


# ---------------------------------------------------------------------------
# Plot type 3: T-comparison — effect of intolerance
# ---------------------------------------------------------------------------

def plot_T_compare(df: pd.DataFrame, plot_dir: Path):
    """
    One figure per (h, F, L).
    Subplots per q/N, lines coloured by T.
    """
    out = plot_dir / "T_compare"

    for (h, F, L), grp in df.groupby(["h", "F", "L"]):
        q_N_vals = sorted(grp["q_N"].unique())
        T_vals   = sorted(grp["T"].unique())
        cmap     = _color_map(T_vals)

        n     = len(q_N_vals)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            sharey=True, squeeze=False,
        )

        for idx, q_N in enumerate(q_N_vals):
            ax  = axes[idx // ncols][idx % ncols]
            sub = grp[grp["q_N"] == q_N]

            for T in T_vals:
                rows = sub[sub["T"] == T]
                if rows.empty:
                    continue
                mcs_ref = rows.iloc[0]["mcs_axis"]
                m_mean  = np.mean(np.vstack(rows["m_values"].values), axis=0)

                ax.plot(
                    mcs_ref, m_mean,
                    color=cmap[T],
                    linewidth=1.5,
                    label=f"T = {T}",
                )

            ax.set_title(f"q/N = {q_N:.3g}", fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            _sci_x(ax)
            if idx % ncols == 0:
                ax.set_ylabel("m(t)")
            if idx // ncols == nrows - 1:
                ax.set_xlabel("t  (MCS)")

        for idx in range(len(q_N_vals), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=cmap[T], linewidth=2, label=f"T = {T}")
            for T in T_vals
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(T_vals), framealpha=1.0,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"h = {h},  F = {F},  L = {L}  — intolerance comparison",
                     fontsize=13, y=1.01)
        fig.tight_layout()

        _save(fig, out / f"mob_h{h}_F{F}_L{L}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--type",
        choices=["individual", "L_compare", "T_compare", "all"],
        default="all",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["as_mobility_experiment"]

    db_path  = Path(cfg.get("db_path", "data/schelling/mobility_trajectories.parquet"))
    plot_dir = Path(cfg.get("plot_dir", "plots/schelling/mobility"))

    print(f"Loading DB from {db_path} ...")
    df = load_db(db_path)
    print(f"  {len(df)} trajectories loaded.")

    _apply_style()

    run_all = args.type == "all"

    if run_all or args.type == "individual":
        print("\n[1/3] Individual panels ...")
        plot_individual(df, plot_dir)

    if run_all or args.type == "L_compare":
        print("\n[2/3] L-comparison panels ...")
        plot_L_compare(df, plot_dir)

    if run_all or args.type == "T_compare":
        print("\n[3/3] T-comparison panels ...")
        plot_T_compare(df, plot_dir)

    print("\nAll plots done.")


if __name__ == "__main__":
    main()