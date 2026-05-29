"""
visualize.py
------------
Comprehensive visualization for the Development Small World sweep results.

Usage:
    python visualize.py --csv results.csv --outdir figures/

    # Only specific plot groups:
    python visualize.py --csv results.csv --plots order_parameter phase_diagram smallworld

    # Override axis variable:
    python visualize.py --csv results.csv --x-axis q_over_N

    # Filter to specific parameter subset:
    python visualize.py --csv results.csv --filter "weight_mode==2" --filter "dev_mode==2"

All figures are saved as high-res PNGs and optionally PDFs.
"""

import argparse
import os
import warnings
from itertools import product
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Georgia", "DejaVu Serif", "Times New Roman"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
WEIGHT_LABELS  = {0: "Linear", 1: "Quadratic", 2: "Biphasic", 3: "Attraction"}
DEV_LABELS     = {0: "Uniform", 1: "Normal", 2: "Pareto", 3: "Bimodal"}
WEIGHT_COLORS  = {0: "#2166ac", 1: "#d6604d", 2: "#1a9641", 3: "#762a83"}
DEV_COLORS     = {0: "#4dac26", 1: "#f1b6da", 2: "#e08214", 3: "#542788"}
P_CMAP         = plt.cm.plasma
SMAX_CMAP      = LinearSegmentedColormap.from_list(
    "smax", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#f5a623"], N=256
)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(csv_path: str, filters: Optional[list] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Derived columns
    df["q_over_N"]   = df["q"] / df["N"]
    df["q_over_F"]   = df["q"] / df["F"]
    df["converged"]  = df["converged"].astype(bool)
    df["final_rewire_rate"] = df["final_rewire_rate"].replace(-1.0, np.nan)

    # Apply user filters
    if filters:
        for f in filters:
            df = df.query(f)

    print(f"Loaded {len(df)} rows from {csv_path}")
    if df.empty:
        raise ValueError("DataFrame is empty after filtering — check your --filter expressions.")
    return df


def _agg(df, groupby, metrics=("s_max", "num_clusters", "mean_dissatisfaction",
                               "L", "C", "degree_variance", "final_rewire_rate",
                               "converged", "total_steps")):
    """Group and aggregate with mean ± std."""
    agg_dict = {}
    for m in metrics:
        if m in df.columns:
            agg_dict[m]         = (m, "mean")
            agg_dict[m + "_std"]= (m, "std")
            agg_dict[m + "_n"]  = (m, "count")
    return df.groupby(groupby).agg(**agg_dict).reset_index()


def _savefig(fig, outdir, name, pdf=False):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path)
    if pdf:
        fig.savefig(os.path.join(outdir, name + ".pdf"))
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Order parameter: s_max vs q/N (or q/F)
# ---------------------------------------------------------------------------

def plot_order_parameter(df, x_col="q_over_N", hue_col="weight_mode",
                         row_col="dev_mode", fixed_filters=None,
                         outdir="figures", label_maps=None):
    """
    s_max vs x_col, one curve per hue_col value,
    one row per row_col value.
    """
    label_maps = label_maps or {"weight_mode": WEIGHT_LABELS, "dev_mode": DEV_LABELS}
    fdf = df if not fixed_filters else df.query(" and ".join(fixed_filters))

    row_vals = sorted(fdf[row_col].unique())
    hue_vals = sorted(fdf[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})

    n_rows = len(row_vals)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, rv in zip(axes, row_vals):
        sub = fdf[fdf[row_col] == rv]
        grp = _agg(sub, groupby=[x_col, hue_col])

        for hv in hue_vals:
            g = grp[grp[hue_col] == hv].sort_values(x_col)
            if g.empty:
                continue
            label = label_maps.get(hue_col, {}).get(hv, str(hv))
            color = color_map.get(hv, "gray")
            ax.plot(g[x_col], g["s_max"], "o-", color=color, label=label,
                    linewidth=1.8, markersize=4)
            if "s_max_std" in g.columns:
                ax.fill_between(g[x_col],
                                g["s_max"] - g["s_max_std"],
                                g["s_max"] + g["s_max_std"],
                                alpha=0.15, color=color)

        rv_label = label_maps.get(row_col, {}).get(rv, f"{row_col}={rv}")
        ax.set_ylabel("$s_{max}$", fontsize=12)
        ax.set_title(f"{row_col.replace('_',' ').title()}: {rv_label}", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(title=hue_col.replace("_", " ").title(), fontsize=8, ncol=2)

    axes[-1].set_xlabel(x_col.replace("_", "/"), fontsize=12)
    fig.suptitle("Order Parameter  $s_{max}$", fontsize=14, y=1.01)
    fig.tight_layout()
    _savefig(fig, outdir, f"order_parameter_{x_col}_by_{hue_col}_rows_{row_col}")


# ---------------------------------------------------------------------------
# 2. Number of clusters vs x_col
# ---------------------------------------------------------------------------

def plot_num_clusters(df, x_col="q_over_N", hue_col="weight_mode",
                      row_col="dev_mode", outdir="figures"):
    row_vals = sorted(df[row_col].unique())
    hue_vals = sorted(df[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})

    n_rows = len(row_vals)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, rv in zip(axes, row_vals):
        sub = df[df[row_col] == rv]
        grp = _agg(sub, groupby=[x_col, hue_col])
        for hv in hue_vals:
            g = grp[grp[hue_col] == hv].sort_values(x_col)
            if g.empty:
                continue
            label = WEIGHT_LABELS.get(hv, str(hv)) if hue_col == "weight_mode" else str(hv)
            ax.plot(g[x_col], g["num_clusters"], "s--",
                    color=color_map.get(hv, "gray"), label=label,
                    linewidth=1.6, markersize=4)
        rv_label = DEV_LABELS.get(rv, str(rv)) if row_col == "dev_mode" else str(rv)
        ax.set_ylabel("# Clusters", fontsize=12)
        ax.set_title(f"{row_col}: {rv_label}", fontsize=11)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel(x_col.replace("_", "/"), fontsize=12)
    fig.suptitle("Number of Cultural Clusters", fontsize=14, y=1.01)
    fig.tight_layout()
    _savefig(fig, outdir, f"num_clusters_{x_col}_by_{hue_col}_rows_{row_col}")


# ---------------------------------------------------------------------------
# 3. Small-world signature: L(p)/L(0) and C(p)/C(0) vs p
# ---------------------------------------------------------------------------

def plot_smallworld_signature(df, hue_col="weight_mode", outdir="figures"):
    """
    Normalized L and C vs rewiring probability p.
    Requires p=0 (or minimum p) as reference.
    """
    hue_vals = sorted(df[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})

    fig, (ax_L, ax_C) = plt.subplots(1, 2, figsize=(12, 4.5))

    for hv in hue_vals:
        sub = df[df[hue_col] == hv]
        grp = _agg(sub, groupby=["p"])
        if grp.empty:
            continue
        p0_row = grp[grp["p"] == grp["p"].min()]
        if p0_row.empty or p0_row["L"].values[0] == 0:
            continue
        L0 = p0_row["L"].values[0]
        C0 = p0_row["C"].values[0] if p0_row["C"].values[0] != 0 else np.nan

        g = grp.sort_values("p")
        label = WEIGHT_LABELS.get(hv, str(hv)) if hue_col == "weight_mode" else str(hv)
        color = color_map.get(hv, "gray")

        ax_L.plot(g["p"], g["L"] / L0, "o-", color=color, label=label,
                  linewidth=1.8, markersize=4)
        ax_C.plot(g["p"], g["C"] / C0, "o-", color=color, label=label,
                  linewidth=1.8, markersize=4)

    for ax, ylabel, title in [
        (ax_L, "$L(p) / L(0)$", "Characteristic Path Length (normalised)"),
        (ax_C, "$C(p) / C(0)$", "Clustering Coefficient (normalised)")
    ]:
        ax.set_xlabel("Rewiring probability  $p$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(title=hue_col.replace("_", " ").title(), fontsize=8)
        ax.set_xscale("log")

    fig.suptitle("Small-World Signature After Dynamics", fontsize=14)
    fig.tight_layout()
    _savefig(fig, outdir, f"smallworld_LC_vs_p_by_{hue_col}")


# ---------------------------------------------------------------------------
# 4. Phase diagram heatmaps: s_max in (dis_threshold × alpha) space
# ---------------------------------------------------------------------------

def plot_phase_diagram(df, x_col="dis_threshold", y_col="alpha",
                       z_col="s_max", panel_col="weight_mode",
                       row_col="dev_mode", outdir="figures",
                       agg_func="mean"):
    panel_vals = sorted(df[panel_col].unique())
    row_vals   = sorted(df[row_col].unique())
    n_panels   = len(panel_vals)
    n_rows_fig = len(row_vals)

    cmap = SMAX_CMAP if z_col == "s_max" else "viridis"
    vmin = 0 if z_col in ("s_max", "converged") else None
    vmax = 1 if z_col in ("s_max", "converged") else None

    fig, axes = plt.subplots(n_rows_fig, n_panels,
                             figsize=(4 * n_panels, 3.5 * n_rows_fig),
                             squeeze=False)

    for ri, rv in enumerate(row_vals):
        for ci, pv in enumerate(panel_vals):
            ax = axes[ri][ci]
            sub = df[(df[panel_col] == pv) & (df[row_col] == rv)]
            if sub.empty:
                ax.set_visible(False)
                continue

            pivot = sub.groupby([y_col, x_col])[z_col].agg(agg_func).unstack()
            sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                        annot=len(pivot) <= 8, fmt=".2f",
                        linewidths=0.3, linecolor="#333",
                        cbar_kws={"shrink": 0.8, "label": z_col})

            panel_label = WEIGHT_LABELS.get(pv, str(pv)) if panel_col == "weight_mode" else str(pv)
            row_label   = DEV_LABELS.get(rv, str(rv))    if row_col == "dev_mode"    else str(rv)
            ax.set_title(f"{panel_label}\n({row_col}: {row_label})", fontsize=10)
            ax.set_xlabel(x_col.replace("_", " "), fontsize=9)
            ax.set_ylabel(y_col.replace("_", " "), fontsize=9)

    fig.suptitle(f"Phase Diagram: {z_col}  [{x_col} × {y_col}]", fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, f"phase_diagram_{z_col}_{x_col}_x_{y_col}_panels_{panel_col}")


# ---------------------------------------------------------------------------
# 5. Convergence rate heatmap
# ---------------------------------------------------------------------------

def plot_convergence_map(df, x_col="dis_threshold", y_col="alpha",
                         panel_col="weight_mode", outdir="figures"):
    """Fraction of converged trials in parameter space."""
    df2 = df.copy()
    df2["converged_int"] = df2["converged"].astype(int)
    plot_phase_diagram(df2, x_col=x_col, y_col=y_col, z_col="converged_int",
                       panel_col=panel_col, row_col="dev_mode",
                       outdir=outdir, agg_func="mean")
    # rename the saved file
    src = os.path.join(outdir, f"phase_diagram_converged_int_{x_col}_x_{y_col}_panels_{panel_col}.png")
    dst = src.replace("converged_int", "convergence_rate")
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"  Renamed → {dst}")


# ---------------------------------------------------------------------------
# 6. s_max vs num_clusters scatter (cluster structure)
# ---------------------------------------------------------------------------

def plot_cluster_scatter(df, hue_col="weight_mode", size_col="mean_dissatisfaction",
                         outdir="figures"):
    fig, axes = plt.subplots(1, len(df["dev_mode"].unique()),
                             figsize=(6 * len(df["dev_mode"].unique()), 5),
                             squeeze=False)

    for ax, (dv, sub) in zip(axes[0], df.groupby("dev_mode")):
        sc = ax.scatter(
            sub["num_clusters"], sub["s_max"],
            c=sub[hue_col],
            s=40 + 200 * sub[size_col].clip(0, 1),
            cmap="tab10",
            alpha=0.55, linewidths=0.3, edgecolors="white"
        )
        ax.set_xlabel("# Clusters", fontsize=12)
        ax.set_ylabel("$s_{max}$", fontsize=12)
        ax.set_title(f"Dev mode: {DEV_LABELS.get(dv, dv)}", fontsize=11)
        cb = fig.colorbar(sc, ax=ax, ticks=sorted(df[hue_col].unique()))
        cb.set_label(hue_col.replace("_", " "), fontsize=9)
        cb.ax.set_yticklabels([WEIGHT_LABELS.get(v, str(v))
                                for v in sorted(df[hue_col].unique())], fontsize=7)

    fig.suptitle("Cluster Structure: $s_{max}$ vs #Clusters\n(point size = mean dissatisfaction)",
                 fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, f"cluster_scatter_by_{hue_col}")


# ---------------------------------------------------------------------------
# 7. Degree variance vs p (topology drift)
# ---------------------------------------------------------------------------

def plot_degree_variance(df, hue_col="weight_mode", outdir="figures"):
    hue_vals  = sorted(df[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})

    fig, axes = plt.subplots(1, len(df["dev_mode"].unique()),
                             figsize=(6 * len(df["dev_mode"].unique()), 4.5),
                             squeeze=False)

    for ax, (dv, sub) in zip(axes[0], df.groupby("dev_mode")):
        grp = _agg(sub, groupby=["p", hue_col])
        for hv in hue_vals:
            g = grp[grp[hue_col] == hv].sort_values("p")
            if g.empty:
                continue
            label = WEIGHT_LABELS.get(hv, str(hv)) if hue_col == "weight_mode" else str(hv)
            ax.plot(g["p"], g["degree_variance"], "o-",
                    color=color_map.get(hv, "gray"), label=label,
                    linewidth=1.8, markersize=4)
        ax.set_xlabel("$p$ (rewiring probability)", fontsize=12)
        ax.set_ylabel("Degree variance", fontsize=12)
        ax.set_title(f"Dev: {DEV_LABELS.get(dv, dv)}", fontsize=11)
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    fig.suptitle("Network Topology Drift: Degree Variance vs $p$", fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, f"degree_variance_vs_p_by_{hue_col}")


# ---------------------------------------------------------------------------
# 8. Mean dissatisfaction vs alpha (equilibrium satisfaction)
# ---------------------------------------------------------------------------

def plot_dissatisfaction_vs_alpha(df, hue_col="weight_mode", outdir="figures"):
    hue_vals  = sorted(df[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})

    row_vals = sorted(df["dev_mode"].unique())
    fig, axes = plt.subplots(1, len(row_vals),
                             figsize=(5.5 * len(row_vals), 4.5), squeeze=False)

    for ax, dv in zip(axes[0], row_vals):
        sub = df[df["dev_mode"] == dv]
        grp = _agg(sub, groupby=["alpha", hue_col])
        for hv in hue_vals:
            g = grp[grp[hue_col] == hv].sort_values("alpha")
            if g.empty:
                continue
            label = WEIGHT_LABELS.get(hv, str(hv)) if hue_col == "weight_mode" else str(hv)
            ax.plot(g["alpha"], g["mean_dissatisfaction"], "o-",
                    color=color_map.get(hv, "gray"), label=label,
                    linewidth=1.8, markersize=4)
            if "mean_dissatisfaction_std" in g.columns:
                ax.fill_between(g["alpha"],
                                g["mean_dissatisfaction"] - g["mean_dissatisfaction_std"],
                                g["mean_dissatisfaction"] + g["mean_dissatisfaction_std"],
                                alpha=0.12, color=color_map.get(hv, "gray"))
        ax.set_xlabel(r"$\alpha$ (weight strength)", fontsize=12)
        ax.set_ylabel("Mean dissatisfaction at convergence", fontsize=11)
        ax.set_title(f"Dev: {DEV_LABELS.get(dv, dv)}", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Equilibrium Social Satisfaction vs Development Weight Strength",
                 fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, f"dissatisfaction_vs_alpha_by_{hue_col}")


# ---------------------------------------------------------------------------
# 9. Convergence time (total_steps) heatmap — critical slowing down
# ---------------------------------------------------------------------------

def plot_convergence_time(df, x_col="dis_threshold", y_col="alpha",
                          panel_col="weight_mode", outdir="figures"):
    df_conv = df[df["converged"]].copy()
    if df_conv.empty:
        print("  No converged trials — skipping convergence time plot.")
        return

    panel_vals = sorted(df_conv[panel_col].unique())
    row_vals   = sorted(df_conv["dev_mode"].unique())

    fig, axes = plt.subplots(len(row_vals), len(panel_vals),
                             figsize=(4.5 * len(panel_vals), 3.5 * len(row_vals)),
                             squeeze=False)

    for ri, rv in enumerate(row_vals):
        for ci, pv in enumerate(panel_vals):
            ax = axes[ri][ci]
            sub = df_conv[(df_conv[panel_col] == pv) & (df_conv["dev_mode"] == rv)]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = sub.groupby([y_col, x_col])["total_steps"].mean().unstack()
            sns.heatmap(pivot, ax=ax, cmap="magma_r",
                        linewidths=0.3, linecolor="#444",
                        cbar_kws={"shrink": 0.8, "label": "Steps to convergence"})
            panel_lbl = WEIGHT_LABELS.get(pv, str(pv)) if panel_col == "weight_mode" else str(pv)
            dev_lbl   = DEV_LABELS.get(rv, str(rv))
            ax.set_title(f"{panel_lbl} | Dev: {dev_lbl}", fontsize=10)
            ax.set_xlabel(x_col.replace("_", " "), fontsize=9)
            ax.set_ylabel(y_col.replace("_", " "), fontsize=9)

    fig.suptitle("Convergence Time (Critical Slowing Down)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, f"convergence_time_{x_col}_x_{y_col}_panels_{panel_col}")


# ---------------------------------------------------------------------------
# 10. Dev distribution effect: uniform vs Pareto overlay
# ---------------------------------------------------------------------------

def plot_dev_distribution_effect(df, x_col="dis_threshold", y_col="s_max",
                                  hue_col="weight_mode", outdir="figures"):
    """
    Side-by-side: same curve, dev_mode=0 (solid) vs dev_mode=2 (dashed).
    Shows whether wealth inequality shifts the phase transition.
    """
    dev_modes = sorted(df["dev_mode"].unique())
    if len(dev_modes) < 2:
        print("  Only one dev_mode present — skipping distribution comparison plot.")
        return

    hue_vals  = sorted(df[hue_col].unique())
    color_map = (WEIGHT_COLORS if hue_col == "weight_mode" else
                 {v: P_CMAP(i / max(len(hue_vals)-1, 1)) for i, v in enumerate(hue_vals)})
    linestyles = {dev_modes[0]: "-", dev_modes[1]: "--"}
    dev_labels  = {dev_modes[0]: DEV_LABELS.get(dev_modes[0], str(dev_modes[0])),
                   dev_modes[1]: DEV_LABELS.get(dev_modes[1], str(dev_modes[1]))}

    fig, ax = plt.subplots(figsize=(9, 5))

    for dv in dev_modes[:2]:
        sub = df[df["dev_mode"] == dv]
        grp = _agg(sub, groupby=[x_col, hue_col])
        for hv in hue_vals:
            g = grp[grp[hue_col] == hv].sort_values(x_col)
            if g.empty:
                continue
            label = f"{WEIGHT_LABELS.get(hv, str(hv))} [{dev_labels[dv]}]"
            ax.plot(g[x_col], g[y_col],
                    linestyle=linestyles[dv],
                    color=color_map.get(hv, "gray"),
                    label=label, linewidth=1.8, markersize=4, marker="o")

    ax.set_xlabel(x_col.replace("_", " / "), fontsize=12)
    ax.set_ylabel("$s_{max}$", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, title="Weight mode [Dev distribution]")
    ax.set_title(f"Development Distribution Effect on Cultural Fragmentation\n"
                 f"(solid = {dev_labels[dev_modes[0]]}, dashed = {dev_labels[dev_modes[1]]})",
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, outdir, f"dev_distribution_effect_{x_col}_vs_{y_col}")


# ---------------------------------------------------------------------------
# 11. Key Figure: 4-panel phase diagram (the main result)
# ---------------------------------------------------------------------------

def plot_key_figure(df, outdir="figures"):
    """
    4-panel heatmap of s_max in (dis_threshold × alpha) space,
    one panel per weight_mode, at p closest to 0.1 and dev_mode=2 (Pareto).
    This is the central publishable figure.
    """
    p_target  = 0.1
    dev_target = 2 if 2 in df["dev_mode"].unique() else df["dev_mode"].min()

    p_val = df["p"].iloc[(df["p"] - p_target).abs().argsort().iloc[0]]
    sub = df[(df["p"] == p_val) & (df["dev_mode"] == dev_target)]

    if sub.empty:
        print(f"  No data for p≈{p_target}, dev_mode={dev_target} — skipping key figure.")
        return

    weight_modes = sorted(sub["weight_mode"].unique())
    n = len(weight_modes)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)

    for ax, wm in zip(axes[0], weight_modes):
        g = sub[sub["weight_mode"] == wm]
        pivot = g.groupby(["alpha", "dis_threshold"])["s_max"].mean().unstack()
        im = ax.imshow(pivot.values, aspect="auto", cmap=SMAX_CMAP,
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
        ax.set_xlabel(r"$\theta$ (dis. threshold)", fontsize=10)
        ax.set_ylabel(r"$\alpha$ (weight strength)", fontsize=10)
        ax.set_title(WEIGHT_LABELS.get(wm, str(wm)), fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="$s_{max}$", shrink=0.85)

    fig.suptitle(
        f"Cultural Fragmentation Phase Diagram\n"
        f"$p={p_val:.2f}$,  Dev: {DEV_LABELS.get(dev_target, str(dev_target))}",
        fontsize=13
    )
    fig.tight_layout()
    _savefig(fig, outdir, "KEY_FIGURE_phase_diagram_4panel")


# ---------------------------------------------------------------------------
# 12. Summary statistics table (printed + saved as CSV)
# ---------------------------------------------------------------------------

def print_summary(df, outdir="figures"):
    print("\n=== Summary by weight_mode ===")
    summary = df.groupby("weight_mode").agg(
        s_max_mean=("s_max", "mean"),
        s_max_std=("s_max", "std"),
        num_clusters_mean=("num_clusters", "mean"),
        convergence_rate=("converged", "mean"),
        mean_dis_mean=("mean_dissatisfaction", "mean"),
        deg_var_mean=("degree_variance", "mean"),
    ).round(4)
    summary.index = [WEIGHT_LABELS.get(i, str(i)) for i in summary.index]
    print(summary.to_string())
    summary.to_csv(os.path.join(outdir, "summary_by_weight_mode.csv"))

    print("\n=== Summary by dev_mode ===")
    summary2 = df.groupby("dev_mode").agg(
        s_max_mean=("s_max", "mean"),
        convergence_rate=("converged", "mean"),
        mean_dis_mean=("mean_dissatisfaction", "mean"),
    ).round(4)
    summary2.index = [DEV_LABELS.get(i, str(i)) for i in summary2.index]
    print(summary2.to_string())
    summary2.to_csv(os.path.join(outdir, "summary_by_dev_mode.csv"))


# ---------------------------------------------------------------------------
# Plot registry — maps names to functions and their default kwargs
# ---------------------------------------------------------------------------

PLOT_REGISTRY = {
    "order_parameter":      (plot_order_parameter,      {}),
    "num_clusters":         (plot_num_clusters,         {}),
    "smallworld":           (plot_smallworld_signature, {}),
    "phase_diagram":        (plot_phase_diagram,        {}),
    "convergence_map":      (plot_convergence_map,      {}),
    "cluster_scatter":      (plot_cluster_scatter,      {}),
    "degree_variance":      (plot_degree_variance,      {}),
    "dissatisfaction":      (plot_dissatisfaction_vs_alpha, {}),
    "convergence_time":     (plot_convergence_time,     {}),
    "dev_distribution":     (plot_dev_distribution_effect, {}),
    "key_figure":           (plot_key_figure,           {}),
    "summary":              (print_summary,             {}),
}

ALL_PLOTS = list(PLOT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Dev-SW sweep results")
    parser.add_argument("--csv",     required=True,  help="Path to results.csv")
    parser.add_argument("--outdir",  default="figures", help="Output directory for figures")
    parser.add_argument("--plots",   nargs="+", default=ALL_PLOTS,
                        choices=ALL_PLOTS + ["all"],
                        help="Which plots to generate (default: all)")
    parser.add_argument("--x-axis", default="q_over_N",
                        choices=["q_over_N", "q_over_F", "q", "p", "alpha", "dis_threshold"],
                        help="X-axis variable for order parameter / cluster plots")
    parser.add_argument("--filter",  nargs="*", dest="filters",
                        help="Pandas query strings to filter data, e.g. 'weight_mode==2'")
    parser.add_argument("--pdf",     action="store_true", help="Also save PDF versions")
    args = parser.parse_args()

    df = load_data(args.csv, filters=args.filters)

    plots = ALL_PLOTS if "all" in args.plots else args.plots
    print(f"\nGenerating {len(plots)} plot(s) → {args.outdir}/\n")

    for name in plots:
        print(f"Plotting: {name}")
        fn, defaults = PLOT_REGISTRY[name]
        kwargs = {**defaults, "outdir": args.outdir}

        # Inject x_axis where relevant
        if name in ("order_parameter", "num_clusters", "dev_distribution"):
            kwargs["x_col"] = args.x_axis

        fn(df, **kwargs)

    print(f"\nDone. All figures in → {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()