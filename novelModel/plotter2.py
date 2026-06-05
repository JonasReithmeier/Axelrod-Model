import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
PLOT_CONFIG = {
    "csv_path": "novelModel/data/results_no_move_if_omega_is_0.csv",
    "output_dir": "novelModel/plots/plotter2/plots_no_move_if_omega_is_0",
    # Cluster metrics to plot on the Y-axis
    "metrics": ["num_clusters", "s_max", "C"],
    # Variables for splitting and mapping
    "facet_var": "alpha",  # One plot per unique value of alpha
    "x_var": "q_over_N",  # Calculated as q / N
    "color_var": "dis_threshold",  # Maps to distinct colors
    "marker_var": "dev_mode",  # Maps to marker styles
    "linestyle_var": "weight_mode",  # Maps to line styles
    # Aesthetic mappings
    "colors": {
        0.0: "#1f77b4",  # Blue
        0.3: "#ff7f0e",  # Orange
        0.7: "#2ca02c",  # Green
        1.0: "#d62728",  # Red
    },
    "markers": {
        0: "o",  # Circle for Uniform/Mode 0
        2: "^",  # Triangle for Pareto/Mode 2
    },
    "linestyles": {
        0: "-",  # Solid line for Linear
        1: "--",  # Dashed line for Quadratic
    },
}


# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(config):
    if not os.path.exists(config["csv_path"]):
        raise FileNotFoundError(f"CSV file not found at: {config['csv_path']}")

    df = pd.read_csv(config["csv_path"])

    # Calculate the x-axis variable: q/N
    df[config["x_var"]] = df["q"] / df["N"]

    return df


# ==========================================
# PLOTTING FUNCTION
# ==========================================
def generate_plots(df, config):
    os.makedirs(config["output_dir"], exist_ok=True)

    # Identify unique alpha values to slice the data
    alphas = sorted(df[config["facet_var"]].unique())

    # Group variables used to aggregate realizations
    group_cols = [
        config["facet_var"],
        config["color_var"],
        config["marker_var"],
        config["linestyle_var"],
        config["x_var"],
    ]

    for metric in config["metrics"]:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the CSV. Skipping.")
            continue

        # Aggregate the realizations (calculating mean and std/sem)
        aggregated = (
            df.groupby(group_cols)[metric].agg(["mean", "std"]).reset_index()
        )

        for alpha in alphas:
            alpha_data = aggregated[aggregated[config["facet_var"]] == alpha]
            if alpha_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            # Group within the specific slice to plot individual lines
            grouped_lines = alpha_data.groupby(
                [config["color_var"], config["marker_var"], config["linestyle_var"]]
            )

            for (color_val, marker_val, line_val), group in grouped_lines:
                # Retrieve visual properties from configuration
                color = config["colors"].get(color_val, "#7f7f7f")
                marker = config["markers"].get(marker_val, "x")
                linestyle = config["linestyles"].get(line_val, "-")

                # Sort values by X-axis to ensure lines connect correctly
                group = group.sort_values(by=config["x_var"])

                # Plot mean value line
                ax.plot(
                    group[config["x_var"]],
                    group["mean"],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=6,
                    linewidth=1.5,
                    label="_nolegend_",  # Handled via custom structured legend
                )

                # Add shaded error area if standard deviation is available and valid
                if "std" in group.columns and not group["std"].isna().all():
                    ax.fill_between(
                        group[config["x_var"]],
                        group["mean"] - group["std"],
                        group["mean"] + group["std"],
                        color=color,
                        alpha=0.1,
                        label="_nolegend_",
                    )

            # Formatting
            ax.set_title(
                f"Metric: {metric} | {config['facet_var']} = {alpha}",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(r"Ratio $q/N$", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.6)

            # Construct structured legend
            legend_elements = []

            # 1. Colors (dis_threshold)
            legend_elements.append(
                mlines.Line2D(
                    [],
                    [],
                    color="none",
                    label=r"$\bf{Threshold\ (dis\_threshold)}$",
                )
            )
            for val, col in config["colors"].items():
                if val in alpha_data[config["color_var"]].values:
                    legend_elements.append(
                        mlines.Line2D([0], [0], color=col, lw=2, label=f"{val}")
                    )

            # Spacer
            legend_elements.append(mlines.Line2D([], [], color="none", label=""))

            # 2. Markers (dev_mode)
            legend_elements.append(
                mlines.Line2D([], [], color="none", label=r"$\bf{Dev\ Mode\ (dev\_mode)}$")
            )
            for val, mark in config["markers"].items():
                if val in alpha_data[config["marker_var"]].values:
                    legend_elements.append(
                        mlines.Line2D(
                            [0],
                            [0],
                            color="gray",
                            marker=mark,
                            linestyle="None",
                            markersize=8,
                            label=f"{val}",
                        )
                    )

            # Spacer
            legend_elements.append(mlines.Line2D([], [], color="none", label=""))

            # 3. Linestyles (weight_mode)
            legend_elements.append(
                mlines.Line2D(
                    [], [], color="none", label=r"$\bf{Weight\ Mode\ (weight\_mode)}$"
                )
            )
            for val, ls in config["linestyles"].items():
                if val in alpha_data[config["linestyle_var"]].values:
                    legend_elements.append(
                        mlines.Line2D(
                            [0], [0], color="gray", linestyle=ls, lw=2, label=f"{val}"
                        )
                    )

            # Place legend outside or adjust layout to avoid overlapping data points
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                borderaxespad=0.0,
                frameon=True,
                fontsize=9,
            )

            plt.tight_layout()

            # Save plot
            file_name = f"{metric}_alpha_{alpha}.png".replace(".", "_dot_")
            save_path = os.path.join(config["output_dir"], file_name)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {save_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        data = load_and_preprocess_data(PLOT_CONFIG)
        generate_plots(data, PLOT_CONFIG)
    except Exception as e:
        print(f"An error occurred during execution: {e}")