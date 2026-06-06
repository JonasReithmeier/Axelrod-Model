import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import itertools
import re

# Categorical mappings
DEV_MODE_MAP = {
    0: "UNIFORM",
    1: "NORMAL",
    2: "PARETO",
    3: "BIMODAL"
}

WEIGHT_MODE_MAP = {
    0: "LINEAR",
    1: "QUADRATIC",
    2: "BIPHASIC",
    3: "ATTRACTION"
}

# Human-readable labels for axes
LABEL_MAP = {
    'dis_threshold': 'T',
    's_max': r'$\langle s_{max} \rangle$',
    'L': 'L',
    'mean_dissatisfaction': 'Mean Dissatisfaction',
    'degree_variance': 'Degree Variance',
    'q': 'q',
    'N': 'N'
}

def load_config(config_path="novelModel/config_plotter3.yaml"):
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_style_generators():
    """Returns dynamic generators for colors, markers, and linestyles."""
    colors = itertools.cycle([
        'dodgerblue', 'forestgreen', 'darkorange', 'mediumpurple', 
        'orchid', 'palevioletred', 'teal', 'coral', 'gray'
    ])
    markers = itertools.cycle(['o', 's', '^', 'D', 'x', 'v', '*'])
    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    return colors, markers, linestyles

def format_param_name(param):
    """Translates parameter names to clean labels."""
    return LABEL_MAP.get(param, param)

def format_param_val(param, val):
    """Translates parameter values into legible strings."""
    if param == 'dev_mode':
        return DEV_MODE_MAP.get(val, str(val))
    elif param == 'weight_mode':
        return WEIGHT_MODE_MAP.get(val, str(val))
    elif param == 'dis_threshold':
        return str(round(1.0 - val, 4))
    elif isinstance(val, float):
        return f"{val:.4g}"
    return str(val)

def format_list_for_fname(lst, param_name):
    """Helper to cleanly format list variables into safe filename segments."""
    formatted_vals = [format_param_val(param_name, v) for v in lst]
    raw_str = "-".join(formatted_vals)
    return re.sub(r'[^\w\-_.]', '_', raw_str)

def get_axis_label(axis_name, is_normalized, norm_by):
    """Generates a structured label for an axis, reflecting division if normalized."""
    base_label = LABEL_MAP.get(axis_name, axis_name)
    if is_normalized and norm_by:
        norm_label = LABEL_MAP.get(norm_by, norm_by)
        return f"{base_label}/{norm_label}"
    return base_label

def generate_dynamic_plot(df, cfg, out_dir):
    """Generates a line plot with dynamic normalization, scale settings, legend, and heading logic."""
    x_axis = cfg.get('x_axis', 'q')
    y_axis = cfg.get('y_axis', 's_max')
    cfg_params = cfg.get('parameters', {})

    # 1. Handle Normalization Row-by-Row
    x_col = x_axis
    y_col = y_axis
    x_normalized = cfg.get('x_normalize', False)
    x_norm_by = cfg.get('x_normalize_by')
    y_normalized = cfg.get('y_normalize', False)
    y_norm_by = cfg.get('y_normalize_by')

    if x_normalized and x_norm_by:
        if x_norm_by in df.columns:
            x_col = f"{x_axis}_normalized_by_{x_norm_by}"
            df[x_col] = df[x_axis] / df[x_norm_by]
        else:
            print(f"[Warning] Normalization column '{x_norm_by}' not found for X-axis. Normalization skipped.")
            x_normalized = False

    if y_normalized and y_norm_by:
        if y_norm_by in df.columns:
            y_col = f"{y_axis}_normalized_by_{y_norm_by}"
            df[y_col] = df[y_axis] / df[y_norm_by]
        else:
            print(f"[Warning] Normalization column '{y_norm_by}' not found for Y-axis. Normalization skipped.")
            y_normalized = False

    # 2. Determine Fixed vs Dynamic (Rollable) Control Parameters (excluding raw x_axis)
    all_control_params = ['F', 'N', 'alpha', 'dev_mode', 'dev_param', 'dis_threshold', 'k', 'p', 'q', 'weight_mode']
    
    filters = {}
    fixed_params = []
    rollable_params = []

    for param in all_control_params:
        if param == x_axis:
            continue
        
        config_key = f"{param}_values"
        if config_key in cfg_params:
            vals = cfg_params[config_key]
        else:
            vals = sorted(df[param].unique().tolist())
        
        filters[param] = vals
        
        if len(vals) == 1:
            fixed_params.append(param)
        elif len(vals) > 1:
            rollable_params.append(param)

    print("\n--- Running Dynamic Plotter ---")
    print(f"X-Axis: {x_axis} (Normalized: {x_normalized} by {x_norm_by})")
    print(f"Y-Axis: {y_axis} (Normalized: {y_normalized} by {y_norm_by})")
    print(f"Fixed parameters (in heading): {fixed_params}")
    print(f"Rollable parameters (in legend): {rollable_params}")

    # Optionally filter raw X-axis coordinates prior to grouping
    x_config_key = f"{x_axis}_values"
    if x_config_key in cfg_params:
        df = df[df[x_axis].isin(cfg_params[x_config_key])]

    plt.figure(figsize=(8, 6))
    colors, markers, linestyles = get_style_generators()
    plotted_lines = 0

    # 3. Generate Combos over Parameter Space
    keys = list(filters.keys())
    values_product = [filters[k] for k in keys]

    for combo_vals in itertools.product(*values_product):
        combo = dict(zip(keys, combo_vals))
        
        # Slicing the dataframe copy
        d = df.copy()
        for param, val in combo.items():
            if isinstance(val, (float, np.floating)):
                d = d[np.isclose(d[param], val)]
            else:
                d = d[d[param] == val]

        if d.empty:
            continue

        # Aggregate on the plotting columns
        grouped = d.groupby(x_col)[y_col].agg(['mean', 'sem']).reset_index().sort_values(x_col)
        grouped['sem'] = grouped['sem'].fillna(0)

        if grouped.empty:
            continue

        # Label matching rollable attributes
        label_parts = []
        for param in rollable_params:
            label_parts.append(f"{format_param_name(param)}={format_param_val(param, combo[param])}")
        
        label_str = ", ".join(label_parts) if label_parts else "Default configuration"

        plt.errorbar(
            grouped[x_col], grouped['mean'], yerr=grouped['sem'],
            fmt=next(markers) + next(linestyles), linewidth=1.2, color=next(colors),
            markerfacecolor='none', markeredgewidth=1.2,
            capsize=3, elinewidth=1, capthick=1,
            label=label_str
        )
        plotted_lines += 1

    if plotted_lines > 0:
        plt.xlabel(get_axis_label(x_axis, x_normalized, x_norm_by))
        plt.ylabel(get_axis_label(y_axis, y_normalized, y_norm_by))

        # Apply logarithmic scale if configured
        x_log = cfg.get('x_log_scale', False)
        y_log = cfg.get('y_log_scale', False)

        if x_log:
            plt.xscale('log')
        if y_log:
            plt.yscale('log')

        # Scientific notation logic (disabled for log scale axes to avoid conflicts)
        if not y_log:
            if df[y_col].max() > 1000 or df[y_col].max() < 0.01:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # Dynamic title building
        title_parts = []
        for param in fixed_params:
            val = filters[param][0]
            title_parts.append(f"{format_param_name(param)}={format_param_val(param, val)}")
        
        title_str = f"Model Output Profile\n"
        if title_parts:
            title_str += f"({', '.join(title_parts)})"
        
        plt.title(title_str, fontsize=12, pad=15)
        plt.legend(loc='best', framealpha=1.0, edgecolor='black')
        plt.tight_layout()

        # Build path-safe filename. Varying params are listed first, then fixed ones.
        # This prevents metadata at the end of the lists (such as weight_mode) from being truncated.
        y_str = f"{y_axis}_norm_by_{y_norm_by}" if y_normalized else y_axis
        x_str = f"{x_axis}_norm_by_{x_norm_by}" if x_normalized else x_axis

        if x_log:
            x_str += "_log"
        if y_log:
            y_str += "_log"

        roll_parts = [f"{p}-{format_list_for_fname(filters[p], p)}" for p in sorted(rollable_params)]
        fix_parts = [f"{p}-{format_list_for_fname(filters[p], p)}" for p in sorted(fixed_params)]

        fname_parts = [f"plot_{y_str}_vs_{x_str}"]
        if roll_parts:
            fname_parts.append("var_" + "_".join(roll_parts))
        if fix_parts:
            fname_parts.append("fix_" + "_".join(fix_parts))
        
        fname = "_".join(fname_parts)
        fname = re.sub(r'[^\w\-_.]', '_', fname)[:220] + ".png" # Expanded character budget to 220
        
        output_path = out_dir / fname
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved plot: {output_path}")
    else:
        print("  -> Skipped: No matching parameters with data were found to plot.")
    plt.close()

def main():
    full_config = load_config()
    if 'plotter_config' not in full_config:
        print("Error: 'plotter_config' entry missing from config_plotter3.yaml")
        return
    cfg = full_config['plotter_config']

    csv_path = Path(cfg.get('input_file', "novelModel/data/results_no_move_if_omega_is_0.csv"))
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}")
        return

    print(f"\n--- Loading dataset from {csv_path} ---")
    df = pd.read_csv(csv_path)
    print(f"Total rows loaded: {len(df)}")
    
    out_dir = Path(cfg.get('output_dir', "novelModel/plots/plotter2/plots_no_move_if_omega_is_0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    generate_dynamic_plot(df, cfg, out_dir)

if __name__ == "__main__":
    main()