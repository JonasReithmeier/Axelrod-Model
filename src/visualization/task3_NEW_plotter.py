import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import itertools

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_color_generator():
    """Returns a dynamic generator for colors."""
    colors = itertools.cycle([
        'dodgerblue', 'forestgreen', 'darkorange', 'mediumpurple', 
        'orchid', 'palevioletred', 'teal', 'coral', 'gray'
    ])
    return colors

def format_list_for_fname(lst):
    """Helper to cleanly format lists into filename strings (e.g., [0.3, 0.7] -> '0.3-0.7')"""
    return "-".join(map(str, lst))

def format_list_for_fname_inverted(lst):
    """Formats list into filename strings while inverting values (1 - T)"""
    inverted_lst = [round(1.0 - x, 4) for x in lst]
    return "-".join(map(str, inverted_lst))

def generate_dynamic_plot(df_f, cfg_dyn, target_F, out_dir):
    """Generates a single condensed line plot with dynamic legend/heading roll logic."""
    L_list = cfg_dyn.get('L_values', [40])
    T_list = cfg_dyn.get('T_values', [0.0])
    h_list = cfg_dyn.get('h_values', [0.1])

    # Determine which parameters are fixed (length == 1) and which are rolled (length > 1)
    fixed_params = []
    rollable_params = []

    if len(L_list) == 1:
        fixed_params.append(f"L={L_list[0]}")
    else:
        rollable_params.append("L")

    if len(T_list) == 1:
        fixed_params.append(f"T={round(1.0 - T_list[0], 4)}")
    else:
        rollable_params.append("T")

    if len(h_list) == 1:
        fixed_params.append(f"h={h_list[0]}")
    else:
        rollable_params.append("h")

    print("\n--- Running Dynamic Plotter ---")
    print(f"Fixed parameters (in heading): {fixed_params}")
    print(f"Rollable parameters (in legend): {rollable_params}")

    plt.figure(figsize=(8, 6))
    colors = get_color_generator()
    plotted_lines = 0

    # Iterate over all combinations of parameters
    for w, t_val, h_val in itertools.product(L_list, T_list, h_list):
        d = df_f[(df_f['L'] == w) & np.isclose(df_f['T'], t_val) & np.isclose(df_f['h'], h_val)]
        if d.empty:
            print(f"  [Warning] No data for L={w}, T={round(1.0 - t_val, 4)}, h={h_val}")
            continue

        # Calculate mean and standard error of the mean (sem) for raw MCS
        grouped = d.groupby('q_N')['s_max'].agg(['mean', 'sem']).reset_index().sort_values('q_N')
        grouped['sem'] = grouped['sem'].fillna(0)

        # Build legend label using ONLY rollable parameters
        label_parts = []
        if "L" in rollable_params:
            label_parts.append(f"L={w}")
        if "T" in rollable_params:
            label_parts.append(f"T={round(1.0 - t_val, 4)}")
        if "h" in rollable_params:
            label_parts.append(f"h={h_val}")
        
        label_str = " ".join(label_parts) if label_parts else "All parameters fixed"

        # Special styling for T=0.0 (visual T=1.0)
        if np.isclose(t_val, 0.0):
            linestyle = '--'
            marker = 'x'
        else:
            linestyle = '-'
            marker = 'o'

        plt.errorbar(
            grouped['q_N'], grouped['mean'], yerr=grouped['sem'],
            fmt=marker + linestyle, linewidth=1.2, color=next(colors),
            markerfacecolor='none', markeredgewidth=1.2,
            capsize=3, elinewidth=1, capthick=1,
            label=label_str
        )
        plotted_lines += 1

    if plotted_lines > 0:
        plt.xlabel('q/N')
        plt.ylabel(r'$\langle \text{MCS} \rangle$')

        # Force scientific notation with 10^n multiplier on the Y-axis
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # Dynamically build plot title based on fixed parameters
        title_str = f"Phase Transition on Schelling-Axelrod Model\n(F={target_F}"
        if fixed_params:
            title_str += f", {', '.join(fixed_params)}"
        title_str += ")"
        
        plt.title(title_str, fontsize=14, pad=15)
        plt.legend(loc='best', framealpha=1.0, edgecolor='black')
        plt.tight_layout()

        # Generate a dynamic informative filename
        fname = f"dynamic_mcs_plot_L{format_list_for_fname(L_list)}_T{format_list_for_fname_inverted(T_list)}_h{format_list_for_fname(h_list)}.png"
        plt.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
        print(f"  -> Saved dynamic plot: {fname}")
    else:
        print("  -> Skipped: No matching lines to plot.")
    plt.close()

def main():
    # 0. Load Configuration
    full_config = load_config()
    if 'schelling_plotter' not in full_config:
        print("Error: 'schelling_plotter' entry missing from config.yaml")
        return
    cfg = full_config['s_max_schelling_plotter']

    db_path = Path(cfg.get('input_file', "data/schelling/schelling_master_results.parquet"))
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    # 1. Load Data
    print(f"\n--- Loading database from {db_path} ---")
    df = pd.read_parquet(db_path)
    df = df[df['is_constant'] == True] # only terminated data gets into plots
    print(f"Total rows loaded: {len(df)}")
    
    # 2. Calculate scaled q (X-axis)
    df['N'] = df['L'] * df['L']
    df['q_N'] = df['q'] / df['N']

    # 3. Setup output directory
    out_dir = Path(cfg.get('output_dir', "plots/task3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter for target F
    target_F = cfg.get('target_F', 3)
    df_f = df[df['F'] == target_F]
    print(f"Rows after filtering for F={target_F}: {len(df_f)}")
    
    if df_f.empty:
        print(f"CRITICAL: No data found for F={target_F}. Available F values: {df['F'].unique()}")
        return

    # Matplotlib styling for publication-quality plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    # Run dynamic plotter
    if 'plot_dynamic' in cfg:
        generate_dynamic_plot(df_f, cfg['plot_dynamic'], target_F, out_dir)
    else:
        print("Error: 'plot_dynamic' entry missing from configuration.")

if __name__ == "__main__":
    main()