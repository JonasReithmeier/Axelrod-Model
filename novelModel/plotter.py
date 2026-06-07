import os
import argparse
import itertools
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_config(config_path="novelModel/plot_config.yaml"):
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_style_generator():
    """Generates cycling markers and colors for plotting curves."""
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '>', '<', 'p', 'h', '*'])
    colors = itertools.cycle([
        'black', 'red', 'blue', 'magenta', 'forestgreen', 'darkorange', 'teal', 'mediumpurple'
    ])
    return markers, colors

def safe_filter(df, filter_dict):
    """Filters dataframe safely, handling floating point precision with np.isclose."""
    mask = np.ones(len(df), dtype=bool)
    for col, val in filter_dict.items():
        if col not in df.columns:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            mask &= np.isclose(df[col], val)
        else:
            mask &= (df[col] == val)
    return df[mask]

def main():
    parser = argparse.ArgumentParser(description="Multi-plot generator for novelModel results.")
    parser.add_argument("--config", type=str, default="novelModel/plot_config.yaml", help="Path to config file.")
    args = parser.parse_args()

    # 1. Load Configuration
    full_config = load_config(args.config)
    if 'novel_plotter' not in full_config:
        print("Error: 'novel_plotter' section missing from config file.")
        return
    cfg = full_config['novel_plotter']

    db_path = Path(cfg.get('input_file', "novelModel/data/results.csv"))
    if not db_path.exists():
        print(f"Dataset file not found at {db_path}")
        return

    # 2. Load Data
    print(f"\n--- Loading database from {db_path} ---")
    df = pd.read_csv(db_path)
    print(f"Total entries loaded: {len(df)}")
    
    # Calculate relative coordinates (X-axis)
    if 'q' in df.columns and 'N' in df.columns:
        df['q_N'] = df['q'] / df['N']
    else:
        print("CRITICAL: 'q' or 'N' column missing from dataset.")
        return

    # Setup output directory
    out_dir = Path(cfg.get('output_dir', "plots/novel_model"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply global filters
    global_filters = cfg.get('global_filters', {})
    df_global = safe_filter(df, global_filters)
    print(f"Entries after applying global filters: {len(df_global)}")
    
    if df_global.empty:
        print("Warning: No records left after global filtering.")
        return

    # Academic-standard style adjustments
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--'
    })

    # 3. Process and Plot Each Target Run
    plots_list = cfg.get('plots', [])
    for idx, plot_cfg in enumerate(plots_list, 1):
        plot_name = plot_cfg.get('name', f"plot_{idx}")
        plot_title = plot_cfg.get('title', "")
        
        # Merge global filters with localized figure filters
        local_filters = plot_cfg.get('filters', {})
        df_local = safe_filter(df_global, local_filters)
        
        if df_local.empty:
            print(f"\n[Skipping] '{plot_name}': No data matches filters {local_filters}")
            continue

        print(f"\nGenerating '{plot_name}'...")
        
        # Get variable parameter listings for generating curves
        curve_vars = plot_cfg.get('curve_variables', {})
        if not curve_vars:
            print(f"  -> Skipping: No curve_variables defined for '{plot_name}'")
            continue
            
        # Create a Cartesian product of all variable lists to find all lines
        keys, values = zip(*curve_vars.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        plt.figure(figsize=(6.5, 5.2))
        markers, colors = get_style_generator()
        plotted_lines = 0

        for comb in combinations:
            # Filter specifically down to this line's parameters
            df_curve = safe_filter(df_local, comb)
            if df_curve.empty:
                continue
            
            # Aggregate multiple realization trials via mean and standard error
            grouped = df_curve.groupby('q_N')['s_max'].agg(['mean', 'std', 'count']).reset_index().sort_values('q_N')
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Build legend label dynamically
            label_parts = [f"{k}={v}" for k, v in comb.items()]
            label_text = ", ".join(label_parts)
            
            # Render lines with empty marker shapes and error boundaries
            plt.errorbar(
                grouped['q_N'], grouped['mean'], yerr=grouped['sem'].fillna(0),
                marker=next(markers), linestyle='--', linewidth=1.2,
                color=next(colors), markerfacecolor='none', markeredgewidth=1.2,
                capsize=3, elinewidth=0.8, label=label_text
            )
            plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('$q / N$')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            if plot_title:
                plt.title(plot_title, fontsize=12, pad=10)
                
            plt.xlim(left=-0.05)
            plt.ylim(-0.02, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black', fancybox=False)
            plt.tight_layout()
            
            output_file = out_dir / f"{plot_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  -> Saved successfully: {output_file}")
        else:
            print(f"  -> Skipping '{plot_name}': No valid combinations found.")
        plt.close()

    print(f"\nDone! Figures saved in: {out_dir.resolve()}/")

if __name__ == "__main__":
    main()