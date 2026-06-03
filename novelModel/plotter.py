import os
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_plot(config):
    # Load dataset
    csv_path = config['data_path']
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 1. Apply filters to isolate the desired sweep parameters
    filters = config.get('filters', {})
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]
        else:
            print(f"Warning: Filter column '{col}' not found in dataset. Skipping.")
            
    if df.empty:
        raise ValueError("The dataset is empty after applying the specified filters. Check your filter values.")

    # 2. Construct the X-axis variable (with optional normalization)
    x_var = config['x_var']
    norm_col = config.get('normalize_x_by')
    
    if norm_col:
        df['x_plot'] = df[x_var] / df[norm_col]
    else:
        df['x_plot'] = df[x_var]

    # 3. Aggregate data over realizations (calculating mean and standard error of the mean)
    group_col = config['group_by']
    y_col = config['y_var']
    
    # Group by the curve identifier and the x-axis value
    grouped = df.groupby([group_col, 'x_plot'])[y_col].agg(['mean', 'std', 'count']).reset_index()
    # Standard Error of the Mean (SEM) = std / sqrt(N)
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    
    # Sort groups to ensure continuous line plotting
    grouped = grouped.sort_values(by=[group_col, 'x_plot'])

    # 4. Initialize Plot Style
    style = config.get('style', {})
    plt.rcParams.update({'font.size': style.get('font_size', 12)})
    fig, ax = plt.subplots(figsize=style.get('aspect_ratio', [7, 6]))
    
    # Classic Physical Review marker and color cycle
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = ['black', 'red', 'blue', 'magenta', 'green', 'orange', 'cyan']
    
    unique_groups = sorted(grouped[group_col].unique())
    
    for i, grp_val in enumerate(unique_groups):
        sub_df = grouped[grouped[group_col] == grp_val]
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        label = f"{group_col}={int(grp_val) if grp_val.is_integer() else grp_val}"
        
        if style.get('show_error_bars', True) and 'sem' in sub_df.columns:
            ax.errorbar(
                sub_df['x_plot'], sub_df['mean'], yerr=sub_df['sem'],
                label=label,
                marker=marker, color=color, markerfacecolor='none',
                linestyle=style.get('line_style', '--'), 
                linewidth=1.2, markersize=style.get('marker_size', 7),
                markeredgewidth=1.2, capsize=3, elinewidth=1.0
            )
        else:
            ax.plot(
                sub_df['x_plot'], sub_df['mean'],
                label=label,
                marker=marker, color=color, markerfacecolor='none',
                linestyle=style.get('line_style', '--'), 
                linewidth=1.2, markersize=style.get('marker_size', 7),
                markeredgewidth=1.2
            )

    # 5. Fine-tune Axes & Labels (matching the target visual style)
    ax.set_xlabel(style.get('xlabel', 'x'), fontsize=style.get('font_size', 14) + 2)
    ax.set_ylabel(style.get('ylabel', 'y'), fontsize=style.get('font_size', 14) + 2)
    
    # Inward-pointing tick marks on all sides
    ax.tick_params(direction='in', top=True, right=True, which='both', length=6, width=1)
    ax.tick_params(direction='in', which='minor', length=3)
    
    # Limits and layout spacing
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=-0.05)
    
    # Legend style: elegant, boxed, matching the example image
    ax.legend(
        loc='upper right', 
        frameon=True, 
        edgecolor='black', 
        fancybox=False, 
        framealpha=1.0,
        fontsize=style.get('font_size', 12) + 1
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.get('output_path', 'novelModel/plots/output_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate customizable academic plots from simulation CSVs.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="novelModel/plot_config.yaml", 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    try:
        config_data = load_config(args.config)
        generate_plot(config_data)
    except Exception as e:
        print(f"Error: {e}")