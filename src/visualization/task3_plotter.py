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

def get_style_generator():
    """Returns dynamic generators for markers and colors to prevent KeyErrors on new params."""
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '>', '<', 'p', 'h', '*'])
    colors = itertools.cycle([
        'gray', 'coral', 'forestgreen', 'mediumpurple', 'orchid', 
        'palevioletred', 'dodgerblue', 'darkorange', 'teal'
    ])
    return markers, colors

def format_list_for_fname(lst):
    """Helper to cleanly format lists into filename strings (e.g., [0.3, 0.7] -> '0.3-0.7')"""
    return "-".join(map(str, lst))

def main():
    # 0. Load Configuration
    full_config = load_config()
    if 'schelling_plotter' not in full_config:
        print("Error: 'schelling_plotter' entry missing from config.yaml")
        return
    cfg = full_config['schelling_plotter']

    db_path = Path(cfg.get('input_file', "data/schelling/schelling_master_results.parquet"))
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    # 1. Load Data
    print(f"\n--- Loading database from {db_path} ---")
    df = pd.read_parquet(db_path)
    print(f"Total rows loaded: {len(df)}")
    
    # 2. Calculate scaled q (X-axis)
    df['N'] = df['L'] * df['L']
    df['q_N'] = df['q'] / df['N']

    # 3. Setup output directory
    out_dir = Path(cfg.get('output_dir', "plots/task3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global variables from config
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

    # =========================================================
    # PLOT 1: Low Empty Density
    # =========================================================
    p1_cfg = cfg['plot1_low_density']
    h1 = p1_cfg['h']
    widths_p1 = p1_cfg['L_values']
    Ts_p1 = p1_cfg['T_values']
    
    print(f"\nGenerating Plot 1 (Low Empty Density h={h1})...")
    mask_p1 = np.isclose(df_f['h'], h1)
    df_p1 = df_f[mask_p1]
    print(f"  -> Rows matching h={h1}: {len(df_p1)}")
    
    if not df_p1.empty:
        plt.figure(figsize=(7, 5))
        markers_p1, colors_p1 = get_style_generator()
        plotted_lines = 0
        
        for w in widths_p1:
            for t_val in Ts_p1:
                d = df_p1[(df_p1['L'] == w) & np.isclose(df_p1['T'], t_val)]
                if d.empty: 
                    print(f"     [Warning] No data for L={w}, T={t_val}")
                    continue
                
                grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
                plt.plot(grouped['q_N'], grouped['s_max'], 
                         marker=next(markers_p1), linestyle='--', linewidth=1,
                         color=next(colors_p1), markerfacecolor='none', markeredgewidth=1.2,
                         label=f'L={w} T={t_val}')
                plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h1}$)', fontsize=15, pad=15)
            plt.xlim(0, max(6, df_p1['q_N'].max() * 1.1))  # Dynamic x-lim based on data
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname1 = f"fig1_low_density_h{h1}_F{target_F}_L{format_list_for_fname(widths_p1)}_T{format_list_for_fname(Ts_p1)}.png"
            plt.savefig(out_dir / fname1, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname1}")
        else:
            print("  -> Plot 1 skipped: No lines to plot.")
        plt.close()

    # =========================================================
    # PLOT 2: Varying Lattice Size L
    # =========================================================
    p2_cfg = cfg['plot2_lattice_scaling']
    h2 = p2_cfg['h']
    T2 = p2_cfg['T']
    widths_p2 = p2_cfg['L_values']

    print(f"\nGenerating Plot 2 (Lattice Size Scaling h={h2}, T={T2})...")
    mask_p2 = np.isclose(df_f['h'], h2) & np.isclose(df_f['T'], T2)
    df_p2 = df_f[mask_p2]
    print(f"  -> Rows matching h={h2}, T={T2}: {len(df_p2)}")
    
    if not df_p2.empty:
        plt.figure(figsize=(7, 5))
        markers_p2, colors_p2 = get_style_generator()
        plotted_lines = 0

        for w in widths_p2:
            d = df_p2[df_p2['L'] == w]
            if d.empty: 
                print(f"     [Warning] No data for L={w}")
                continue
            
            grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
            plt.plot(grouped['q_N'], grouped['s_max'], 
                     marker=next(markers_p2), linestyle='--', linewidth=1,
                     color=next(colors_p2), markerfacecolor='none', markeredgewidth=1.2,
                     label=f'L={w}')
            plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h2}$, $T={T2}$)', fontsize=15, pad=15)
            plt.xlim(0, max(6, df_p2['q_N'].max() * 1.1))
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname2 = f"fig2a_lattice_scaling_h{h2}_F{target_F}_T{T2}_L{format_list_for_fname(widths_p2)}.png"
            plt.savefig(out_dir / fname2, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname2}")
        else:
            print("  -> Plot 2 skipped: No lines to plot.")
        plt.close()

    # =========================================================
    # PLOT 3: Varying Tolerance T
    # =========================================================
    p3_cfg = cfg['plot3_tolerance_scaling']
    h3 = p3_cfg['h']
    L3 = p3_cfg['L']
    Ts_p3 = p3_cfg['T_values']

    print(f"\nGenerating Plot 3 (Tolerance Scaling h={h3}, L={L3})...")
    mask_p3 = np.isclose(df_f['h'], h3) & (df_f['L'] == L3)
    df_p3 = df_f[mask_p3]
    print(f"  -> Rows matching h={h3}, L={L3}: {len(df_p3)}")
    
    if not df_p3.empty:
        plt.figure(figsize=(7, 5))
        markers_p3, colors_p3 = get_style_generator()
        plotted_lines = 0

        for t_val in Ts_p3:
            d = df_p3[np.isclose(df_p3['T'], t_val)]
            if d.empty: 
                print(f"     [Warning] No data for T={t_val}")
                continue
            
            grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
            plt.plot(grouped['q_N'], grouped['s_max'], 
                     marker=next(markers_p3), linestyle='--', linewidth=1,
                     color=next(colors_p3), markerfacecolor='none', markeredgewidth=1.2,
                     label=f'T={t_val}')
            plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h3}$, $L={L3}$)', fontsize=15, pad=15)
            plt.xlim(0, max(6, df_p3['q_N'].max() * 1.1))
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname3 = f"fig2b_tolerance_scaling_h{h3}_F{target_F}_L{L3}_T{format_list_for_fname(Ts_p3)}.png"
            plt.savefig(out_dir / fname3, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname3}")
        else:
            print("  -> Plot 3 skipped: No lines to plot.")
        plt.close()

    print(f"\nDone! Output directory: {out_dir}/")

if __name__ == "__main__":
    main()