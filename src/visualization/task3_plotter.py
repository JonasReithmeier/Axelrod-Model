import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    db_path = Path("data/schelling/schelling_master_results.parquet")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    # 1. Load Data
    print("Loading database...")
    df = pd.read_parquet(db_path)
    
    # 2. Calculate scaled q (X-axis)
    df['N'] = df['L'] * df['L']
    df['q_N'] = df['q'] / df['N']

    # 3. Setup output directory
    out_dir = Path("plots/task3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter for F=3 as per the new config
    target_F = 3
    print(f"Filtering data for F={target_F}...")
    df_f = df[df['F'] == target_F]

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
    # PLOT 1: Low Empty Density (Reproducing the new FIG 1)
    # Fixed: h = 0.05
    # Varying: L in [20, 30, 40] and T in [0.2, 0.8]
    # =========================================================
    print("Generating Plot 1 (Low Empty Density h=0.05)...")
    h1 = 0.05
    
    mask_p1 = np.isclose(df_f['h'], h1)
    df_p1 = df_f[mask_p1]
    
    plt.figure(figsize=(7, 5))
    
    widths_p1 = [20, 30, 40]
    Ts_p1 = [0.2, 0.8] # Using your config's T values instead of 0.3/0.7
    
    # Mapping exact markers and colors from the paper
    styles_p1 = {
        (20, 0.2): {'marker': 'o', 'color': 'gray'},
        (20, 0.8): {'marker': 's', 'color': 'coral'},
        (30, 0.2): {'marker': 'D', 'color': 'forestgreen'},
        (30, 0.8): {'marker': '^', 'color': 'mediumpurple'},
        (40, 0.2): {'marker': 'v', 'color': 'orchid'},
        (40, 0.8): {'marker': '>', 'color': 'palevioletred'}
    }

    plotted_lines = 0
    for w in widths_p1:
        for t_val in Ts_p1:
            d = df_p1[(df_p1['L'] == w) & np.isclose(df_p1['T'], t_val)]
            if d.empty: 
                continue
            
            grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
            style = styles_p1[(w, t_val)]
            
            plt.plot(grouped['q_N'], grouped['s_max'], 
                     marker=style['marker'], linestyle='--', linewidth=1,
                     color=style['color'], markerfacecolor='none', markeredgewidth=1.2,
                     label=f'L={w} T={t_val}')
            plotted_lines += 1

    plt.xlabel('q/N')
    plt.ylabel(r'$\langle S_{max} \rangle / N$')
    plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h1}$)', fontsize=15, pad=15)
    plt.xlim(0, 6)
    plt.ylim(0, 1.05)
    
    if plotted_lines > 0:
        plt.legend(loc='upper right', framealpha=1.0, edgecolor='black')
        
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_low_density_h0.05.png", dpi=300, bbox_inches='tight')
    plt.close()


    # =========================================================
    # PLOT 2: Varying Lattice Size L (Reproducing FIG 2a)
    # Fixed: h = 0.5, T = 0.8 (Using 0.8 from config instead of 0.7)
    # =========================================================
    print("Generating Plot 2 (Lattice Size Scaling)...")
    h2 = 0.5
    T2 = 0.2
    
    mask_p2 = np.isclose(df_f['h'], h2) & np.isclose(df_f['T'], T2)
    df_p2 = df_f[mask_p2]
    
    plt.figure(figsize=(7, 5))
    
    widths_p2 = [10, 20, 30, 40]
    markers_L = {10: 'v', 20: 'o', 30: 's', 40: 'D'}
    colors_L = {10: 'forestgreen', 20: 'gray', 30: 'coral', 40: 'mediumpurple'}

    plotted_lines = 0
    for w in widths_p2:
        d = df_p2[df_p2['L'] == w]
        if d.empty: 
            continue
        
        grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
        
        plt.plot(grouped['q_N'], grouped['s_max'], 
                 marker=markers_L[w], linestyle='--', linewidth=1,
                 color=colors_L[w], markerfacecolor='none', markeredgewidth=1.2,
                 label=f'L={w}')
        plotted_lines += 1

    plt.xlabel('q/N')
    plt.ylabel(r'$\langle S_{max} \rangle / N$')
    plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h2}$, $T={T2}$)', fontsize=15, pad=15)
    plt.xlim(0, 6)
    plt.ylim(0, 1.05)
    
    if plotted_lines > 0:
        plt.legend(loc='upper right', framealpha=1.0, edgecolor='black')
        
    plt.tight_layout()
    plt.savefig(out_dir / "fig2a_lattice_scaling_h0.5.png", dpi=300, bbox_inches='tight')
    plt.close()


    # =========================================================
    # PLOT 3: Varying Tolerance T (Reproducing FIG 2b)
    # Fixed: h = 0.5, L = 40
    # =========================================================
    print("Generating Plot 3 (Tolerance Scaling)...")
    
    mask_p3 = np.isclose(df_f['h'], 0.5) & (df_f['L'] == 40)
    df_p3 = df_f[mask_p3]
    
    plt.figure(figsize=(7, 5))
    
    Ts_p3 = [0.2, 0.5, 0.8]
    markers_T = {0.2: '^', 0.5: 's', 0.8: 'o'}
    colors_T = {0.2: 'orchid', 0.5: 'coral', 0.8: 'gray'}

    plotted_lines = 0
    for t_val in Ts_p3:
        d = df_p3[np.isclose(df_p3['T'], t_val)]
        if d.empty: 
            continue
        
        grouped = d.groupby('q_N')['s_max'].mean().reset_index().sort_values('q_N')
        
        plt.plot(grouped['q_N'], grouped['s_max'], 
                 marker=markers_T[t_val], linestyle='--', linewidth=1,
                 color=colors_T[t_val], markerfacecolor='none', markeredgewidth=1.2,
                 label=f'T={t_val}')
        plotted_lines += 1

    plt.xlabel('q/N')
    plt.ylabel(r'$\langle S_{max} \rangle / N$')
    plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h2}$, $L=40$)', fontsize=15, pad=15)
    plt.xlim(0, 6)
    plt.ylim(0, 1.05)
    
    if plotted_lines > 0:
        plt.legend(loc='upper right', framealpha=1.0, edgecolor='black')
        
    plt.tight_layout()
    plt.savefig(out_dir / "fig2b_tolerance_scaling_h0.5.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDone! Plots saved to {out_dir}/")

if __name__ == "__main__":
    main()