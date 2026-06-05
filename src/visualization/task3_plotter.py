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
        'gray', 'coral', 'forestgreen', 'mediumpurple', 'orchid', 
        'palevioletred', 'dodgerblue', 'darkorange', 'teal'
    ])
    return colors

def format_list_for_fname(lst):
    """Helper to cleanly format lists into filename strings (e.g., [0.3, 0.7] -> '0.3-0.7')"""
    return "-".join(map(str, lst))

def format_list_for_fname_inverted(lst):
    """Formats list into filename strings while inverting values (1 - T)"""
    inverted_lst = [round(1.0 - x, 4) for x in lst]
    return "-".join(map(str, inverted_lst))

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
    df = df[df['is_constant'] == True] # only terminated data gets into plots
    print(f"Total rows loaded: {len(df)}")
    
    # 2. Calculate scaled q (X-axis)
    df['N'] = df['L'] * df['L']
    df['q_N'] = df['q'] / df['N']

    # 3. Setup output directory
    out_dir = Path(cfg.get('output_dir', "plots/task3/s_max_over_q"))
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
        colors_p1 = get_color_generator()
        plotted_lines = 0
        
        for w in widths_p1:
            for t_val in Ts_p1:
                d = df_p1[(df_p1['L'] == w) & np.isclose(df_p1['T'], t_val)]
                if d.empty: 
                    print(f"     [Warning] No data for L={w}, T={t_val}")
                    continue
                
                # Calculate mean and standard deviation
                grouped = d.groupby('q_N')['s_max'].agg(['mean', 'std']).reset_index().sort_values('q_N')
                grouped['std'] = grouped['std'].fillna(0)
                
                vis_T = round(1.0 - t_val, 4)
                plt.errorbar(
                    grouped['q_N'], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', linestyle='--', linewidth=1,
                    color=next(colors_p1), markerfacecolor='none', markeredgewidth=1.2,
                    capsize=3, elinewidth=1, capthick=1,
                    label=f'L={w} T={vis_T}'
                )
                plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h1}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname1 = f"fig1_low_density_h{h1}_F{target_F}_L{format_list_for_fname(widths_p1)}_T{format_list_for_fname_inverted(Ts_p1)}.png"
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

    print(f"\nGenerating Plot 2 (Lattice Size Scaling h={h2}, T={round(1.0 - T2, 4)})...")
    mask_p2 = np.isclose(df_f['h'], h2) & np.isclose(df_f['T'], T2)
    df_p2 = df_f[mask_p2]
    print(f"  -> Rows matching h={h2}, T={T2}: {len(df_p2)}")
    
    if not df_p2.empty:
        plt.figure(figsize=(7, 5))
        colors_p2 = get_color_generator()
        plotted_lines = 0

        for w in widths_p2:
            d = df_p2[df_p2['L'] == w]
            if d.empty: 
                print(f"     [Warning] No data for L={w}")
                continue
            
            # Calculate mean and standard deviation
            grouped = d.groupby('q_N')['s_max'].agg(['mean', 'std']).reset_index().sort_values('q_N')
            grouped['std'] = grouped['std'].fillna(0)

            plt.errorbar(
                grouped['q_N'], grouped['mean'], yerr=grouped['std'],
                fmt='o-', linestyle='--', linewidth=1,
                color=next(colors_p2), markerfacecolor='none', markeredgewidth=1.2,
                capsize=3, elinewidth=1, capthick=1,
                label=f'L={w}'
            )
            plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            
            vis_T2 = round(1.0 - T2, 4)
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h2}$, $T={vis_T2}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)  
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname2 = f"fig2a_lattice_scaling_h{h2}_F{target_F}_T{vis_T2}_L{format_list_for_fname(widths_p2)}.png"
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
        colors_p3 = get_color_generator()
        plotted_lines = 0

        for t_val in Ts_p3:
            d = df_p3[np.isclose(df_p3['T'], t_val)]
            if d.empty: 
                print(f"     [Warning] No data for T={t_val}")
                continue
            
            # Calculate mean and standard deviation
            grouped = d.groupby('q_N')['s_max'].agg(['mean', 'std']).reset_index().sort_values('q_N')
            grouped['std'] = grouped['std'].fillna(0)

            vis_T = round(1.0 - t_val, 4)
            plt.errorbar(
                grouped['q_N'], grouped['mean'], yerr=grouped['std'],
                fmt='o-', linestyle='--', linewidth=1,
                color=next(colors_p3), markerfacecolor='none', markeredgewidth=1.2,
                capsize=3, elinewidth=1, capthick=1,
                label=f'T={vis_T}'
            )
            plotted_lines += 1

        if plotted_lines > 0:
            plt.xlabel('q/N')
            plt.ylabel(r'$\langle S_{max} \rangle / N$')
            plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $h={h3}$, $L={L3}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)  
            plt.ylim(0, 1.05)
            plt.legend(loc='best', framealpha=1.0, edgecolor='black')
            plt.tight_layout()
            
            fname3 = f"fig2b_tolerance_scaling_h{h3}_F{target_F}_L{L3}_T{format_list_for_fname_inverted(Ts_p3)}.png"
            plt.savefig(out_dir / fname3, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname3}")
        else:
            print("  -> Plot 3 skipped: No lines to plot.")
        plt.close()

    # =========================================================
    # PLOT 4: Discrete Grid Heatmap of S_max over q/N and T (Fixed h and L)
    # =========================================================
    if 'plot4_heatmap_q_T' in cfg:
        p4_cfg = cfg['plot4_heatmap_q_T']
        h4 = p4_cfg['h']
        L4 = p4_cfg['L']

        print(f"\nGenerating Plot 4 (Grid Heatmap q/N vs T | h={h4}, L={L4})...")
        mask_p4 = np.isclose(df_f['h'], h4) & (df_f['L'] == L4)
        df_p4 = df_f[mask_p4]
        print(f"  -> Rows matching h={h4}, L={L4}: {len(df_p4)}")

        if not df_p4.empty:
            grouped_p4 = df_p4.groupby(['T', 'q_N'])['s_max'].mean().reset_index()
            grouped_p4['T_vis'] = 1.0 - grouped_p4['T']
            
            pivot_p4 = grouped_p4.pivot(index='T_vis', columns='q_N', values='s_max')
            pivot_p4 = pivot_p4.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(
                pivot_p4.columns, pivot_p4.index, pivot_p4.values,
                cmap='viridis', shading='auto'
            )

            cbar = plt.colorbar(mesh)
            cbar.set_label(r'$\langle S_{max} \rangle / N$', rotation=270, labelpad=20)
            
            plt.xlabel('q/N')
            plt.ylabel('T (Tolerance)')
            plt.title(f'Phase Diagram (Grid Heatmap)\n($F={target_F}$, $h={h4}$, $L={L4}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)
            plt.grid(False)
            plt.tight_layout()

            fname4 = f"fig3a_grid_heatmap_q_T_h{h4}_F{target_F}_L{L4}.png"
            plt.savefig(out_dir / fname4, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname4}")
            plt.close()

    # =========================================================
    # PLOT 5: Discrete Grid Heatmap of S_max over q/N and h (Fixed T and L)
    # =========================================================
    if 'plot5_heatmap_q_h' in cfg:
        p5_cfg = cfg['plot5_heatmap_q_h']
        T5 = p5_cfg['T']
        L5 = p5_cfg['L']

        vis_T5 = round(1.0 - T5, 4)
        print(f"\nGenerating Plot 5 (Grid Heatmap q/N vs h | T={vis_T5}, L={L5})...")
        mask_p5 = np.isclose(df_f['T'], T5) & (df_f['L'] == L5)
        df_p5 = df_f[mask_p5]
        print(f"  -> Rows matching T={T5}, L={L5}: {len(df_p5)}")

        if not df_p5.empty:
            grouped_p5 = df_p5.groupby(['h', 'q_N'])['s_max'].mean().reset_index()
            
            pivot_p5 = grouped_p5.pivot(index='h', columns='q_N', values='s_max')
            pivot_p5 = pivot_p5.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(
                pivot_p5.columns, pivot_p5.index, pivot_p5.values,
                cmap='viridis', shading='auto'
            )

            cbar = plt.colorbar(mesh)
            cbar.set_label(r'$\langle S_{max} \rangle / N$', rotation=270, labelpad=20)
            
            plt.xlabel('q/N')
            plt.ylabel('h (Empty Density)')
            plt.title(f'Phase Diagram (Grid Heatmap)\n($F={target_F}$, $T={vis_T5}$, $L={L5}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)
            plt.grid(False)
            plt.tight_layout()

            fname5 = f"fig3b_grid_heatmap_q_h_T{vis_T5}_F{target_F}_L{L5}.png"
            plt.savefig(out_dir / fname5, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname5}")
            plt.close()

    # =========================================================
    # PLOT 6: Interpolated Heatmap of S_max over q/N and T (Fixed h and L)
    # =========================================================
    if 'plot4_heatmap_q_T' in cfg:
        p6_cfg = cfg['plot4_heatmap_q_T']
        h6 = p6_cfg['h']
        L6 = p6_cfg['L']

        print(f"\nGenerating Plot 6 (Interpolated Heatmap q/N vs T | h={h6}, L={L6})...")
        mask_p6 = np.isclose(df_f['h'], h6) & (df_f['L'] == L6)
        df_p6 = df_f[mask_p6]
        print(f"  -> Rows matching h={h6}, L={L6}: {len(df_p6)}")

        if not df_p6.empty:
            grouped_p6 = df_p6.groupby(['q_N', 'T'])['s_max'].mean().reset_index()
            grouped_p6['T_vis'] = 1.0 - grouped_p6['T']

            plt.figure(figsize=(8, 6))
            contour = plt.tricontourf(
                grouped_p6['q_N'], grouped_p6['T_vis'], grouped_p6['s_max'], 
                levels=15, cmap='viridis'
            )
            plt.scatter(
                grouped_p6['q_N'], grouped_p6['T_vis'], 
                c='black', s=10, alpha=0.3, label='Data Points'
            )

            cbar = plt.colorbar(contour)
            cbar.set_label(r'$\langle S_{max} \rangle / N$', rotation=270, labelpad=20)
            
            plt.xlabel('q/N')
            plt.ylabel('T (Tolerance)')
            plt.title(f'Phase Diagram (Interpolated Heatmap)\n($F={target_F}$, $h={h6}$, $L={L6}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            fname6 = f"fig4a_interpolated_heatmap_q_T_h{h6}_F{target_F}_L{L6}.png"
            plt.savefig(out_dir / fname6, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname6}")
            plt.close()

    # =========================================================
    # PLOT 7: Interpolated Heatmap of S_max over q/N and h (Fixed T and L)
    # =========================================================
    if 'plot5_heatmap_q_h' in cfg:
        p7_cfg = cfg['plot5_heatmap_q_h']
        T7 = p7_cfg['T']
        L7 = p7_cfg['L']

        vis_T7 = round(1.0 - T7, 4)
        print(f"\nGenerating Plot 7 (Interpolated Heatmap q/N vs h | T={vis_T7}, L={L7})...")
        mask_p7 = np.isclose(df_f['T'], T7) & (df_f['L'] == L7)
        df_p7 = df_f[mask_p7]
        print(f"  -> Rows matching T={T7}, L={L7}: {len(df_p7)}")

        if not df_p7.empty:
            grouped_p7 = df_p7.groupby(['q_N', 'h'])['s_max'].mean().reset_index()

            plt.figure(figsize=(8, 6))
            contour = plt.tricontourf(
                grouped_p7['q_N'], grouped_p7['h'], grouped_p7['s_max'], 
                levels=15, cmap='viridis'
            )
            plt.scatter(
                grouped_p7['q_N'], grouped_p7['h'], 
                c='black', s=10, alpha=0.3, label='Data Points'
            )

            cbar = plt.colorbar(contour)
            cbar.set_label(r'$\langle S_{max} \rangle / N$', rotation=270, labelpad=20)
            
            plt.xlabel('q/N')
            plt.ylabel('h (Empty Density)')
            plt.title(f'Phase Diagram (Interpolated Heatmap)\n($F={target_F}$, $T={vis_T7}$, $L={L7}$)', fontsize=15, pad=15)
            plt.xlim(0, 6)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            fname7 = f"fig4b_interpolated_heatmap_q_h_T{vis_T7}_F{target_F}_L{L7}.png"
            plt.savefig(out_dir / fname7, dpi=300, bbox_inches='tight')
            print(f"  -> Saved: {fname7}")
            plt.close()

    # =========================================================
    # PLOT 8: Varying Empty Density h (Fixed T and L)
    # =========================================================
    if 'plot8_density_scaling' in cfg:
        p8_cfg = cfg['plot8_density_scaling']
        T8 = p8_cfg['T']
        L8 = p8_cfg['L']
        hs_p8 = p8_cfg['h_values']

        vis_T8 = round(1.0 - T8, 4)
        print(f"\nGenerating Plot 8 (Density Scaling T={vis_T8}, L={L8})...")
        mask_p8 = np.isclose(df_f['T'], T8) & (df_f['L'] == L8)
        df_p8 = df_f[mask_p8]
        print(f"  -> Rows matching T={T8}, L={L8}: {len(df_p8)}")

        if not df_p8.empty:
            plt.figure(figsize=(7, 5))
            colors_p8 = get_color_generator()
            plotted_lines = 0

            for h_val in hs_p8:
                d = df_p8[np.isclose(df_p8['h'], h_val)]
                if d.empty:
                    print(f"     [Warning] No data for h={h_val}")
                    continue

                # Calculate mean and standard deviation
                grouped = d.groupby('q_N')['s_max'].agg(['mean', 'std']).reset_index().sort_values('q_N')
                grouped['std'] = grouped['std'].fillna(0)

                plt.errorbar(
                    grouped['q_N'], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', linestyle='--', linewidth=1,
                    color=next(colors_p8), markerfacecolor='none', markeredgewidth=1.2,
                    capsize=3, elinewidth=1, capthick=1,
                    label=f'h={h_val}'
                )
                plotted_lines += 1

            if plotted_lines > 0:
                plt.xlabel('q/N')
                plt.ylabel(r'$\langle S_{max} \rangle / N$')
                plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $T={vis_T8}$, $L={L8}$)', fontsize=15, pad=15)
                plt.xlim(0, 6)
                plt.ylim(0, 1.05)
                plt.legend(loc='best', framealpha=1.0, edgecolor='black')
                plt.tight_layout()

                fname8 = f"fig5_density_scaling_T{vis_T8}_L{L8}_h{format_list_for_fname(hs_p8)}.png"
                plt.savefig(out_dir / fname8, dpi=300, bbox_inches='tight')
                print(f"  -> Saved: {fname8}")
            else:
                print("  -> Plot 8 skipped: No lines to plot.")
            plt.close()

    # =========================================================
    # PLOT 9: Varying Tolerance T and Empty Density h (Fixed L)
    # =========================================================
    if 'plot9_tolerance_density_scaling' in cfg:
        p9_cfg = cfg['plot9_tolerance_density_scaling']
        L9 = p9_cfg['L']
        Ts_p9 = p9_cfg['T_values']
        hs_p9 = p9_cfg['h_values']

        print(f"\nGenerating Plot 9 (Tolerance & Density Scaling L={L9})...")
        mask_p9 = (df_f['L'] == L9)
        df_p9 = df_f[mask_p9]
        print(f"  -> Rows matching L={L9}: {len(df_p9)}")

        if not df_p9.empty:
            plt.figure(figsize=(8, 6))
            colors_p9 = get_color_generator()
            plotted_lines = 0

            # Generate parameter combinations
            for t_val, h_val in itertools.product(Ts_p9, hs_p9):
                d = df_p9[np.isclose(df_p9['T'], t_val) & np.isclose(df_p9['h'], h_val)]
                if d.empty:
                    print(f"     [Warning] No data for T={round(1.0 - t_val, 4)}, h={h_val}")
                    continue

                # Calculate mean and standard deviation
                grouped = d.groupby('q_N')['s_max'].agg(['mean', 'std']).reset_index().sort_values('q_N')
                grouped['std'] = grouped['std'].fillna(0)

                vis_T = round(1.0 - t_val, 4)
                plt.errorbar(
                    grouped['q_N'], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', linestyle='--', linewidth=1,
                    color=next(colors_p9), markerfacecolor='none', markeredgewidth=1.2,
                    capsize=3, elinewidth=1, capthick=1,
                    label=f'T={vis_T} h={h_val}'
                )
                plotted_lines += 1

            if plotted_lines > 0:
                plt.xlabel('q/N')
                plt.ylabel(r'$\langle S_{max} \rangle / N$')
                plt.title(f'Phase Transition on Schelling-Axelrod Model\n($F={target_F}$, $L={L9}$)', fontsize=15, pad=15)
                plt.xlim(0, 6)
                plt.ylim(0, 1.05)
                # Position legend outside or adaptively to manage size with multiple combinations
                plt.legend(loc='best', framealpha=1.0, edgecolor='black')
                plt.tight_layout()

                fname9 = f"fig6_tolerance_density_scaling_L{L9}_T{format_list_for_fname_inverted(Ts_p9)}_h{format_list_for_fname(hs_p9)}.png"
                plt.savefig(out_dir / fname9, dpi=300, bbox_inches='tight')
                print(f"  -> Saved: {fname9}")
            else:
                print("  -> Plot 9 skipped: No lines to plot.")
            plt.close()

    print(f"\nDone! Output directory: {out_dir}/")

if __name__ == "__main__":
    main()