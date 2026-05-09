import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_axelrod_data():
    # --- 1. Load Data ---
    data_path = Path("data/task1/axelrod_master_results.parquet")
    if not data_path.exists():
        print(f"Error: Database not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    
    # Filter: Usually, phase transition plots only consider reached equilibrium
    # If you want to include non-frozen ones, comment out the line below.
    #df = df[df['is_frozen'] == True]

    if df.empty:
        print("No frozen data available to plot.")
        return

    # --- 2. Setup Output Folder ---
    plot_dir = Path("plots/task1")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Scientific Styling ---
    plt.style.use('seaborn-v0_8-paper') # Professional clean style
    sns.set_context("paper", font_scale=1.5)
    
    # --- 4. Plotting Logic ---
    unique_Fs = sorted(df['F'].unique())
    metrics = [('s_max', 'Largest Cluster Size (Normalized)'), 
               ('s_mean', 'Average Cluster Size (Normalized)')]

    M_min = -1 # realizations (smallest group)

    #for F in unique_Fs:
    for F in [3]:
        df_f = df[df['F'] == F]
        
        for metric_key, metric_name in metrics:
            plt.figure(figsize=(10, 7))
            
            # Group by width (N) and q to get statistics
            # We use width for the legend label
            widths = sorted(df_f['width'].unique())
            
            #for width in widths:
            for width in [20,40,50]:
            #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == width]

                M = len(df_w)
                if M_min < 0:
                    M_min = M
                else:
                    if M < M_min:
                        M_min = M
                
                # Aggregate data for each q
                stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
                
                # Plot line with error bars (standard deviation)
                plt.errorbar(
                    stats['q'], stats['mean'], yerr=stats['std'],
                    label=f"N = {width}²",
                    marker='o', markersize=4, capsize=3, elinewidth=1, linestyle='-'
                )

            # Labels and Aesthetics
            plt.title(f"Axelrod Model, regular lattice, M=500 realizations | F = {F} features", pad=20)
            plt.xlabel("Number of Traits per Feature (q)")
            plt.ylabel(metric_name)
            plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(title="Grid Size", frameon=True)
            
            # Save Logic
            file_name =  "task1_graph_recreation.png"  #f"axelrod_F{F}_{metric_key}.png"
            save_path = plot_dir / file_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to free memory

    print(f"Update complete. Plots saved to: {plot_dir}")

if __name__ == "__main__":
    plot_axelrod_data()