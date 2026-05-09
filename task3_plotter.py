import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

def plot_as_phase_transition(F_val=5, q_val=10, data_path="data/task3/axelrod_as_master_results.parquet"):
    """
    Plots the Axelrod-Schelling phase transition.
    X-axis: Tolerance (T)
    Y-axis: Largest Cultural Cluster (s_max)
    Lines: Different densities of empty sites (h)
    """
    # 1. Path Management
    # Update this path if you saved your Task 3 parquet file somewhere else!
    file_path = Path(data_path)
    if not file_path.exists():
        print(f"Error: Could not find data file at {file_path}")
        print("Please check the 'data_path' variable in the script.")
        return

    # 2. Load and Filter Data
    df = pd.read_parquet(file_path)

    # Filter for the specific F and q used in the config
    df_filtered = df[(df['F'] == F_val) & (df['q'] == q_val)]
    
    if df_filtered.empty:
        print(f"Error: No data found for F={F_val} and q={q_val} in the database.")
        return

    # 3. Aggregate Data: Mean and StdDev for s_max grouped by h and T
    grouped = df_filtered.groupby(['h', 'T'])['s_max'].agg(['mean', 'std']).reset_index()

    # 4. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    
    # Get unique 'h' values and sort them
    h_values = sorted(grouped['h'].unique())
    
    # Use a colormap for different empty site densities
    colors = cm.plasma(np.linspace(0.1, 0.9, len(h_values)))

    # 5. Plot lines with error bars for each 'h'
    for i, h in enumerate(h_values):
        subset = grouped[grouped['h'] == h].sort_values(by='T')
        
        T_vals = subset['T']
        mean_smax = subset['mean']
        std_smax = subset['std'].fillna(0) # Fill NaN with 0 if only 1 realization exists
        
        ax.errorbar(T_vals, mean_smax, yerr=std_smax, fmt='-o', 
                    markersize=6, linewidth=2, capsize=4, capthick=1.5, 
                    color=colors[i], label=f'h = {h}')

    # 6. Formatting
    ax.set_xlabel('Tolerance ($T$)', fontsize=14)
    ax.set_ylabel(r'Largest Cultural Cluster $\langle S_{max} \rangle / N_{active}$', fontsize=14)
    
    # Dynamic Title
    ax.set_title(f'Axelrod-Schelling Model: Impact of Mobility\n(Traits $q={q_val}$, Features $F={F_val}$)', 
                 fontsize=15, pad=15)
    
    ax.set_ylim(-0.05, 1.05)
    # Set X-axis to go from 0 (Strict/Intolerant) to 1.0 (Pure Axelrod/Tolerant)
    ax.set_xlim(0.0, 1.05) 
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Legend setup
    ax.legend(title="Empty Density ($h$)", fontsize=11, title_fontsize=12, loc='lower right')

    # Add descriptive text to explain the physics on the plot
    ax.text(0.05, 0.85, 'Segregated Phase\n(Mobility Fragments Culture)', 
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.text(0.70, 0.85, 'Globalized Phase\n(Stable Consensus)', 
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # 7. Output Management
    out_dir = Path("plots/task3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"schelling_transition_F{F_val}_q{q_val}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig) 
    
    print(f"Success! Plot saved to: {out_file}")

if __name__ == "__main__":
    # Ensure the path matches wherever your runner_as.py saves the parquet file!
    plot_as_phase_transition(
        F_val=5, 
        q_val=10, 
        data_path="data/schelling/schelling_master_results.parquet"
    )