import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def recreate_watts_strogatz_plot(data_path="data/small_world/axelrod_sw_master_results.parquet"):
    file_path = Path(data_path)
    if not file_path.exists():
        print(f"Error: Could not find data file at {file_path}")
        return

    # 1. Load the data
    df = pd.read_parquet(file_path)

    # 2. Filter for Watts-Strogatz exact parameters (N=1000, k=10)
    # The Axelrod variables (F, q, max_steps) don't affect graph topology, 
    # so grouping by 'p' will safely average L and C across all those realizations.
    df_ws = df[(df['N'] == 1000) & (df['k'] == 10)]
    
    if df_ws.empty:
        print("Error: No data found for N=1000 and k=10 in the database.")
        return

    # 3. Calculate average L and C for each rewiring probability p
    grouped = df_ws.groupby('p')[['L', 'C']].mean().reset_index()
    grouped = grouped.sort_values(by='p')

    # 4. Extract baseline values at p = 0.0 (Regular Ring Lattice)
    if 0.0 not in grouped['p'].values:
        print("Error: Baseline p=0.0 not found in data. Cannot normalize!")
        return
        
    L_0 = grouped.loc[grouped['p'] == 0.0, 'L'].values[0]
    C_0 = grouped.loc[grouped['p'] == 0.0, 'C'].values[0]
    
    print(f"Baseline Values (p=0.0): L(0) = {L_0:.2f}, C(0) = {C_0:.4f}")

    # 5. Normalize metrics
    grouped['L_norm'] = grouped['L'] / L_0
    grouped['C_norm'] = grouped['C'] / C_0

    # 6. Remove p=0.0 for plotting (log(0) is undefined)
    plot_data = grouped[grouped['p'] > 0.0]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Plot C(p) / C(0) as open squares
    ax.plot(plot_data['p'], plot_data['C_norm'], marker='s', markersize=7, 
            linestyle='', color='black', markerfacecolor='white', markeredgewidth=1.2, 
            label='C(p) / C(0)')

    # Plot L(p) / L(0) as filled circles
    ax.plot(plot_data['p'], plot_data['L_norm'], marker='o', markersize=7, 
            linestyle='', color='black', markerfacecolor='black', 
            label='L(p) / L(0)')

    # Formatting to match the paper
    ax.set_xscale('log')
    ax.set_xlim(0.00008, 1.2) # Give a slight padding around 0.0001 and 1.0
    ax.set_ylim(-0.05, 1.05)

    # Labels and Ticks
    ax.set_xlabel('$p$', fontsize=16, fontstyle='italic')
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
    ax.tick_params(axis='both', which='minor', direction='in', length=3)

    # Add text directly onto the plot (like the original figure)
    # We place these dynamically based on the log scale
    ax.text(0.002, 0.8, '$C(p) / C(0)$', fontsize=16)
    ax.text(0.0003, 0.25, '$L(p) / L(0)$', fontsize=16)

    # Final touches
    #plt.title("Recreation of Watts-Strogatz (1998) Figure 2", fontsize=14, pad=15)
    plt.tight_layout()
    
    # Save and Show
    plot_dir = Path("plots/task2")
    plot_dir.mkdir(parents=True, exist_ok=True)

    file_name =  "watts_strogatz_fig2.png"
    save_path = plot_dir / file_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 


if __name__ == "__main__":
    recreate_watts_strogatz_plot()