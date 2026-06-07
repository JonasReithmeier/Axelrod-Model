import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import yaml
import math
import seaborn as sns

def recreate_watts_strogatz_plot(N_par,k_par):

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exp_cfg = config['sw_experiment']

    dataDir_path = Path("data/small_world")

    data_path = dataDir_path / (exp_cfg['database_name'] + '.parquet')
    file_path = Path(data_path)
    if not file_path.exists():
        print(f"Error: Could not find data file at {file_path}")
        return

    # 1. Load the data
    df = pd.read_parquet(file_path)

    # 2. Filter for Watts-Strogatz exact parameters (N=1000, k=10)
    # The Axelrod variables (F, q, max_steps) don't affect graph topology, 
    # so grouping by 'p' will safely average L and C across all those realizations.
    df_ws = df[(df['N'] == N_par) & (df['k'] == k_par)]
    
    if df_ws.empty:
        print(f"Error: No data found for N={N_par} and k={k_par} in the database.")
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
    #ax.text(0.002, 0.8, '$C(p) / C(0)$', fontsize=16)
    #ax.text(0.0003, 0.25, '$L(p) / L(0)$', fontsize=16)
    ax.legend(title="Grid Coefficients", fontsize=11, title_fontsize=12, loc='upper right')

    # Final touches
    #plt.title("Recreation of Watts-Strogatz (1998) Figure 2", fontsize=14, pad=15)
    plt.tight_layout()
    
    # Save and Show
    plot_dir = Path("reportPlots/task2")
    plot_dir.mkdir(parents=True, exist_ok=True)

    file_name =  f"watts_strogatz_fig_N{N_par}_k{k_par}.png"
    save_path = plot_dir / file_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def init_db_connection():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exp_cfg = config['sw_experiment']

    dataDir_path = Path("data/small_world")

    data_path = dataDir_path / (exp_cfg['database_name'] + '.parquet')

    file_path = Path(data_path)
    if not file_path.exists():
        print(f"Error: Could not find data file at {file_path}")
        return

    # 1. Load data
    df = pd.read_parquet(file_path)
    return df

def plot_sw_phase_transition(N_val,F_val, k_val, target_p_values,name):
    """
    Plots the Axelrod model phase transition (s_max vs q) for specific rewiring 
    probabilities (p), fixing F and k. Uses error bars for standard deviation.
    """
    df = init_db_connection()

    # 2. Filter for the specific F and k
    df_filtered = df[(df['N'] == N_val) & (df['F'] == F_val) & (df['k'] == k_val)]
    
    # 3. Filter OUT old runs: keep only the targeted p values
    df_filtered = df_filtered[df_filtered['p'].isin(target_p_values)]
    
    if df_filtered.empty:
        print(f"Error: No data found for F={F_val}, k={k_val}, and specified p values.")
        return
        
    #N_val = df_filtered['N'].iloc[0]

    # 4. Aggregate data: Mean and StdDev for s_max grouped by p and q
    grouped = df_filtered.groupby(['p', 'q'])['s_max'].agg(['mean', 'std']).reset_index()

    # 5. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    
    # Get the unique 'p' values that actually exist in the filtered data
    p_values = sorted(grouped['p'].unique())
    
    # Setup colormap
    colors = cm.viridis(np.linspace(0.1, 0.9, len(p_values)))

    # 6. Plot using error bars
    for i, p in enumerate(p_values):
        subset = grouped[grouped['p'] == p].sort_values(by='q')

        q_vals = subset['q']
        number_realisations = len(q_vals)
        mean_smax = subset['mean'] 
        std_smax = subset['std'].fillna(-1) # Fill NaN with 0 if only 1 realization exists
         
        # plt.errorbar replaces plot() and fill_between()
        # fmt='-o' creates a line with circular markers
        # capsize defines the width of the error bar caps
        ax.errorbar(q_vals, mean_smax, yerr=std_smax/math.sqrt(number_realisations), fmt='-o', 
                    markersize=5, linewidth=2, capsize=4, capthick=1.5, 
                    color=colors[i], label=f'p = {p}')

    # 7. Formatting
    ax.set_xlabel('Number of traits ($q$)', fontsize=14)
    ax.set_ylabel(r'Largest Cultural Cluster $\langle S_{max} \rangle / N$', fontsize=14)
    ax.set_title(f'Axelrod Phase Transition on Small-World Network\n($N={N_val}$, $F={F_val}$, $k={k_val}$)', fontsize=15, pad=15)
    
    # Ensure axes limits make sense
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Legend setup
    ax.legend(title="Rewiring Prob ($p$)", fontsize=11, title_fontsize=12, loc='upper right')

    # 8. Output Management
    out_dir = Path("reportPlots/task2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"phase_transition_N{N_val}_F{F_val}_k{k_val}_s{name}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig) 
    
    print(f"Success! Plot saved to: {out_file}")
    
def std_error(value):
    value = value/math.sqrt(number_realisations)
    return value;
def generateGraphics(df_filtered,metric, N_val, F_val, k_val):
    grouped = df_filtered.groupby(['p', 'q'])[metric].agg(['mean', 'std']).reset_index()
    global number_realisations 
    number_realisations = len(grouped[grouped['p'] == 0])
    grouped['std'] = grouped['std'].apply(std_error)
    heatmap_data = grouped.pivot(index = 'q', columns='p', values='mean')
    error_data = grouped.pivot(index = 'q', columns='p', values='std')

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data)  # defaults include a colorbar
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title(f"{metric} Heatmap N = {N_val}, F = {F_val}, k = {k_val}")
    plt.tight_layout()

    out_dir = Path("reportPlots/task2") # TODO add report dir to config
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"heatmap_{metric}_N{N_val}_F{F_val}_k{k_val}.png"

    plt.savefig(out_file)

    plt.figure(figsize=(12, 6))
    sns.heatmap(error_data)  # defaults include a colorbar
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title(f"{metric} Errormap N = {N_val}, F = {F_val}, k = {k_val}")
    plt.tight_layout()
    out_file = out_dir / f"errorMap_{metric}_N{N_val}_F{F_val}_k{k_val}.png"

    plt.savefig(out_file)

def heatmap(N_val,F_val,k_val): #0 has to be part of probability dataset, TODO change it to number of occurencies of first element
    sns.set_theme(style="white")
    #flights = sns.load_dataset("flights")
    #print(flights.head())
    df = init_db_connection()
    df_filtered = df[(df['N'] == N_val) & (df['F'] == F_val) & (df['k'] == k_val)]
    generateGraphics(df_filtered,'s_max',N_val,F_val,k_val)
    generateGraphics(df_filtered,'s_mean',N_val,F_val,k_val)
    
def plot2DCuts(N_val,F_val,k_val):
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.0, 0.001, 0.01, 0.1, 1.0],name="default")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.0, 0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08,0.09],name="0.0")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.1, 0.11, 0.12, 0.13, 0.14,0.15,0.16,0.17,0.18,0.19],name="0.1")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.2, 0.21, 0.22, 0.23, 0.24,0.25,0.26,0.27,0.28,0.29],name="0.2")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.3, 0.31, 0.32, 0.33, 0.34,0.35,0.36,0.37,0.38,0.39],name="0.3")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.4, 0.41, 0.42, 0.43, 0.44,0.45,0.46,0.47,0.48,0.49],name="0.4")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.5, 0.51, 0.52, 0.53, 0.54,0.55,0.56,0.57,0.58,0.59],name="0.5")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.6, 0.61, 0.62, 0.63, 0.64,0.65,0.66,0.67,0.68,0.69],name="0.6")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.7, 0.71, 0.72, 0.73, 0.74,0.75,0.76,0.77,0.78,0.79],name="0.7")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.8, 0.81, 0.82, 0.83, 0.84,0.85,0.86,0.87,0.88,0.89],name="0.8")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.9, 0.91, 0.92, 0.93, 0.94,0.95,0.96,0.97,0.98,0.99,1],name="0.9")
    plot_sw_phase_transition(N_val = N_val,F_val=F_val, k_val=k_val, target_p_values=[0.0, 0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08,0.09,
                                                                                      0.1, 0.11, 0.12, 0.13, 0.14,0.15,0.16,0.17,0.18,0.19,
                                                                                      0.2, 0.21, 0.22, 0.23, 0.24,0.25,0.26,0.27,0.28,0.29,
                                                                                      0.3, 0.31, 0.32, 0.33, 0.34,0.35,0.36,0.37,0.38,0.39,
                                                                                      0.4, 0.41, 0.42, 0.43, 0.44,0.45,0.46,0.47,0.48,0.49,
                                                                                      0.5, 0.51, 0.52, 0.53, 0.54,0.55,0.56,0.57,0.58,0.59,
                                                                                      0.6, 0.61, 0.62, 0.63, 0.64,0.65,0.66,0.67,0.68,0.69,
                                                                                      0.7, 0.71, 0.72, 0.73, 0.74,0.75,0.76,0.77,0.78,0.79,
                                                                                      0.8, 0.81, 0.82, 0.83, 0.84,0.85,0.86,0.87,0.88,0.89,
                                                                                      0.9,0.91, 0.92, 0.93, 0.94,0.95,0.96,0.97,0.98,0.99,1],name="all")
    

if __name__ == "__main__":
    recreate_watts_strogatz_plot(400,4)
    #recreate_watts_strogatz_plot(900,4)
    plot2DCuts(400,3,4)
    heatmap(400,3,4)