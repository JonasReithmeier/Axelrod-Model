import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from src.model_as import AxelrodSchellingModel
from src.core_as import run_steps_as, calculate_mobility_as

DB_PATH = Path("data/schelling/mobility_trajectories.parquet")
PLOT_DIR = Path("plots/task3/mobility")

def calc_and_save():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    cfg = config['as_mobility_experiment']
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    sweep = cfg['sweep']
    combinations = list(product(sweep['F'], sweep['q'], sweep['width'], sweep['h'], sweep['T']))
    
    all_data = []
    
    for f_val, q_val, w_val, h_val, T_val in combinations:
        print(f"Running trajectory for L={w_val}, q={q_val}, F={f_val}, h={h_val}, T={T_val}...")
        
        model = AxelrodSchellingModel(
            width=w_val, height=w_val, F=f_val, q=q_val, 
            h=h_val, T=T_val, seed=cfg['seed']
        )
        model.initialize_new_simulation()
        
        N = model.N_cells
        max_steps = cfg['max_steps']
        q_N = q_val / N
        
        # Pitch 3: Data Thinning. We want exactly 1000 data points to plot.
        data_points = 1000
        steps_per_sample = max(N, max_steps // data_points)
        
        current_step = 0
        huge_threshold = np.iinfo(np.int64).max # Disable early stopping
        
        # Record initial state
        m = calculate_mobility_as(model.grid, w_val, w_val, f_val, T_val, model.num_empty)
        all_data.append({'h': h_val, 'F': f_val, 'L': w_val, 'T': T_val, 'q_N': q_N, 'step': current_step, 'm': m})
        
        while current_step < max_steps:
            steps_done, u, _ = run_steps_as(
                model.grid, model.W, model.H, model.F, model.empty_locs, model.num_empty, 
                model.T, steps_per_sample, model.updates_since_last_change, huge_threshold, model.rng
            )
            current_step += steps_done
            model.updates_since_last_change = u
            
            m = calculate_mobility_as(model.grid, w_val, w_val, f_val, T_val, model.num_empty)
            all_data.append({'h': h_val, 'F': f_val, 'L': w_val, 'T': T_val, 'q_N': q_N, 'step': current_step, 'm': m})

    # Save to database
    df_new = pd.DataFrame(all_data)
    if DB_PATH.exists():
        df_old = pd.read_parquet(DB_PATH)
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['h', 'F', 'L', 'T', 'q_N', 'step'], keep='last')
    else:
        df_final = df_new
        
    df_final.to_parquet(DB_PATH, index=False)
    print(f"Data saved to {DB_PATH}")

def plot():
    if not DB_PATH.exists():
        print("No data found to plot!")
        return
        
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DB_PATH)
    
    # Styling to match the uploaded image
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 11,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'
    })
    
    # 1. Create Individual Plots
    group_cols = ['h', 'F', 'L', 'T']
    groups = df.groupby(group_cols)
    
    markers = ['o', 's', 'D', '^', 'v', '>', '<']
    
    for name, group in groups:
        h_val, f_val, L_val, T_val = name
        
        plt.figure(figsize=(7, 5))
        
        q_N_values = sorted(group['q_N'].unique())
        for i, q_n in enumerate(q_N_values):
            d = group[group['q_N'] == q_n].sort_values('step')
            
            # The reference image uses lines with markers spaced out
            plt.plot(d['step'], d['m'], 
                     marker=markers[i % len(markers)], markevery=20, # Mark every 20th point
                     linestyle='-', linewidth=1.5,
                     label=f'q/N={q_n}')
                     
        plt.xlabel('t')
        plt.ylabel('m')
        plt.ylim(0, 1.05)
        # Format x-axis to scientific notation (e.g., 1x10^7)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        plt.legend(loc='center right', framealpha=1.0, edgecolor='black')
        plt.tight_layout()
        
        filename = f"mob_h{h_val}_F{f_val}_L{L_val}_T{T_val}.png"
        plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")

    # 2. Master Plot (All combinations)
    plt.figure(figsize=(10, 6))
    for name, group in groups:
        h_val, f_val, L_val, T_val = name
        q_N_values = sorted(group['q_N'].unique())
        
        for q_n in q_N_values:
            d = group[group['q_N'] == q_n].sort_values('step')
            # Use semi-transparent lines without markers for the spaghetti plot
            plt.plot(d['step'], d['m'], linewidth=1, alpha=0.6,
                     label=f'h{h_val} L{L_val} T{T_val} q/N={q_n}')
                     
    plt.xlabel('t')
    plt.ylabel('m')
    plt.ylim(0, 1.05)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Legend outside the plot to avoid covering data
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mob_master_all.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved mob_master_all.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        calc_and_save()
        plot()