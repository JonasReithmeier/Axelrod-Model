import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from src.model import AxelrodModel_regularLattice
from src.utils import largest_cluster_normalized

def run_single_realization(params):
    """
    Worker function: Executes one realization until it reaches a frozen state.
    """
    # 1. Initialize Model
    model = AxelrodModel_regularLattice(
        width=params['width'],
        height=params['width'], # Square lattice
        F=params['F'],
        q=params['q'],
        seed=params['seed']
    )

    # 2. Simulation Loop with Frozen Logic
    # Heuristic: Check for absolute freezing after N*100 failed steps
    threshold = model.N * 100 
    max_steps = params['max_steps']
    
    steps_taken = 0
    for i in range(max_steps):
        model.step()
        steps_taken += 1

        # Check for stability
        if model.updates_since_last_change >= threshold:
            if model.is_totally_frozen():
                break
            else:
                updates_since_last_change = 0 # False alarm, reset and continue

    # Final check after loop breaks
    is_frozen = model.is_totally_frozen()

    # 3. Data Collection
    s_max = largest_cluster_normalized(model, N=model.N)
    
    return {
        'width': params['width'],
        'N': model.N,
        'q': params['q'],
        'F': params['F'],
        'seed': params['seed'],
        's_max': s_max,
        'steps_to_freeze': steps_taken,
        'is_frozen': is_frozen
    }

def main():
    # --- Preparation ---
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config['experiment']
    sim_defaults = config['simulation_defaults']
    
    # Path Setup: Create 'data' folder inside the main project directory
    data_path = Path("data/task1")
    data_path.mkdir(exist_ok=True)

    # --- Parameter Sweep Setup ---
    master_rng = np.random.default_rng(exp_cfg['master_seed'])
    
    # Generate all combinations of sweep parameters
    sweep_params = exp_cfg['sweep']
    # product() creates the Cartesian product of all provided lists
    combinations = list(product(sweep_params['q'], sweep_params['width']))
    
    tasks = []
    for q, width in combinations:
        # Generate M independent child seeds for this specific (q, width) pair
        seeds = master_rng.integers(0, 2**32, size=exp_cfg['M_realizations'])
        for m in range(exp_cfg['M_realizations']):
            tasks.append({
                'q': q,
                'width': width,
                'F': sim_defaults['features'],
                'max_steps': sim_defaults['max_steps'],
                'seed': int(seeds[m])
            })

    print(f"Starting experiment: {len(tasks)} total realizations...")
    print(f"Using {os.cpu_count()} CPU cores.")

    # --- Parallel Execution ---
    results = []
    with ProcessPoolExecutor() as executor:
        # We use executor.map to run the worker function on all tasks
        results = list(executor.map(run_single_realization, tasks))

    # --- Data Saving ---
    df = pd.DataFrame(results)

    # Identify non-frozen runs
    non_frozen = df[df['is_frozen'] == False]
    
    if not non_frozen.empty:
        print("\n" + "!"*30)
        print(f"WARNING: {len(non_frozen)} realizations failed to freeze!")
        print("Problematic parameter sets (Top 5):")
        # This groups by params and counts failures, so you see where the issues are
        summary = non_frozen.groupby(['width', 'q']).size().reset_index(name='failures')
        print(summary.head(5))
        print("!"*30 + "\n")
    else:
        print("\nAll realizations froze successfully.")
    
    # Save as Parquet (Scientific standard: compressed, fast, preserves types)
    output_file = data_path / "raw_simulation_data.parquet"
    df.to_parquet(output_file, index=False)
    
    # Also save a CSV version for easy manual inspection
    df.to_csv(data_path / "raw_simulation_data.csv", index=False)

    print(f"\nSweep Complete.")
    print(f"Data saved to: {output_file}")
    print(f"Total rows collected: {len(df)}")

if __name__ == "__main__":
    main()