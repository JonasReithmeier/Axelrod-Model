import os
import yaml
import pandas as pd
import numpy as np
import zlib
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from src.model import AxelrodModel_regularLattice
from src.utils import largest_cluster_normalized, average_cluster_normalized

def run_single_realization(params):
    """
    Worker function: Executes one realization until it reaches a frozen state.
    """
    model = AxelrodModel_regularLattice(
        width=params['width'],
        height=params['width'], 
        F=params['F'],
        q=params['q'],
        seed=params['seed']
    )

    threshold = model.N * 100 
    max_steps = params['max_steps']
    steps_taken = 0

    while model.updates_since_last_change < threshold:
        model.step()
        steps_taken += 1

        if model.updates_since_last_change >= threshold:
            if model.is_totally_frozen():
                break
            else:
                # IMPORTANT: Reset the model's internal counter on false alarm
                model.updates_since_last_change = 0 

        if steps_taken >= max_steps:
            break

    is_frozen = model.is_totally_frozen()
    s_max = largest_cluster_normalized(model, N=model.N)
    s_mean = average_cluster_normalized(model, N=model.N)
    
    return {
        'width': params['width'],
        'N': model.N,
        'q': params['q'],
        'F': params['F'],
        'seed': params['seed'],
        's_max': s_max,
        's_mean': s_mean,
        'steps_to_freeze': steps_taken,
        'is_frozen': is_frozen
    }

def main():
    # --- 1. Preparation ---
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config['experiment']
    
    # Path Setup: parents=True allows creating nested folders like data/task1/raw
    data_path = Path("data/task1")
    data_path.mkdir(parents=True, exist_ok=True)
    master_file = data_path / "axelrod_master_results.parquet"

    # --- 2. Load Existing Data for Filtering ---
    existing_combinations = set()
    if master_file.exists():
        old_df = pd.read_parquet(master_file)
        # We store (width, q, seed) as a tuple in a set for O(1) lookup speed
        existing_combinations = set(zip(old_df['width'], old_df['q'], old_df['F'], old_df['seed']))
        print(f"Loaded existing database with {len(existing_combinations)} realizations.")

    # --- 3. Build Task List (Filtering Logic) ---
    sweep_params = exp_cfg['sweep']
    combinations = list(product(sweep_params['F'], sweep_params['q'], sweep_params['width']))
    
    tasks = []
    for f_val, q_val, w_val in combinations:
        for m in range(exp_cfg['M_realizations']):
            # Create a unique string identifier for this specific realization.
            # Including 'm' (realization index) ensures that the M different runs 
            # for the same parameters get different, but consistent, seeds.
            seed_context = f"{exp_cfg['master_seed']}_{f_val}_{q_val}_{w_val}_{m}"
            
            # Use Adler-32 checksum to generate a deterministic 32-bit integer seed.
            # Unlike Python's built-in hash(), zlib.adler32 is NOT randomized 
            # across different Python sessions, which is crucial for the Upsert logic.
            seed = zlib.adler32(seed_context.encode())
            
            # Upsert Logic: Only add to the task list if this specific 
            # parameter set + seed combination is not already in the master database.
            if (w_val, q_val, f_val, seed) not in existing_combinations:
                tasks.append({
                    'q': q_val,
                    'width': w_val,
                    'F': f_val, 
                    'max_steps': exp_cfg['max_steps'],
                    'seed': seed
                })

    if not tasks:
        print("Done! All requested realizations already exist in the database.")
        return

    print(f"Starting experiment: Running {len(tasks)} NEW realizations...")
    print(f"Parallelizing across {os.cpu_count()} CPU cores.")

    # --- 4. Parallel Execution ---
    with ProcessPoolExecutor() as executor:
        new_results_list = list(executor.map(run_single_realization, tasks))

    # --- 5. Merge and Save ---
    new_df = pd.DataFrame(new_results_list)
    
    # Report on the quality of the NEW runs
    non_frozen = new_df[new_df['is_frozen'] == False]
    if not non_frozen.empty:
        print(f"\nWARNING: {len(non_frozen)} NEW realizations failed to freeze.")
        print(non_frozen.groupby(['width', 'F', 'q']).size().reset_index(name='failures').head())
    else:
        print("\nAll new realizations froze successfully.")

    # Combine with old data if it exists
    if master_file.exists():
        old_df = pd.read_parquet(master_file)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        # Safety check: remove duplicates just in case
        final_df = final_df.drop_duplicates(subset=['width', 'q', 'F', 'seed'])
    else:
        final_df = new_df

    # Save to Master File (Parquet and CSV)
    final_df.to_parquet(master_file, index=False)
    final_df.to_csv(master_file.with_suffix('.csv'), index=False)

    print(f"\nDatabase Updated.")
    print(f"Total realizations in master file: {len(final_df)}")
    print(f"Master file located at: {master_file}")

if __name__ == "__main__":
    main()