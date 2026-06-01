import os
import yaml
import pandas as pd
import numpy as np
import zlib
import time
import pickle
import csv
import random
import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from src.model import AxelrodModel

# --- 1. The Result Listener (Background Process) ---
def result_listener(queue, journal_path):
    """
    Background process that writes results to a CSV journal in real-time.
    """
    fieldnames = ['width', 'N', 'q', 'F', 'iterationNumber', 'seed', 's_max', 's_mean', 'steps_to_freeze', 'is_frozen']
    
    # Open in append mode
    with open(journal_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write header only if file is new/empty
        if journal_path.stat().st_size == 0:
            writer.writeheader()
        
        while True:
            result = queue.get()
            if result == "STOP":
                break
            writer.writerow(result)
            f.flush() # Force write to physical disk immediately

# --- Global Worker Variable ---
WORKER_QUEUE = None

def init_worker(q):
    """
    Initializes the queue connection ONCE per CPU core when it boots up.
    Prevents Socket/File Descriptor exhaustion.
    """
    global WORKER_QUEUE
    WORKER_QUEUE = q

# --- 2. The Worker Function ---
def run_single_realization(params):
    global WORKER_QUEUE
    queue = WORKER_QUEUE  # Use the globally initialized queue
    cp_dir = Path("data/task1/checkpoints")
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_file = cp_dir / f"cp_w{params['width']}_q{params['q']}_F{params['F']}_iterationNumber{params['iterationNumber']}_s{params['seed']}.pkl"

    # Initialize Fast Model
    model = AxelrodModel(
        width=params['width'], height=params['width'], 
        F=params['F'], q=params['q'], iterationNumber=params['iterationNumber'], seed=params['seed']
    )

    steps_already_done = 0
    if cp_file.exists():
        try:
            with open(cp_file, 'rb') as f:
                model.load_checkpoint_data(pickle.load(f))
            steps_already_done = model.total_steps_done
        except:
            model.initialize_new_simulation()
    else:
        model.initialize_new_simulation()

    additional_steps = params['max_steps'] - steps_already_done
    
    # -----------------------------------------------------
    # THE MASSIVE SPEEDUP: 
    # One Python call executes up to 100 million steps in C
    # -----------------------------------------------------
    is_frozen = model.run(additional_steps)
    total_steps = model.total_steps_done

    # Fast metrics 
    s_max, s_mean = model.get_metrics()

    result = {
        'width': params['width'], 'N': model.N, 'q': params['q'], 'F': params['F'],'iterationNumber': params['iterationNumber'],
        'seed': params['seed'], 's_max': s_max, 's_mean': s_mean,
        'steps_to_freeze': total_steps,
        'is_frozen': is_frozen
    }

    queue.put(result)

    if is_frozen:
        if cp_file.exists(): cp_file.unlink()
    else:
        with open(cp_file, 'wb') as f:
            pickle.dump(model.get_checkpoint_data(), f)

    return True

# --- 3. Helper for DB Ingestion ---
def ingest_journal_to_master(master_file, journal_file):
    """Merges the temporary CSV journal into the Master Parquet file."""
    # Check if file exists and has content (not just a header)
    if not journal_file.exists() or journal_file.stat().st_size < 10:
        return
    
    try:
        journal_df = pd.read_csv(journal_file)
        if journal_df.empty:
            journal_file.unlink()
            return
            
        print(f"Merging journal into database...")
        
        if master_file.exists():
            old_df = pd.read_parquet(master_file)
            # To avoid the FutureWarning, ensure both have the same dtypes
            # or filter out empty DFs
            final_df = pd.concat([old_df, journal_df], ignore_index=True).reset_index(drop=True)
            # Keep the latest run (highest max_steps) for each combination
            final_df = final_df.drop_duplicates(subset=['width', 'q', 'F', 'iterationNumber'], keep='last')
        else:
            final_df = journal_df
        
        final_df.to_parquet(master_file, index=False)
        final_df.to_csv(master_file.with_suffix('.csv'), index=False)
        
        # Clear the journal file instead of deleting to keep the handle safe for the listener
        with open(journal_file, 'w') as f:
            f.truncate(0) 
            
    except Exception as e:
        print(f"Error during journal ingestion: {e}")

# --- 4. Main Orchestrator ---
def main():
    start_time = time.perf_counter()
    
    # --- Config and Paths ---
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exp_cfg = config['experiment']
    
    data_path = Path("data/task1")
    data_path.mkdir(parents=True, exist_ok=True)
    master_file = data_path / (config['grid_visualization']['database_name'] + '.parquet')
    journal_file = data_path / "journal_temp.csv"
    # Ensure journal exists for the listener
    if not journal_file.exists():
        journal_file.touch()

    # --- RECOVERY: If previous run crashed, ingest journal now ---
    ingest_journal_to_master(master_file, journal_file)

    # --- Load Database for filtering ---
    existing_data_map = {}
    if master_file.exists():
        old_df = pd.read_parquet(master_file)
        existing_data_map = {
            (row.width, row.q, row.F, row.iterationNumber): (row.is_frozen, row.steps_to_freeze) 
            for row in old_df.itertuples(index=False)
        }
        print(f"Database loaded: {len(existing_data_map)} realizations existing.")

    # --- Build Task List ---
    sweep_params = exp_cfg['sweep']
    combinations = list(product(sweep_params['F'], sweep_params['q'], sweep_params['width']))
    tasks = []
    for f_val, q_val, w_val in combinations:
        for m in range(exp_cfg['M_realizations']):
            seed_context = f"{exp_cfg['master_seed']}_{f_val}_{q_val}_{w_val}_{m}"
            seed = zlib.adler32(seed_context.encode())
            
            is_frozen, prev_steps = existing_data_map.get((w_val, q_val, f_val, m), (None, 0))
            
            if is_frozen is None or (is_frozen == False and exp_cfg['max_steps'] > prev_steps):
                tasks.append({
                    'q': q_val, 'width': w_val, 'F': f_val, 
                    'max_steps': exp_cfg['max_steps'], 
                    'iterationNumber': m,
                    #'prev_steps': prev_steps,    prev steps are now included in the checkpoint file for more safety: in case the db is behind, more calculations may be done, but seed and steps are guaranteed in sync
                    'seed': seed
                })

    if not tasks:
        print("All experiments complete or frozen. Nothing to do.")
        return

    # shuffling to let cpus work on mix of slow and fast freezing tasks to avoid the listener being the bottleneck
    random.shuffle(tasks)

    # --- Setup Multiprocessing Listener ---
    manager = mp.Manager()
    queue = manager.Queue()
    listener = mp.Process(target=result_listener, args=(queue, journal_file))
    listener.start()

    print(f"Starting {len(tasks)} tasks. Journaling active...")
    print(f"Graceful Stop: Create a file named 'STOP' to exit early.")

    # --- Parallel Execution ---
    stop_file = Path("STOP")
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(queue,)) as executor:
            
            # Use submit for better control if we want to stop early
            futures = []
            for t in tasks:
                if stop_file.exists():
                    print("\nSTOP file detected. No more tasks will be submitted.")
                    stop_file.unlink()
                    break
                futures.append(executor.submit(run_single_realization, t))
            
            # Wait for submitted tasks to complete
            for future in futures:
                future.result()

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down pool... Journaling safe.")
    finally:
        # Poison pill for Listener
        queue.put("STOP")
        listener.join()
        
        # FINAL REPORT & MERGE
        if journal_file.exists() and journal_file.stat().st_size > 0:
            journal_df = pd.read_csv(journal_file)
            non_frozen = journal_df[journal_df['is_frozen'] == False]
            if not non_frozen.empty:
                print(f"\nWARNING: {len(non_frozen)} runs in this session failed to freeze.")
                print(non_frozen.groupby(['width', 'F', 'q']).size().reset_index(name='failures').head())
            
            ingest_journal_to_master(master_file, journal_file)

    # Final Timing
    duration = time.perf_counter() - start_time
    print("\n" + "="*40)
    print(f"COMPLETED in {str(timedelta(seconds=round(duration)))}")
    print(f"Total entries in Master DB: {len(pd.read_parquet(master_file))}")
    print("="*40)

if __name__ == "__main__":
    main()