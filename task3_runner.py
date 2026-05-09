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
from src.model_as import AxelrodSchellingModel

# --- 1. The Result Listener (Background Process) ---
def result_listener(queue, journal_path):
    # FIXED: Added 'h' and 'T' to the fieldnames
    fieldnames = ['width', 'N', 'q', 'F', 'h', 'T', 'seed', 's_max', 's_mean', 'steps_to_freeze', 'is_frozen']
    
    with open(journal_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if journal_path.stat().st_size == 0:
            writer.writeheader()
        
        while True:
            result = queue.get()
            if result == "STOP":
                break
            writer.writerow(result)
            f.flush()

# --- Global Worker Variable ---
WORKER_QUEUE = None

def init_worker(q):
    global WORKER_QUEUE
    WORKER_QUEUE = q

# --- 2. The Worker Function ---
def run_single_realization(params):
    global WORKER_QUEUE
    queue = WORKER_QUEUE
    cp_dir = Path("data/schelling/checkpoints")
    cp_dir.mkdir(parents=True, exist_ok=True)
    
    # FIXED: Added h and T to the checkpoint filename to prevent overwrites
    cp_file = cp_dir / f"cp_w{params['width']}_q{params['q']}_F{params['F']}_h{params['h']}_T{params['T']}_s{params['seed']}.pkl"

    model = AxelrodSchellingModel(
        width=params['width'], height=params['width'], 
        F=params['F'], q=params['q'], h=params['h'], T=params['T'], seed=params['seed']
    )

    steps_already_done = 0
    if cp_file.exists():
        try:
            with open(cp_file, 'rb') as f:
                model.load_checkpoint_data(pickle.load(f))
            # FIXED: Corrected attribute name
            steps_already_done = model.total_steps
        except:
            model.initialize_new_simulation()
    else:
        model.initialize_new_simulation()

    additional_steps = params['max_steps'] - steps_already_done
    
    # FIXED: Corrected tuple unpacking
    _, is_frozen = model.run(additional_steps)
    
    # FIXED: Corrected attribute name
    total_steps = model.total_steps

    s_max, s_mean = model.get_metrics()

    # FIXED: Added 'h' and 'T' to the result dictionary
    result = {
        'width': params['width'], 'N': model.N_cells, 'q': params['q'], 'F': params['F'],
        'h': params['h'], 'T': params['T'],
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
            final_df = pd.concat([old_df, journal_df], ignore_index=True).reset_index(drop=True)
            # FIXED: Added 'h' and 'T' to subset so they don't overwrite each other!
            final_df = final_df.drop_duplicates(subset=['width', 'q', 'F', 'h', 'T', 'seed'], keep='last')
        else:
            final_df = journal_df
        
        final_df.to_parquet(master_file, index=False)
        final_df.to_csv(master_file.with_suffix('.csv'), index=False)
        
        with open(journal_file, 'w') as f:
            f.truncate(0) 
            
    except Exception as e:
        print(f"Error during journal ingestion: {e}")

# --- 4. Main Orchestrator ---
def main():
    start_time = time.perf_counter()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exp_cfg = config['as_experiment']
    
    data_path = Path("data/schelling")
    data_path.mkdir(parents=True, exist_ok=True)
    master_file = data_path / "schelling_master_results.parquet"
    journal_file = data_path / "journal_temp_schelling.csv"
    
    if not journal_file.exists():
        journal_file.touch()

    ingest_journal_to_master(master_file, journal_file)

    existing_data_map = {}
    if master_file.exists():
        old_df = pd.read_parquet(master_file)
        existing_data_map = {
            (row.width, row.q, row.F, row.h, row.T, row.seed): (row.is_frozen, row.steps_to_freeze) 
            for row in old_df.itertuples(index=False)
        }
        print(f"Database loaded: {len(existing_data_map)} realizations existing.")

    sweep_params = exp_cfg['sweep']
    combinations = list(product(sweep_params['F'], sweep_params['q'], sweep_params['width'], sweep_params['h'], sweep_params['T']))
    tasks = []
    
    for f_val, q_val, w_val, h_val, T_val in combinations:
        for m in range(exp_cfg['M_realizations']):
            seed_context = f"{exp_cfg['master_seed']}_{f_val}_{q_val}_{w_val}_{h_val}_{T_val}_{m}"
            seed = zlib.adler32(seed_context.encode())
            
            is_frozen, prev_steps = existing_data_map.get((w_val, q_val, f_val, h_val, T_val, seed), (None, 0))
            
            if is_frozen is None or (is_frozen == False and exp_cfg['max_steps'] > prev_steps):
                tasks.append({
                    'q': q_val, 'width': w_val, 'F': f_val, 'h': h_val, 'T': T_val,
                    'max_steps': exp_cfg['max_steps'], 
                    'seed': seed
                })

    if not tasks:
        print("All experiments complete or frozen. Nothing to do.")
        return

    random.shuffle(tasks)

    manager = mp.Manager()
    queue = manager.Queue()
    listener = mp.Process(target=result_listener, args=(queue, journal_file))
    listener.start()

    print(f"Starting {len(tasks)} tasks. Journaling active...")
    print(f"Graceful Stop: Create a file named 'STOP' to exit early.")

    stop_file = Path("STOP")
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(queue,)) as executor:
            futures = []
            for t in tasks:
                if stop_file.exists():
                    print("\nSTOP file detected. No more tasks will be submitted.")
                    stop_file.unlink()
                    break
                futures.append(executor.submit(run_single_realization, t))
            
            for future in futures:
                future.result()

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down pool... Journaling safe.")
    finally:
        queue.put("STOP")
        listener.join()
        
        if journal_file.exists() and journal_file.stat().st_size > 0:
            journal_df = pd.read_csv(journal_file)
            non_frozen = journal_df[journal_df['is_frozen'] == False]
            if not non_frozen.empty:
                print(f"\nWARNING: {len(non_frozen)} runs in this session failed to freeze.")
                # FIXED: Added h and T to the warning output formatter
                print(non_frozen.groupby(['width', 'F', 'q', 'h', 'T']).size().reset_index(name='failures').head())
            
            ingest_journal_to_master(master_file, journal_file)

    duration = time.perf_counter() - start_time
    print("\n" + "="*40)
    print(f"COMPLETED in {str(timedelta(seconds=round(duration)))}")
    print(f"Total entries in Master DB: {len(pd.read_parquet(master_file))}")
    print("="*40)

if __name__ == "__main__":
    main()