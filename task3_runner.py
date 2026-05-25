import os
import yaml
import pandas as pd
import numpy as np
import zlib
import hashlib
import time
import pickle
import csv
import random
import traceback
from datetime import timedelta
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.model_as import AxelrodSchellingModel

# --- 1. The Worker Function ---
# (Removed the global queue. The worker now just returns a dictionary!)
def run_single_realization(params):
    try:
        cp_dir = Path("data/schelling/checkpoints")
        cp_dir.mkdir(parents=True, exist_ok=True)
        
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
                steps_already_done = model.total_steps
            except Exception as e:
                print(f"Warning: Checkpoint corrupted, restarting seed {params['seed']}")
                model.initialize_new_simulation()
        else:
            model.initialize_new_simulation()

        additional_steps = params['max_steps'] - steps_already_done
        
        if additional_steps <= 0:
            return None # Already finished, safe abort

        _, is_frozen, avg_mob = model.run(additional_steps)
        
        total_steps = model.total_steps
        s_max, s_mean = model.get_metrics()

        result = {
            'width': params['width'], 'N': model.N_cells, 'q': params['q'], 'F': params['F'],
            'h': params['h'], 'T': params['T'],
            'seed': params['seed'], 's_max': s_max, 's_mean': s_mean,
            'steps_to_freeze': total_steps,
            'is_frozen': is_frozen,
            'avg_mobility': avg_mob
        }

        # ATOMIC SAVING: Fixes the checkpoint corruption bottleneck
        if is_frozen:
            if cp_file.exists():
                try: cp_file.unlink()
                except: pass
        else:
            temp_file = cp_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(model.get_checkpoint_data(), f)
            temp_file.replace(cp_file) # Atomic rename (cannot be corrupted by crashes)

        return result

    except Exception as e:
        # If the worker crashes, return the error so the main thread doesn't hang!
        return {"ERROR": traceback.format_exc()}


# --- 2. Helper for DB Ingestion ---
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
            final_df = final_df.drop_duplicates(subset=['width', 'q', 'F', 'h', 'T', 'seed'], keep='last')
        else:
            final_df = journal_df
        
        final_df.to_parquet(master_file, index=False)
        final_df.to_csv(master_file.with_suffix('.csv'), index=False)
        
        with open(journal_file, 'w') as f:
            f.truncate(0) 
            
    except Exception as e:
        print(f"Error during journal ingestion: {e}")


# --- 3. Main Orchestrator ---
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
            
            # --- THE DUAL-HASH BRIDGE ---
            seed_context = f"{exp_cfg['master_seed']}_{f_val}_{q_val}_{w_val}_{h_val}_{T_val}_{m}"
            
            legacy_seed = zlib.adler32(seed_context.encode())
            new_md5_seed = int(hashlib.md5(seed_context.encode()).hexdigest()[:16], 16)
            
            is_frozen_leg, steps_leg = existing_data_map.get((w_val, q_val, f_val, h_val, T_val, legacy_seed), (None, 0))
            is_frozen_new, steps_new = existing_data_map.get((w_val, q_val, f_val, h_val, T_val, new_md5_seed), (None, 0))
            
            if is_frozen_leg is not None:
                active_is_frozen, active_steps, active_seed = is_frozen_leg, steps_leg, legacy_seed
            elif is_frozen_new is not None:
                active_is_frozen, active_steps, active_seed = is_frozen_new, steps_new, new_md5_seed
            else:
                active_is_frozen, active_steps, active_seed = None, 0, new_md5_seed

            # Add task if not finished
            if active_is_frozen is None or (active_is_frozen == False and exp_cfg['max_steps'] > active_steps):
                tasks.append({
                    'q': q_val, 'width': w_val, 'F': f_val, 'h': h_val, 'T': T_val,
                    'max_steps': exp_cfg['max_steps'], 
                    'seed': active_seed
                })

    if not tasks:
        print("All experiments complete or frozen. Nothing to do.")
        return

    random.shuffle(tasks)
    print(f"Starting {len(tasks)} tasks. Live journaling active...")
    print(f"Graceful Stop: Create a file named 'STOP' to exit early.")

    stop_file = Path("STOP")
    fieldnames = ['width', 'N', 'q', 'F', 'h', 'T', 'seed', 's_max', 's_mean', 'steps_to_freeze', 'is_frozen', 'avg_mobility']
    
    # --- LIVE JOURNALING WITHOUT THE BOTTLENECK QUEUE ---
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all
            future_to_task = {executor.submit(run_single_realization, t): t for t in tasks}
            
            # Open CSV in main thread
            with open(journal_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if journal_file.stat().st_size == 0:
                    writer.writeheader()

                # Process exactly as they finish
                for count, future in enumerate(as_completed(future_to_task), 1):
                    
                    if stop_file.exists():
                        print("\nSTOP file detected. Cancelling pending tasks...")
                        for f_cancel in future_to_task: f_cancel.cancel()
                        stop_file.unlink()
                        break

                    result = future.result()
                    
                    if result is None:
                        continue
                        
                    if "ERROR" in result:
                        print(f"\n[FATAL WORKER ERROR]:\n{result['ERROR']}")
                        continue
                        
                    # Write to live journal and force flush to disk
                    writer.writerow(result)
                    f.flush()
                    
                    if count % 100 == 0:
                        print(f"Completed {count} / {len(tasks)} tasks...")

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down pool... Journaling safe.")
    finally:
        # Check for non-frozen warning
        if journal_file.exists() and journal_file.stat().st_size > 0:
            try:
                journal_df = pd.read_csv(journal_file)
                non_frozen = journal_df[journal_df['is_frozen'] == False]
                if not non_frozen.empty:
                    print(f"\nWARNING: {len(non_frozen)} runs in this session failed to freeze.")
            except: pass
            
            # Ingest final batch
            ingest_journal_to_master(master_file, journal_file)

    duration = time.perf_counter() - start_time
    print("\n" + "="*40)
    print(f"COMPLETED in {str(timedelta(seconds=round(duration)))}")
    print(f"Total entries in Master DB: {len(pd.read_parquet(master_file))}")
    print("="*40)

if __name__ == "__main__":
    main()