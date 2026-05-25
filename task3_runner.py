import os
import yaml
import pandas as pd
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

def run_single_realization(params):
    try:
        cp_dir = Path("data/schelling/checkpoints")
        cp_dir.mkdir(parents=True, exist_ok=True)
        
        cp_file = cp_dir / f"cp_L{params['L']}_q{params['q']}_F{params['F']}_h{params['h']}_T{params['T']}_m{params['m']}.pkl"

        model = AxelrodSchellingModel(
            L=params['L'], F=params['F'], q=params['q'], h=params['h'], 
            T=params['T'], m=params['m'], master_seed=params['master_seed']
        )

        steps_already_done = 0
        if cp_file.exists():
            try:
                with open(cp_file, 'rb') as f:
                    model.load_checkpoint_data(pickle.load(f))
                steps_already_done = model.total_mcs
            except Exception as e:
                print(f"Warning: Checkpoint corrupted, restarting m={params['m']}")
                model.initialize_new_simulation()
        else:
            model.initialize_new_simulation()

        additional_mcs = params['max_mcs'] - steps_already_done
        if additional_mcs <= 0:
            return None 

        is_constant, steps_to_const, avg_mob = model.run(additional_mcs, transient_mcs=params['transient_mcs'])  
        
        s_max, s_mean = model.get_metrics()

        result = {
            'L': params['L'], 'q': params['q'], 'F': params['F'], 'h': params['h'], 'T': params['T'],
            'm': params['m'], 's_max': s_max, 's_mean': s_mean,
            'steps_to_const': steps_to_const,
            'is_constant': is_constant,
            'avg_mobility': avg_mob
        }

        if is_constant:
            if cp_file.exists():
                try: cp_file.unlink()
                except: pass
        else:
            temp_file = cp_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(model.get_checkpoint_data(), f)
            temp_file.replace(cp_file) 

        return result

    except Exception as e:
        return {"ERROR": traceback.format_exc()}


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
            final_df = final_df.drop_duplicates(subset=['L', 'q', 'F', 'h', 'T', 'm'], keep='last')
        else:
            final_df = journal_df
        
        final_df.to_parquet(master_file, index=False)
        final_df.to_csv(master_file.with_suffix('.csv'), index=False)
        
        with open(journal_file, 'w') as f:
            f.truncate(0) 
            
    except Exception as e:
        print(f"Error during journal ingestion: {e}")


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
            (row.L, row.q, row.F, row.h, row.T, row.m): (row.is_constant, row.steps_to_const) 
            for row in old_df.itertuples(index=False)
        }
        print(f"Database loaded: {len(existing_data_map)} realizations existing.")

    sweep_params = exp_cfg['sweep']
    # Note: Make sure 'L' is now used in your YAML config instead of 'width'
    combinations = list(product(sweep_params['F'], sweep_params['q'], sweep_params['L'], sweep_params['h'], sweep_params['T']))
    tasks = []
    
    for f_val, q_val, L_val, h_val, T_val in combinations:
        for m in range(1, exp_cfg['M_realizations'] + 1):
            
            is_const, steps_const = existing_data_map.get((L_val, q_val, f_val, h_val, T_val, m), (None, 0))

            if is_const is None or (is_const == False and exp_cfg['max_mcs'] > steps_const):
                tasks.append({
                    'L': L_val, 'q': q_val, 'F': f_val, 'h': h_val, 'T': T_val, 'm': m,
                    'master_seed': exp_cfg['master_seed'],
                    'max_mcs': exp_cfg['max_mcs'], 
                    'transient_mcs': exp_cfg.get('transient_mcs', None)
                })

    if not tasks:
        print("All experiments complete or constant. Nothing to do.")
        return

    random.shuffle(tasks)
    print(f"Starting {len(tasks)} tasks. Live journaling active...")
    print(f"Graceful Stop: Create a file named 'STOP' to exit early.")

    stop_file = Path("STOP")
    fieldnames = ['L', 'q', 'F', 'h', 'T', 'm', 's_max', 's_mean', 'steps_to_const', 'is_constant', 'avg_mobility']
    
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_task = {executor.submit(run_single_realization, t): t for t in tasks}
            
            with open(journal_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if journal_file.stat().st_size == 0:
                    writer.writeheader()

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
                        
                    writer.writerow(result)
                    f.flush()
                    
                    if count % 100 == 0:
                        print(f"Completed {count} / {len(tasks)} tasks...")

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down pool... Journaling safe.")
    finally:
        if journal_file.exists() and journal_file.stat().st_size > 0:
            try:
                journal_df = pd.read_csv(journal_file)
                non_constant = journal_df[journal_df['is_constant'] == False]
                if not non_constant.empty:
                    print(f"\nWARNING: {len(non_constant)} runs in this session failed to converge to constant.")
            except: pass
            
            ingest_journal_to_master(master_file, journal_file)

    duration = time.perf_counter() - start_time
    print("\n" + "="*40)
    print(f"COMPLETED in {str(timedelta(seconds=round(duration)))}")
    print(f"Total entries in Master DB: {len(pd.read_parquet(master_file))}")
    print("="*40)

if __name__ == "__main__":
    main()