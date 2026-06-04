"""
sweep.py
--------
Parallel parameter sweep using Ray (execution) + Optuna (search strategy).

Usage:
    python -m model_dev_sw.sweep --config config.yaml --output results.csv

Optuna study is persisted to a local SQLite DB so runs survive interruption.
Ray distributes trials across CPUs (or a cluster if Ray is initialised externally).

Sweep modes (set in config.yaml under sweep.mode):
    grid    : exhaustive grid over all listed values (small sweeps)
    random  : random sampling (medium sweeps, no Optuna needed)
    tpe     : Optuna TPE sampler (large sweeps, optimises a target metric)
    cmaes   : Optuna CMA-ES sampler (continuous parameter optimisation)
"""

import os
import csv
import time
import logging
import argparse
import itertools
import hashlib

import numpy as np
import yaml
import ray
import optuna


#from .novelModel_runner import run_trial

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------

@ray.remote
def ray_run_trial(params, checkpoint_dir, steps_per_chunk):
    import sys, os, logging
    # Make sure the repo root is on the path for every worker process
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO)
    from novelModel.novelModel_runner import run_trial as _run_trial
    return _run_trial(params, checkpoint_dir=checkpoint_dir, steps_per_chunk=steps_per_chunk)

# ---------------------------------------------------------------------------
# Config loading & param expansion
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_trial_id(params):
    """Deterministic short ID from param dict (for checkpoint filenames)."""
    key = "_".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "trial_id")
    return hashlib.md5(key.encode()).hexdigest()[:12]


def grid_params(sweep_cfg, fixed_cfg):
    """Expand sweep.params lists into all combinations."""
    sweep_keys = list(sweep_cfg["params"].keys())
    sweep_vals = [sweep_cfg["params"][k] for k in sweep_keys]
    for combo in itertools.product(*sweep_vals):
        p = dict(zip(sweep_keys, combo))
        p.update(fixed_cfg)
        yield p


# ---------------------------------------------------------------------------
# CSV result writer (thread-safe via append mode)
# ---------------------------------------------------------------------------

def append_result(result, output_path):
    # Ensure the containing directory (e.g., 'novelModel/') exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------

def run_grid_sweep(config, output_path, checkpoint_dir, steps_per_chunk):
    fixed = config.get("fixed", {})
    sweep_cfg = config["sweep"]
    n_realizations = sweep_cfg.get("n_realizations", 1)

    all_params = list(grid_params(sweep_cfg, fixed))
    logger.info(
        f"Grid sweep: {len(all_params)} settings, {n_realizations} realizations each "
        f"({len(all_params) * n_realizations} total trials)"
    )

    futures = []
    for p in all_params:
        for m in range(n_realizations):
            p_real = p.copy()
            p_real["m"] = m
            p_real["trial_id"] = make_trial_id(p_real)
            futures.append(ray_run_trial.remote(p_real, checkpoint_dir, steps_per_chunk))

    # --- Out-of-Order Recovery Loop ---
    unresolved = futures.copy()
    completed_count = 0
    total_count = len(futures)

    while unresolved:
        # Get whichever task finishes first
        resolved, unresolved = ray.wait(unresolved, num_returns=1)
        
        # This is guaranteed to be finished, so ray.get() is instantaneous
        result = ray.get(resolved[0])
        append_result(result, output_path)
        
        completed_count += 1
        logger.info(
            f"[{completed_count}/{total_count}] Saved: {result['trial_id']} "
            f"(m={result.get('m', 0)})"
        )

def run_optuna_sweep(config, output_path, checkpoint_dir, steps_per_chunk, n_trials, mode):
    fixed = config.get("fixed", {})
    sweep_cfg = config["sweep"]
    param_ranges = sweep_cfg["params"]
    target_metric = sweep_cfg.get("target_metric", "s_max")
    direction = sweep_cfg.get("target_direction", "maximize")
    n_realizations = sweep_cfg.get("n_realizations", 1)

    db_path = output_path.replace(".csv", ".db")
    storage = f"sqlite:///{db_path}"

    sampler = (
        optuna.samplers.CmaEsSampler()
        if mode == "cmaes"
        else optuna.samplers.TPESampler()
    )

    study = optuna.create_study(
        study_name=config.get("study_name", "dev_sw_sweep"),
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
    )

    n_parallel = config.get("n_parallel", max(1, os.cpu_count() - 1))

    completed = 0
    while completed < n_trials:
        # Determine the number of unique settings we can evaluate in parallel based on CPU limits
        batch_size = max(1, n_parallel // n_realizations)
        batch_size = min(batch_size, n_trials - completed)
        
        trials = [study.ask() for _ in range(batch_size)]

        futures_map = {}  # Map each trial to its list of parallel realizations
        for trial in trials:
            p_base = {}
            for k, v in param_ranges.items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], float):
                    p_base[k] = trial.suggest_float(k, v[0], v[1])
                elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
                    p_base[k] = trial.suggest_int(k, v[0], v[1])
                elif isinstance(v, list):
                    p_base[k] = trial.suggest_categorical(k, v)
                else:
                    p_base[k] = v
            p_base.update(fixed)

            futures_map[trial] = []
            for m in range(n_realizations):
                p_real = p_base.copy()
                p_real["m"] = m
                p_real["trial_id"] = make_trial_id(p_real)
                future = ray_run_trial.remote(p_real, checkpoint_dir, steps_per_chunk)
                futures_map[trial].append(future)

        # Retrieve results and report averages to Optuna
        for trial in trials:
            realization_results = ray.get(futures_map[trial])
            
            # Write all individual realizations to the CSV for analysis
            for res in realization_results:
                append_result(res, output_path)

            # Average the target metric across all realizations for Optuna
            values = [res.get(target_metric, float("nan")) for res in realization_results]
            avg_value = float(np.nanmean(values))
            
            study.tell(trial, avg_value)
            completed += 1
            logger.info(
                f"Trial {completed}/{n_trials}: avg_{target_metric}={avg_value:.4f} "
                f"across {n_realizations} realizations. params={trial.params}"
            )

    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best value: {study.best_value}")


def run_random_sweep(config, output_path, checkpoint_dir, steps_per_chunk, n_trials):
    fixed = config.get("fixed", {})
    sweep_cfg = config["sweep"]
    param_ranges = sweep_cfg["params"]
    n_realizations = sweep_cfg.get("n_realizations", 1)
    rng = np.random.default_rng(config.get("sweep_seed", 0))

    futures = []
    for _ in range(n_trials):
        p_base = {}
        for k, v in param_ranges.items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], float):
                p_base[k] = float(rng.uniform(v[0], v[1]))
            elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
                p_base[k] = int(rng.integers(v[0], v[1] + 1))
            elif isinstance(v, list):
                p_base[k] = v[int(rng.integers(0, len(v)))]
            else:
                p_base[k] = v
        p_base.update(fixed)

        for m in range(n_realizations):
            p_real = p_base.copy()
            p_real["m"] = m
            p_real["trial_id"] = make_trial_id(p_real)
            futures.append(ray_run_trial.remote(p_real, checkpoint_dir, steps_per_chunk))

    # --- Out-of-Order Recovery Loop ---
    unresolved = futures.copy()
    completed_count = 0
    total_count = len(futures)

    while unresolved:
        resolved, unresolved = ray.wait(unresolved, num_returns=1)
        result = ray.get(resolved[0])
        append_result(result, output_path)
        
        completed_count += 1
        logger.info(
            f"[{completed_count}/{total_count}] Saved Random Trial: {result['trial_id']} "
            f"(m={result.get('m', 0)})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Dev-SW parameter sweep")
    parser.add_argument("--config", default="novelModel/config_novelModel.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default="novelModel/data/results_no_move_if_omega_is_0.csv", help="Output CSV path")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of trials (random/tpe/cmaes modes)")
    args = parser.parse_args()

    config = load_config(args.config)
    sweep_mode = config.get("sweep", {}).get("mode", "grid")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    steps_per_chunk = config.get("steps_per_chunk", None)
    n_parallel = config.get("n_parallel", max(1, os.cpu_count() - 1))

    # Initialise Ray (no-op if already initialised by external cluster setup)
    if not ray.is_initialized():
        ray.init(num_cpus=n_parallel, ignore_reinit_error=True)
        logger.info(f"Ray initialised with {n_parallel} CPUs")

    t0 = time.perf_counter()

    if sweep_mode == "grid":
        run_grid_sweep(config, args.output, checkpoint_dir, steps_per_chunk)
    elif sweep_mode == "random":
        run_random_sweep(config, args.output, checkpoint_dir, steps_per_chunk, args.n_trials)
    elif sweep_mode in ("tpe", "cmaes"):
        run_optuna_sweep(
            config, args.output, checkpoint_dir, steps_per_chunk,
            args.n_trials, sweep_mode
        )
    else:
        raise ValueError(f"Unknown sweep mode: {sweep_mode}")

    logger.info(f"Sweep complete in {time.perf_counter() - t0:.1f}s → {args.output}")
    ray.shutdown()


if __name__ == "__main__":
    main()