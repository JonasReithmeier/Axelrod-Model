"""
runner.py
---------
Single-trial execution function. Designed to be:
  - called directly for testing
  - wrapped as a Ray remote function in sweep.py
  - checkpointed to disk so sweeps can resume after interruption

Checkpoints are written as .npz files to checkpoint_dir/{trial_id}.npz
Results are returned as plain dicts (JSON-serializable, Ray-friendly).
"""

import os
import time
import logging
import numpy as np

from .novelModel import AxelrodDevSmallWorld

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir, trial_id):
    return os.path.join(checkpoint_dir, f"{trial_id}.npz")


def save_checkpoint(model, trial_id, checkpoint_dir, params):
    os.makedirs(checkpoint_dir, exist_ok=True)
    cp = model.get_checkpoint_data()
    path = _checkpoint_path(checkpoint_dir, trial_id)
    np.savez_compressed(
        path,
        grid=cp["grid"],
        dev=cp["dev"],
        padded_edges=cp["padded_edges"],
        adj_matrix=cp["adj_matrix"],
        updates=np.array([cp["updates"]], dtype=np.int64),
        total_steps=np.array([cp["total_steps"]], dtype=np.int64),
        capacity_exceeded=np.array([cp["capacity_exceeded"]], dtype=np.int64),
        # RNG state is a dict — serialize its arrays manually
        rng_state_state=cp["rng_state"]["state"]["state"],
        rng_state_inc=cp["rng_state"]["state"]["inc"],
        # Store params so we can reconstruct the model on resume
        **{f"param_{k}": np.array([v]) for k, v in params.items()
           if isinstance(v, (int, float))},
    )
    logger.debug(f"Checkpoint saved: {path}")


# Updated load_checkpoint in novelModel_runner.py

def load_checkpoint(model, trial_id, checkpoint_dir):
    path = _checkpoint_path(checkpoint_dir, trial_id)
    if not os.path.exists(path):
        return False
    
    try:
        # Load the file safely
        data = np.load(path, allow_pickle=True)

        grid_data = data["grid"]
        padded_edges_data = data["padded_edges"]
        dev_data = data["dev"]
        adj_matrix_data = data["adj_matrix"]

        # Validate shapes
        if grid_data.shape[0] != model.N or grid_data.shape[1] != model.F:
            logger.warning(f"Shape mismatch 'grid' in checkpoint {path}. Starting fresh.")
            return False

        expected_edges_size = model.N * model.max_degree
        if padded_edges_data.shape[0] != expected_edges_size:
            logger.warning(f"Shape mismatch 'padded_edges' in checkpoint {path}. Starting fresh.")
            return False

    except Exception as e:
        # Handle BadZipFile, OSError, KeyErrors, or other loading corruptions gracefully
        logger.warning(
            f"[{trial_id}] Checkpoint {path} is corrupted or incomplete ({type(e).__name__}: {e}). "
            f"Starting this trial fresh."
        )
        # Attempt to delete the corrupted file so it does not cause future issues
        try:
            os.remove(path)
        except OSError:
            pass
        return False

    # Reconstruct state if loading succeeded
    rng_state = {
        "bit_generator": "PCG64",
        "state": {
            "state": int(data["rng_state_state"]),
            "inc": int(data["rng_state_inc"]),
        },
        "has_uint32": 0,
        "uinteger": 0,
    }

    cp = {
        "grid": grid_data.copy(),
        "dev": dev_data.copy(),
        "padded_edges": padded_edges_data.copy(),
        "adj_matrix": adj_matrix_data.astype(np.bool_),
        "updates": int(data["updates"][0]),
        "total_steps": int(data["total_steps"][0]),
        "capacity_exceeded": int(data["capacity_exceeded"][0]) if "capacity_exceeded" in data else 0,
        "rng_state": rng_state,
    }
    
    model.load_checkpoint_data(cp)
    logger.debug(f"Checkpoint loaded: {path}")
    return True

# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(params, checkpoint_dir="checkpoints", steps_per_chunk=None):
    """
    Run a single trial defined by params dict.

    params keys (all required unless marked optional):
        N, k, p, F, q
        weight_mode, alpha
        dis_threshold
        dev_mode, dev_param (optional, default None)
        seed
        max_steps       : hard budget (elementary steps)
        trial_id        : unique string for checkpointing

    Returns dict with params + results, suitable for Optuna / logging.
    """
    trial_id = params.get("trial_id", "trial_0")
    max_steps = int(params["max_steps"])

    model = AxelrodDevSmallWorld(
        N=int(params["N"]),
        k=int(params["k"]),
        p=float(params["p"]),
        F=int(params["F"]),
        q=int(params["q"]),
        weight_mode=int(params["weight_mode"]),
        alpha=float(params["alpha"]),
        dis_threshold=float(params["dis_threshold"]),
        dev_mode=int(params.get("dev_mode", 0)),
        dev_param=params.get("dev_param", None),
        seed=int(params["seed"]),
        m=int(params.get("m", 0)),
    )

    # Try to resume from checkpoint
    resumed = load_checkpoint(model, trial_id, checkpoint_dir)
    if not resumed:
        model.initialize_new_simulation()
        logger.info(f"[{trial_id}] Starting fresh.")
    else:
        logger.info(
            f"[{trial_id}] Resumed from step {model.total_steps}."
            f"Prior capacity exceeded count: {model.capacity_exceeded}"
            )

    if steps_per_chunk is None:
        steps_per_chunk = model.N * 1000  # 1000 sweeps per checkpoint interval

    t_start = time.perf_counter()
    converged = False
    final_rewire_rate = -1.0

    while model.total_steps < max_steps:
        remaining = max_steps - model.total_steps
        to_run = min(steps_per_chunk, remaining)

        converged, _, final_rewire_rate = model.run(to_run)

        save_checkpoint(model, trial_id, checkpoint_dir, params)
        logger.info(
            f"[{trial_id}] step={model.total_steps}/{max_steps} "
            f"converged={converged} capacity_blocked={model.capacity_exceeded}"
        )

        if converged:
            break

    elapsed = time.perf_counter() - t_start

    # Collect metrics (L, C are expensive — only called once at end)
    metrics = model.get_all_metrics()
    metrics["final_rewire_rate"] = (
        final_rewire_rate if final_rewire_rate >= 0 else metrics.get("final_rewire_rate", -1.0)
    )

    result = {
        **params,
        "converged": converged,
        "total_steps": model.total_steps,
        "elapsed_s": elapsed,
        "capacity_exceeded": model.capacity_exceeded,
        **metrics,
    }

    logger.info(
        f"[{trial_id}] Done. converged={converged} metrics={metrics}"
        f"capacity_blocked={model.capacity_exceeded} metrics={metrics}"
        )
    return result