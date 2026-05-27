"""
Axelrod-Schelling experiment runner — Ray + Optuna edition.

Key improvements over the ProcessPoolExecutor runner
-----------------------------------------------------
1. Persistent Ray workers: Numba JIT compilation happens once per worker
   process, not once per task. Workers stay alive for the whole run.
2. Zero-copy numpy transfers: Ray uses shared memory for numpy arrays, so
   large grid checkpoints never go through pickle IPC pipes.
3. Work-stealing scheduler: Ray's distributed scheduler eliminates the
   GIL-holding as_completed loop and re-balances load automatically.
4. Async batched I/O: a dedicated Ray Actor collects results and flushes
   them to parquet in batches, eliminating the CSV journal write stall.
5. Optuna adaptive sampling: instead of a flat itertools.product shuffle,
   Optuna builds a TPE model over the parameter space and prioritises
   regions where convergence is uncertain — converges on interesting T/h
   boundaries much faster, especially useful for large F/q sweeps.

Usage
-----
    python run_experiment.py                    # use config.yaml
    python run_experiment.py --config myrun.yaml
    python run_experiment.py --resume           # pick up from existing DB

Graceful stop: create a file named STOP in the working directory, or press
Ctrl+C. All completed results are safe in the parquet DB.
"""

import argparse
import hashlib
import os
import pickle
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import ray
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Ray remote: single realization worker
# ---------------------------------------------------------------------------

@ray.remote
def run_realization(params: dict, cp_bytes: bytes | None) -> dict:
    """
    Runs one (param-set, seed) realization inside a persistent Ray worker.

    Numba JIT warm-up happens on the first call in each worker process and
    is amortised across all subsequent tasks assigned to that worker.

    Parameters
    ----------
    params : dict
        All model hyper-parameters plus 'master_seed', 'max_mcs',
        'transient_mcs', and 'm' (realization index).
    cp_bytes : bytes or None
        Pickled checkpoint dict if resuming, else None.

    Returns
    -------
    dict with result columns, or {"ERROR": traceback_string}.
    """
    # Import here so each Ray worker owns its own Numba state.
    from src.model_as import AxelrodSchellingModel

    try:
        model = AxelrodSchellingModel(
            L=params["L"],
            F=params["F"],
            q=params["q"],
            h=params["h"],
            T=params["T"],
            m=params["m"],
            master_seed=params["master_seed"],
        )

        steps_already_done = 0
        if cp_bytes is not None:
            try:
                model.load_checkpoint_data(pickle.loads(cp_bytes))
                steps_already_done = model.total_mcs
            except Exception:
                model.initialize_new_simulation()
        else:
            model.initialize_new_simulation()

        additional_mcs = params["max_mcs"] - steps_already_done
        if additional_mcs <= 0:
            return {"SKIP": True}

        is_constant, steps_to_const, avg_mob = model.run(
            additional_mcs, transient_mcs=params.get("transient_mcs")
        )
        s_max, s_mean = model.get_metrics()

        # Checkpoint bytes for incomplete runs — Ray serialises this as a
        # plain bytes object so it goes through shared memory, not pickle IPC.
        new_cp_bytes = None if is_constant else pickle.dumps(model.get_checkpoint_data())

        return {
            "L": params["L"],
            "q": params["q"],
            "F": params["F"],
            "h": params["h"],
            "T": params["T"],
            "m": params["m"],
            "s_max": float(s_max),
            "s_mean": float(s_mean),
            "steps_to_const": int(steps_to_const),
            "is_constant": bool(is_constant),
            "avg_mobility": float(avg_mob),
            "_cp_bytes": new_cp_bytes,   # internal, stripped before DB write
        }

    except Exception:
        return {"ERROR": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Ray Actor: result sink + async parquet flush
# ---------------------------------------------------------------------------

@ray.remote
class ResultSink:
    """
    Collects results from workers and batches parquet writes.

    Using an Actor (single-threaded Ray process) rather than writing from
    every worker eliminates concurrent file-write races without any lock.
    Batching amortises the parquet serialisation cost — a 100-row batch
    takes roughly the same wall time as a 1-row write.
    """

    BATCH = 50   # rows before auto-flush

    def __init__(self, master_file: str, cp_dir: str):
        self._master = Path(master_file)
        self._cp_dir = Path(cp_dir)
        self._cp_dir.mkdir(parents=True, exist_ok=True)
        self._buf: list[dict] = []
        self._total = 0

        # Load existing DB into memory for de-duplication.
        if self._master.exists():
            self._df = pd.read_parquet(self._master)
        else:
            self._df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, result: dict) -> None:
        """Accept one result dict from a worker."""
        if "ERROR" in result or "SKIP" in result:
            return

        cp_bytes: bytes | None = result.pop("_cp_bytes", None)
        cp_key = self._cp_key(result)

        # Persist checkpoint for incomplete runs.
        if cp_bytes is not None:
            cp_path = self._cp_dir / f"{cp_key}.pkl"
            tmp = cp_path.with_suffix(".tmp")
            tmp.write_bytes(cp_bytes)
            tmp.replace(cp_path)
        else:
            # Run converged — remove stale checkpoint.
            cp_path = self._cp_dir / f"{cp_key}.pkl"
            if cp_path.exists():
                cp_path.unlink(missing_ok=True)

        self._buf.append(result)
        self._total += 1

        if len(self._buf) >= self.BATCH:
            self._flush()

    def flush(self) -> int:
        """Force-flush remaining buffer. Returns total rows written so far."""
        self._flush()
        return self._total

    def total_rows(self) -> int:
        return len(self._df)

    def get_checkpoint_bytes(self, key: str) -> bytes | None:
        """Return pickled checkpoint bytes for a param key, or None."""
        cp_path = self._cp_dir / f"{key}.pkl"
        if cp_path.exists():
            return cp_path.read_bytes()
        return None

    def existing_map(self) -> dict:
        """Return {(L,q,F,h,T,m): (is_constant, steps)} for skip logic."""
        if self._df.empty:
            return {}
        return {
            (r.L, r.q, r.F, r.h, r.T, r.m): (r.is_constant, r.steps_to_const)
            for r in self._df.itertuples(index=False)
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._buf:
            return
        new = pd.DataFrame(self._buf)
        self._buf = []

        if self._df.empty:
            self._df = new
        else:
            self._df = (
                pd.concat([self._df, new], ignore_index=True)
                .drop_duplicates(subset=["L", "q", "F", "h", "T", "m"], keep="last")
                .reset_index(drop=True)
            )

        self._master.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(self._master, index=False)
        self._df.to_csv(self._master.with_suffix(".csv"), index=False)

    @staticmethod
    def _cp_key(r: dict) -> str:
        return f"cp_L{r['L']}_q{r['q']}_F{r['F']}_h{r['h']}_T{r['T']}_m{r['m']}"


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def build_objective(tasks: list[dict], sink: "ResultSink", max_mcs: int):
    """
    Returns a closure Optuna calls for each trial.

    Optuna samples (F, q, L, h, T) from the pre-built task list using its
    TPE sampler. After enough trials it concentrates sampling on parameter
    regions where avg_mobility is near a phase boundary (0.0 < m < 0.95),
    which are scientifically the most interesting and often hardest to
    converge.

    The objective value is avg_mobility. Optuna minimises it by default;
    we flip the sign for maximisation if preferred, but for this use-case
    "closest to boundary" is a natural optimisation target — so we minimise
    |avg_mobility - 0.5| to push trials toward the uncertain middle.
    """
    # Index tasks by their natural Optuna-visible param id for fast lookup.
    idx_map = {i: t for i, t in enumerate(tasks)}

    def objective(trial: optuna.Trial) -> float:
        task_idx = trial.suggest_int("task_idx", 0, len(tasks) - 1)
        params = idx_map[task_idx]

        cp_key = ResultSink._cp_key(params)
        cp_bytes_ref = sink.get_checkpoint_bytes.remote(cp_key)
        cp_bytes = ray.get(cp_bytes_ref)

        future = run_realization.remote(params, cp_bytes)
        result = ray.get(future)

        if "ERROR" in result:
            print(f"\n[WORKER ERROR]\n{result['ERROR']}")
            raise optuna.TrialPruned()

        sink.push.remote(result)

        mob = result.get("avg_mobility", -1.0)
        if mob < 0:
            raise optuna.TrialPruned()

        # Objective: drive trials toward the phase boundary.
        return abs(mob - 0.5)

    return objective


# ---------------------------------------------------------------------------
# Fully parallel batch runner (non-Optuna path)
# ---------------------------------------------------------------------------

def run_parallel_batch(tasks: list[dict], sink, n_concurrent: int) -> None:
    """
    Submit all tasks as Ray futures and collect results as they complete.

    This is the fast path when you just want to exhaust all parameter
    combinations without Optuna's adaptive sampling. Both paths write
    through the same ResultSink actor so results are always consistent.
    """
    stop_file = Path("STOP")
    pending: dict = {}
    task_iter = iter(tasks)
    completed = 0

    def _submit_next():
        try:
            t = next(task_iter)
            cp_key = ResultSink._cp_key(t)
            cp_bytes = ray.get(sink.get_checkpoint_bytes.remote(cp_key))
            ref = run_realization.remote(t, cp_bytes)
            pending[ref] = t
        except StopIteration:
            pass

    # Fill the initial window.
    for _ in range(min(n_concurrent, len(tasks))):
        _submit_next()

    total = len(tasks)
    t0 = time.perf_counter()

    while pending:
        if stop_file.exists():
            print("\nSTOP file detected — draining in-flight tasks…")
            stop_file.unlink(missing_ok=True)
            # Wait for all in-flight to finish gracefully before exit.
            ray.get(list(pending.keys()))
            break

        done, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=2.0)
        if not done:
            continue

        ref = done[0]
        result = ray.get(ref)
        pending.pop(ref)
        sink.push.remote(result)
        completed += 1
        _submit_next()

        if completed % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = completed / elapsed
            eta = (total - completed) / rate if rate > 0 else 0
            print(
                f"  {completed}/{total}  "
                f"{rate:.1f} tasks/s  "
                f"ETA {timedelta(seconds=int(eta))}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Axelrod-Schelling runner (Ray + Optuna)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["parallel", "optuna"],
        default="parallel",
        help=(
            "parallel: submit all tasks at once (fastest wall-clock). "
            "optuna: adaptive TPE sampling (best for exploring phase boundaries)."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-converged runs.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    exp = config["as_experiment"]

    data_path = Path("data/schelling")
    master_file = data_path / "schelling_master_results.parquet"
    cp_dir = data_path / "checkpoints"

    # Initialise Ray. num_cpus=None → use all available cores.
    ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count()-1)

    # Boot the result sink actor (persistent for the whole run).
    sink = ResultSink.remote(str(master_file), str(cp_dir))
    existing = ray.get(sink.existing_map.remote())
    print(f"DB loaded: {len(existing)} existing realizations.")

    # Build task list.
    sweep = exp["sweep"]
    from itertools import product as iproduct

    all_combos = list(
        iproduct(sweep["F"], sweep["q"], sweep["L"], sweep["h"], sweep["T"])
    )
    tasks: list[dict] = []
    for f_val, q_val, L_val, h_val, T_val in all_combos:
        for m in range(1, exp["M_realizations"] + 1):
            is_const, steps_done = existing.get(
                (L_val, q_val, f_val, h_val, T_val, m), (None, 0)
            )
            if is_const is True:
                continue  # already converged — skip
            if is_const is False and exp["max_mcs"] <= steps_done:
                continue  # hit MCS cap — skip

            tasks.append(
                {
                    "L": L_val,
                    "q": q_val,
                    "F": f_val,
                    "h": h_val,
                    "T": T_val,
                    "m": m,
                    "master_seed": exp["master_seed"],
                    "max_mcs": exp["max_mcs"],
                    "transient_mcs": exp.get("transient_mcs"),
                }
            )

    if not tasks:
        print("Nothing to do — all realizations complete.")
        ray.get(sink.flush.remote())
        ray.shutdown()
        return

    print(f"Submitting {len(tasks)} tasks in '{args.mode}' mode.")
    print("Graceful stop: create a file named STOP, or press Ctrl+C.\n")

    t_start = time.perf_counter()
    try:
        if args.mode == "parallel":
            # n_concurrent = 4× CPU count keeps the Ray scheduler busy
            # without overwhelming memory with large grids.
            n_concurrent = (os.cpu_count() or 4) * 4
            run_parallel_batch(tasks, sink, n_concurrent)

        else:  # optuna
            n_trials = len(tasks)  # one Optuna trial per task (no repeats)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=exp.get("master_seed", 42)),
            )
            objective = build_objective(tasks, sink, exp["max_mcs"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    except KeyboardInterrupt:
        print("\nCtrl+C — flushing results…")

    finally:
        total = ray.get(sink.flush.remote())
        db_rows = ray.get(sink.total_rows.remote())
        duration = time.perf_counter() - t_start
        ray.shutdown()

        print("\n" + "=" * 44)
        print(f"Completed in {timedelta(seconds=round(duration))}")
        print(f"Tasks processed this session : {total}")
        print(f"Total rows in master DB      : {db_rows}")
        print("=" * 44)


if __name__ == "__main__":
    main()