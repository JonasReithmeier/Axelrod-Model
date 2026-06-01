"""
mobility_compute.py
-------------------
Computes mobility m(t) trajectories for the Axelrod-Schelling model and
journals results incrementally to a parquet database.

Each (h, F, L, T, q_N, m) combination is one trajectory — a time series
of (mcs, mobility) pairs stored as a single compressed row in the DB.

Storing the full trajectory as an array per row (rather than one row per
time point) keeps the parquet file compact and makes loading fast: reading
one trajectory is one row fetch, not a filtered scan over millions of rows.

Usage:
    python mobility_compute.py              # use config.yaml
    python mobility_compute.py --config path/to/config.yaml
    python mobility_compute.py --dry-run    # print task list, do nothing

Graceful stop: create a file named STOP in the working directory.
Completed trajectories are safe in the DB immediately after each run.
"""
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import argparse
import time
import pickle
import random
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.model_as import AxelrodSchellingModel
from src.core_as import run_mcs_chunk, calculate_mobility_as


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_db(db_path: Path) -> pd.DataFrame:
    if db_path.exists():
        return pd.read_parquet(db_path)
    return pd.DataFrame()


def _save_db(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(db_path)


def _existing_keys(df: pd.DataFrame) -> set:
    """
    Return set of (h, F, L, T, q_N, realization) tuples already in DB.
    Tolerates old DBs that used 'm' instead of 'realization', and DBs
    that are missing columns entirely (treats them as empty).
    """
    if df.empty:
        return set()
    required = {"h", "F", "L", "T", "q_N"}
    if not required.issubset(df.columns):
        return set()
    # Accept either column name for the realization index
    if "realization" in df.columns:
        real_col = df["realization"]
    elif "m" in df.columns:
        real_col = df["m"]
    else:
        # No realization column at all — treat every row as realization 1
        real_col = pd.Series(1, index=df.index)
    return set(zip(df["h"], df["F"], df["L"], df["T"], df["q_N"], real_col))


def _append_to_db(db_path: Path, row: dict) -> None:
    """
    Merge one completed trajectory row into the parquet DB atomically.
    Trajectories are stored with numpy arrays serialised via pickle bytes
    inside an object column — parquet handles bytes columns natively.
    Migrates old DBs that used 'm' instead of 'realization' on the fly.
    """
    df_old = _load_db(db_path)
    df_new = pd.DataFrame([row])

    if df_old.empty:
        df_final = df_new
    else:
        if "m" in df_old.columns and "realization" not in df_old.columns:
            df_old = df_old.rename(columns={"m": "realization"})
        df_final = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(
                subset=["h", "F", "L", "T", "q_N", "realization"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    _save_db(df_final, db_path)


# ---------------------------------------------------------------------------
# Single trajectory runner
# ---------------------------------------------------------------------------

def run_trajectory(
    L: int, F: int, q: int, h: float, T: float,
    realization: int, master_seed: int,
    max_mcs: int, data_points: int,
) -> dict:
    """
    Run one full mobility trajectory and return a result dict.

    The time axis (mcs_axis) and mobility values (m_values) are stored as
    raw numpy arrays — compact and ready for direct plotting.
    """
    model = AxelrodSchellingModel(
        L=L, F=F, q=q, h=h, T=T,
        m=realization, master_seed=master_seed,
    )
    model.initialize_new_simulation()

    N_cells    = model.N_cells
    q_N        = q / N_cells
    max_degree = int(np.max(model.edge_ptrs[1:] - model.edge_ptrs[:-1]))

    mcs_per_sample = max(1, max_mcs // data_points)
    huge           = np.iinfo(np.int64).max   # disable early-stopping

    mcs_axis = [0]
    m_values = [
        calculate_mobility_as(
            model.grid, N_cells, F, T,
            model.num_empty, model.edge_ptrs, model.edges,
        )
    ]

    current_mcs = 0
    while current_mcs < max_mcs:
        chunk = min(mcs_per_sample, max_mcs - current_mcs)

        mcs_done, model.updates_since_last_change, _ = run_mcs_chunk(
            model.grid, N_cells, F,
            model.empty_locs, model.num_empty, T,
            chunk, model.updates_since_last_change, huge,
            model.rng,
            model.edge_ptrs, model.edges, max_degree,
        )
        current_mcs += mcs_done

        m_values.append(
            calculate_mobility_as(
                model.grid, N_cells, F, T,
                model.num_empty, model.edge_ptrs, model.edges,
            )
        )
        mcs_axis.append(current_mcs)

    return {
        "L":           L,
        "F":           F,
        "q":           q,
        "h":           h,
        "T":           T,
        "q_N":         round(q_N, 8),
        "realization": realization,
        "max_mcs":     max_mcs,
        "data_points": len(mcs_axis),
        # stored as pickle bytes — avoids ragged-array issues in parquet
        "mcs_axis":    pickle.dumps(np.array(mcs_axis, dtype=np.int64)),
        "m_values":    pickle.dumps(np.array(m_values, dtype=np.float64)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print task list without running anything.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["as_mobility_experiment"]

    db_path = Path(cfg.get("db_path", "data/schelling/mobility_trajectories.parquet"))
    stop    = Path("STOP")

    sweep = cfg["sweep"]
    combos = list(product(
        sweep["F"], sweep["q"], sweep["L"], sweep["h"], sweep["T"]
    ))
    M = cfg.get("M_realizations", 1)

    # Build task list excluding already-completed trajectories
    df_existing   = _load_db(db_path)
    existing_keys = _existing_keys(df_existing)
    print(f"DB loaded: {len(existing_keys)} trajectories already complete.")

    tasks = []
    for f_val, q_val, L_val, h_val, T_val in combos:
        for real in range(1, M + 1):
            q_N = round(q_val / (L_val * L_val), 8)
            key = (h_val, f_val, L_val, T_val, q_N, real)
            if key not in existing_keys:
                tasks.append({
                    "L": L_val, "F": f_val, "q": q_val,
                    "h": h_val, "T": T_val,
                    "realization":  real,
                    "master_seed":  cfg["master_seed"],
                    "max_mcs":      cfg["max_mcs"],
                    "data_points":  cfg.get("data_points", 1000),
                })

    if not tasks:
        print("All trajectories already complete. Nothing to do.")
        return

    random.shuffle(tasks)
    print(f"{len(tasks)} trajectories to compute.")

    if args.dry_run:
        for t in tasks:
            print(f"  L={t['L']} F={t['F']} q={t['q']} "
                  f"h={t['h']} T={t['T']} real={t['realization']}")
        return

    print("Graceful stop: create a file named STOP in this directory.\n")

    t0        = time.perf_counter()
    completed = 0

    for task in tasks:
        if stop.exists():
            stop.unlink(missing_ok=True)
            print("\nSTOP detected — exiting cleanly.")
            break

        t_task = time.perf_counter()
        label  = (f"L={task['L']} F={task['F']} q={task['q']} "
                  f"h={task['h']} T={task['T']} r={task['realization']}")
        print(f"  Running {label} ...", end=" ", flush=True)

        row = run_trajectory(**task)
        _append_to_db(db_path, row)

        completed += 1
        elapsed   = time.perf_counter() - t0
        task_time = time.perf_counter() - t_task
        rate      = completed / elapsed
        remaining = len(tasks) - completed
        eta       = remaining / rate if rate > 0 else 0

        print(f"done ({task_time:.1f}s)  "
              f"[{completed}/{len(tasks)}  "
              f"ETA {timedelta(seconds=int(eta))}]")

    total = time.perf_counter() - t0
    final_df = _load_db(db_path)
    print(f"\nFinished in {timedelta(seconds=int(total))}. "
          f"DB now has {len(final_df)} trajectories.")


if __name__ == "__main__":
    main()