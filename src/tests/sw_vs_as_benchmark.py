
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import time

# Import both models from your project
from src.model_as import AxelrodSchellingModel
from src.model_sw import AxelrodSmallWorld

def main():
    print("="*50)
    print(" AXELROD MODEL BENCHMARK: SCHELLING vs SMALL WORLD")
    print("="*50)

    # --- 1. Configuration ---
    # We want exact equivalence: 1,600 agents, 10,000,000 steps
    L = 40
    N_agents = L * L  # 1600
    F = 5
    q = 10
    total_agent_steps = 10_000_000

    # Calculate exact inputs for both models
    sw_max_steps = total_agent_steps
    as_max_mcs = total_agent_steps // N_agents  # 6250 MCS

    print(f"\n[CONFIGURATION]")
    print(f"Grid/Network Size : N = {N_agents} (L = {L} for Schelling)")
    print(f"Total Agent Steps : {total_agent_steps:,}")
    print(f"Small World Input : {sw_max_steps:,} steps")
    print(f"Schelling Input   : {as_max_mcs:,} MCS")

    # Initialize models
    print("\n[INITIALIZING MODELS...]")
    sw_model = AxelrodSmallWorld(N=N_agents, k=4, p=0.1, F=F, q=q, seed=42)
    sw_model.initialize_new_simulation()

    as_model = AxelrodSchellingModel(L=L, F=F, q=q, h=0.1, T=0.5, m=1, master_seed=42)
    as_model.initialize_new_simulation()

    # --- 2. Warm-up Phase (CRITICAL FOR NUMBA) ---
    # Numba takes 2-5 seconds to compile C-code on the very first run. 
    # We run 1 step/MCS to get compilation out of the way.
    print("\n[WARMING UP NUMBA] (Compiling C-Code... this takes a few seconds)")
    sw_model.run(1) 
    as_model.run(1, transient_mcs=1) 

    # Re-initialize to reset the grids/networks for the actual test
    sw_model.initialize_new_simulation()
    as_model.initialize_new_simulation()
    print("Warm-up complete. Numba is compiled to C-speed.")

    # --- 3. Run Benchmark: Small World ---
    print(f"\n--- RUNNING SMALL WORLD ({total_agent_steps:,} steps) ---")
    start_sw = time.perf_counter()
    sw_model.run(sw_max_steps)
    sw_time = time.perf_counter() - start_sw
    
    sw_steps_per_sec = total_agent_steps / sw_time
    print(f"Time taken       : {sw_time:.4f} seconds")
    print(f"Speed            : {sw_steps_per_sec:,.0f} steps / second")

    # --- 4. Run Benchmark: Schelling ---
    print(f"\n--- RUNNING SCHELLING ({as_max_mcs:,} MCS = {total_agent_steps:,} steps) ---")
    start_as = time.perf_counter()
    # transient_mcs is set high so it doesn't waste time checking if it froze
    as_model.run(as_max_mcs, transient_mcs=as_max_mcs)
    as_time = time.perf_counter() - start_as
    
    as_steps_per_sec = total_agent_steps / as_time
    print(f"Time taken       : {as_time:.4f} seconds")
    print(f"Speed            : {as_steps_per_sec:,.0f} steps / second")

    # --- 5. Analysis ---
    print("\n" + "="*50)
    print(" PERFORMANCE ANALYSIS")
    print("="*50)
    if as_time > sw_time:
        ratio = as_time / sw_time
        print(f"Small World is {ratio:.1f}x faster per agent step.")
    else:
        print("Schelling was faster (This shouldn't happen mathematically unless SW has a bottleneck!)")
    print("="*50)


if __name__ == "__main__":
    main()