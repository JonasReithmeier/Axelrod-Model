import hashlib
import numpy as np
from .core_as import (run_mcs_chunk, run_monitoring_phase, 
                      check_constant_as, get_cluster_metrics_as)

class AxelrodSchellingModel:
    def __init__(self, L=40, F=5, q=10, h=0.1, T=0.5, m=1, master_seed=42):
        self.L = L
        self.F = F
        self.q = q
        self.h = h
        self.T = T
        self.m = m
        self.master_seed = master_seed

        self.N_cells = self.L * self.L
        self.num_empty = int(self.N_cells * self.h)
        
        # Self-contained hashed RNG setup
        seed_context = f"{master_seed}_{F}_{q}_{L}_{h}_{T}_{m}"
        self.seed = int(hashlib.md5(seed_context.encode()).hexdigest()[:16], 16)
        self.rng = np.random.default_rng(self.seed)
        
        self.grid = None
        self.empty_locs = None
        self.updates_since_last_change = 0
        self.total_mcs = 0

    def initialize_new_simulation(self):
        self.grid = self.rng.integers(0, self.q, size=(self.L, self.L, self.F), dtype=np.int16)
        self.empty_locs = np.zeros((self.num_empty, 2), dtype=np.int32)
        
        if self.num_empty > 0:
            flat_indices = self.rng.choice(self.N_cells, size=self.num_empty, replace=False)
            for i, idx in enumerate(flat_indices):
                ex, ey = idx % self.L, idx // self.L
                self.grid[ex, ey, :] = -1
                self.empty_locs[i] = [ex, ey]
                
        self.updates_since_last_change = 0
        self.total_mcs = 0

    def run(self, additional_mcs, transient_mcs=None):
        if additional_mcs <= 0:
            return False, self.total_mcs, -1.0 
            
        # Default un-checked transient threshold equivalent to old "10^7 steps"
        if transient_mcs is None:
            transient_mcs = 10_000_000 // self.N_cells

        threshold_updates = self.N_cells * 100000
        mcs_left = additional_mcs
        last_m = -1.0 
        
        # Phase 1: Force deep uninterruptable chunk without thresholds
        if self.total_mcs < transient_mcs:
            mcs_to_do = min(mcs_left, transient_mcs - self.total_mcs)
            mcs_done, self.updates_since_last_change, _ = run_mcs_chunk(
                self.grid, self.L, self.F, self.empty_locs, self.num_empty, 
                self.T, mcs_to_do, self.updates_since_last_change, np.iinfo(np.int64).max, self.rng
            )
            self.total_mcs += mcs_done
            mcs_left -= mcs_done
        
        # Phase 2: Interleaved checking 
        while mcs_left > 0:
            chunk = min(mcs_left, 10000) # Prevents Python starvation / allows checkpoints
            mcs_done, self.updates_since_last_change, hit_threshold = run_mcs_chunk(
                self.grid, self.L, self.F, self.empty_locs, self.num_empty, 
                self.T, chunk, self.updates_since_last_change, threshold_updates, self.rng
            )
            self.total_mcs += mcs_done
            mcs_left -= mcs_done
            
            if hit_threshold:
                # Direct switch into alternative pure @njit logic checks
                if check_constant_as(self.grid, self.L, self.F, self.T, self.num_empty):
                    return True, self.total_mcs, 0.0
                
                mon_sweeps, final_m, slope, new_updates = run_monitoring_phase(
                    self.grid, self.L, self.F, self.empty_locs, self.num_empty, 
                    self.T, self.rng, self.updates_since_last_change
                )
                
                self.total_mcs += mon_sweeps
                mcs_left -= mon_sweeps 
                last_m = final_m
                self.updates_since_last_change = new_updates
                
                if abs(slope) < 1e-4 and final_m < 0.95:
                    return True, self.total_mcs, final_m
                else:
                    self.updates_since_last_change = 0 
                    
        return False, self.total_mcs, last_m

    def get_metrics(self):
        return get_cluster_metrics_as(self.grid, self.L, self.F)

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "empty_locs": self.empty_locs,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,
            "total_mcs": self.total_mcs
        }

    def load_checkpoint_data(self, cp):
        self.grid = cp["grid"]
        self.empty_locs = cp["empty_locs"]
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"]
        self.total_mcs = cp.get("total_mcs", 0)