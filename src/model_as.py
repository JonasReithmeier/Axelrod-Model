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

        # Graph CSR Structures
        self.edge_ptrs = None
        self.edges = None

    def _build_graph(self):
        """Generates 2D periodic von Neumann grid mapped directly into fast 1D CSR structure."""
        self.edge_ptrs = np.arange(0, 4 * self.N_cells + 1, 4, dtype=np.int32)
        self.edges = np.empty(4 * self.N_cells, dtype=np.int32)
        
        idx = 0
        for i in range(self.L):
            for j in range(self.L):
                # Using the exact same neighbor order as previous (0,1), (0,-1), (1,0), (-1,0)
                right = i * self.L + ((j + 1) % self.L)
                left = i * self.L + ((j - 1) % self.L)
                down = ((i + 1) % self.L) * self.L + j
                up = ((i - 1) % self.L) * self.L + j
                
                self.edges[idx]   = right
                self.edges[idx+1] = left
                self.edges[idx+2] = down
                self.edges[idx+3] = up
                idx += 4

    def initialize_new_simulation(self):
        self._build_graph()
        # 1D array of F-D agent arrays -> much more cache-friendly
        self.grid = self.rng.integers(0, self.q, size=(self.N_cells, self.F), dtype=np.int16)
        self.empty_locs = np.zeros(self.num_empty, dtype=np.int32)
        
        if self.num_empty > 0:
            flat_indices = self.rng.choice(self.N_cells, size=self.num_empty, replace=False)
            for i, idx in enumerate(flat_indices):
                self.grid[idx, :] = -1
                self.empty_locs[i] = idx
                
        self.updates_since_last_change = 0
        self.total_mcs = 0

    def run(self, additional_mcs, transient_mcs=None):
        if additional_mcs <= 0:
            return False, self.total_mcs, -1.0 
            
        if transient_mcs is None:
            transient_mcs = 10_000_000 // self.N_cells

        threshold_updates = self.N_cells * 100000
        mcs_left = additional_mcs
        last_m = -1.0 
        
        # Determine maximum neighbors natively so underlying numba bounds are safe/dynamic
        max_degree = np.max(self.edge_ptrs[1:] - self.edge_ptrs[:-1])
        
        # Phase 1: Force deep uninterruptable chunk without thresholds
        if self.total_mcs < transient_mcs:
            mcs_to_do = min(mcs_left, transient_mcs - self.total_mcs)
            mcs_done, self.updates_since_last_change, _ = run_mcs_chunk(
                self.grid, self.N_cells, self.F, self.empty_locs, self.num_empty, 
                self.T, mcs_to_do, self.updates_since_last_change, np.iinfo(np.int64).max, self.rng,
                self.edge_ptrs, self.edges, max_degree
            )
            self.total_mcs += mcs_done
            mcs_left -= mcs_done
        
        # Phase 2: Interleaved checking 
        while mcs_left > 0:
            chunk = min(mcs_left, 10000) # Prevents Python starvation
            mcs_done, self.updates_since_last_change, hit_threshold = run_mcs_chunk(
                self.grid, self.N_cells, self.F, self.empty_locs, self.num_empty, 
                self.T, chunk, self.updates_since_last_change, threshold_updates, self.rng,
                self.edge_ptrs, self.edges, max_degree
            )
            self.total_mcs += mcs_done
            mcs_left -= mcs_done
            
            if hit_threshold:
                if check_constant_as(self.grid, self.N_cells, self.F, self.T, self.num_empty, self.edge_ptrs, self.edges):
                    return True, self.total_mcs, 0.0
                
                mon_sweeps, final_m, slope, new_updates = run_monitoring_phase(
                    self.grid, self.N_cells, self.F, self.empty_locs, self.num_empty, 
                    self.T, self.rng, self.updates_since_last_change,
                    self.edge_ptrs, self.edges, max_degree
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
        return get_cluster_metrics_as(self.grid, self.N_cells, self.F, self.edge_ptrs, self.edges)

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "empty_locs": self.empty_locs,
            "edge_ptrs": self.edge_ptrs,
            "edges": self.edges,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,
            "total_mcs": self.total_mcs
        }

    def load_checkpoint_data(self, cp):
        self.grid = cp["grid"]
        # Convert older 3D Checkpoint models to 1D flat structures safely
        if self.grid.ndim == 3:
            self.grid = self.grid.reshape(-1, self.F)
            
        self.empty_locs = cp["empty_locs"]
        if self.empty_locs.ndim == 2:
            self.empty_locs = np.array([x * self.L + y for x, y in self.empty_locs], dtype=np.int32)
            
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"]
        self.total_mcs = cp.get("total_mcs", 0)

        if "edge_ptrs" in cp and "edges" in cp:
            self.edge_ptrs = cp["edge_ptrs"]
            self.edges = cp["edges"]
        else:
            self._build_graph()