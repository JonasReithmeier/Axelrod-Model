import numpy as np
from .core_as import run_steps_as, get_cluster_metrics_as

class AxelrodSchellingModel:
    def __init__(self, width=40, height=40, F=5, q=10, h=0.1, T=0.5, seed=42):
        self.W = width
        self.H = height
        self.F = F
        self.q = q
        self.h = h # Fraction of empty sites
        self.T = T # Tolerance for differences
        self.seed = seed

        self.N_cells = self.W * self.H
        self.num_empty = int(self.N_cells * self.h)
        
        self.rng = np.random.default_rng(self.seed)
        self.grid = None
        self.empty_locs = None
        self.updates_since_last_change = 0
        self.total_steps = 0

    def initialize_new_simulation(self):
        # 1. Fill completely with random agents
        self.grid = self.rng.integers(0, self.q, size=(self.W, self.H, self.F), dtype=np.int16)
        
        # 2. Punch holes to create empty sites (-1)
        self.empty_locs = np.zeros((self.num_empty, 2), dtype=np.int32) # num_empty (rows) x 2 (colums) 
        
        if self.num_empty > 0:
            flat_indices = self.rng.choice(self.N_cells, size=self.num_empty, replace=False)
            for i, idx in enumerate(flat_indices):
                ex, ey = idx % self.W, idx // self.W  # % modulo, "x // y" return how often y fits into x (= (x-(x%y))/y)
                self.grid[ex, ey, :] = -1
                self.empty_locs[i] = [ex, ey]
                
        self.updates_since_last_change = 0
        self.total_steps = 0

    def run(self, additional_steps):
        if additional_steps <= 0: 
            ValueError("additional steps 0 reached run()")
            return 0, False
        threshold = self.N_cells * 100
        
        steps_done, new_updates, is_frozen = run_steps_as(
            self.grid, self.W, self.H, self.F, self.empty_locs, self.num_empty, 
            self.T, additional_steps, self.updates_since_last_change, threshold, self.rng
        )
        
        self.updates_since_last_change = new_updates
        self.total_steps += steps_done
        return steps_done, is_frozen

    def get_metrics(self):
        return get_cluster_metrics_as(self.grid, self.W, self.H, self.F)

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "empty_locs": self.empty_locs,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,
            "total_steps": self.total_steps
        }

    def load_checkpoint_data(self, cp):
        self.grid = cp["grid"]
        self.empty_locs = cp["empty_locs"]
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"]
        self.total_steps = cp.get("total_steps", 0)