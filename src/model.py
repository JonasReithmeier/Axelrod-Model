import numpy as np
from .core import run_steps, get_cluster_metrics

class FastAxelrodModel_regularLattice:
    def __init__(self, width=None, height=None, F=None, q=None, seed=None, config=None):
        cfg = config if config is not None else {}
        self.width = width if width is not None else cfg.get('width', 10)
        self.height = height if height is not None else cfg.get('height', 10)
        self.F = F if F is not None else cfg.get('features', 5)
        self.q = q if q is not None else cfg.get('traits', 10)
        self.seed = seed if seed is not None else cfg.get('seed')

        self.N = self.width * self.height
        
        # Initialize Modern NumPy RNG (non randomized for=> reproducibility for same seed and same state for sifferent runs)
        self.rng = np.random.default_rng(self.seed)
        self.grid = None
        self.updates_since_last_change = 0  
        self.total_steps_done = 0

    def initialize_new_simulation(self):
        # Generate initial grid using the explicit RNG
        self.grid = self.rng.integers(0, self.q, size=(self.width, self.height, self.F), dtype=np.int16)
        self.updates_since_last_change = 0

    def run(self, max_steps):
        threshold = self.N * 100
        
        # Pass the RNG object into Numba! Numba will mutate its state in-place.
        steps_done, new_updates, is_frozen = run_steps(
            self.grid, max_steps, self.F, self.width, self.height, 
            self.updates_since_last_change, threshold, self.rng
        )
        
        self.updates_since_last_change = new_updates
        self.total_steps_done += steps_done
        return is_frozen

    def get_metrics(self):
        return get_cluster_metrics(self.grid, self.width, self.height, self.F)

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,  # Exact RNG state!
            "prev_steps": self.total_steps_done
        }

    def load_checkpoint_data(self, cp):
        self.grid = cp["grid"]
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"] # Restore Exact RNG State!
        self.total_steps_done = cp['prev_steps']


    def single_step_ONLY_visualization(self):
        diff_indices = np.empty(self.F, dtype=np.int32)
        # Use the explicit RNG passed from Python
        x = self.rng.integers(0, self.width)
        y = self.rng.integers(0, self.height)
        
        direction = self.rng.integers(0, 4)
        nx, ny = x, y
        if direction == 0: nx = (x + 1) % self.width
        elif direction == 1: nx = (x - 1) % self.width
        elif direction == 2: ny = (y + 1) % self.height
        else: ny = (y - 1) % self.height
            
        shared = 0
        diff_count = 0
        for i in range(self.F):
            if self.grid[x, y, i] == self.grid[nx, ny, i]:
                shared += 1
            else:
                diff_indices[diff_count] = i
                diff_count += 1
                
        # Interaction Rule
        if 0 < shared < self.F:
            if self.rng.random() * self.F < shared:
                target_trait = diff_indices[self.rng.integers(0, diff_count)]
                self.grid[x, y, target_trait] = self.grid[nx, ny, target_trait]
