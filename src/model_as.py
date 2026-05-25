import numpy as np
from .core_as import run_steps_as, get_cluster_metrics_as, calculate_mobility_as

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

    def run(self, additional_steps, check_mobility_NESS=True, MCS_before_check=0):
        if additional_steps <= 0: 
            return 0, False, -1.0 
            
        threshold = self.N_cells * 100000
        transient_steps_required = MCS_before_check * self.N_cells
        
        steps_left = additional_steps
        total_steps_this_call = 0
        last_calculated_m = -1.0 
        
        while steps_left > 0:
            
            steps_done, new_updates, status = run_steps_as(
                self.grid, self.W, self.H, self.F, self.empty_locs, self.num_empty, 
                self.T, steps_left, self.updates_since_last_change, threshold, self.rng
            )
            
            self.total_steps += steps_done
            total_steps_this_call += steps_done
            steps_left -= steps_done
            self.updates_since_last_change = new_updates
            
            if status == 0:
                # run didnt hit threshold
                return total_steps_this_call, False, last_calculated_m
                
            elif status == 1:
                #classical frozen
                return total_steps_this_call, True, 0.0
                
            elif status == 2:
                # hit threshold not classical frozen; periodid erosion or false alarm
                # 1. Check if we are allowed to monitor yet
                if not check_mobility_NESS or self.total_steps < transient_steps_required:
                    self.updates_since_last_change = 0 # Ignore timeout, keep running
                    continue
                
                # 2. Enter Monitoring Phase
                monitoring_sweeps = 50000 # Reduced slightly to save compute
                m_history = np.zeros(monitoring_sweeps)
                extra_steps = 0
                huge_threshold = np.iinfo(np.int64).max
                
                for sweep in range(monitoring_sweeps):
                    s, u, _ = run_steps_as(
                        self.grid, self.W, self.H, self.F, self.empty_locs, self.num_empty, 
                        self.T, self.N_cells, self.updates_since_last_change, huge_threshold, self.rng
                    )
                    extra_steps += s
                    self.updates_since_last_change = u
                    m_history[sweep] = calculate_mobility_as(self.grid, self.W, self.H, self.F, self.T, self.num_empty)
                    
                self.total_steps += extra_steps
                total_steps_this_call += extra_steps
                steps_left -= extra_steps 
                
                x_axis = np.arange(monitoring_sweeps)
                slope, _ = np.polyfit(x_axis, m_history, 1)
                final_m = np.mean(m_history)
                last_calculated_m = final_m 
                
                # 3. The "High-Plateau Veto" (m < 0.95 prevents false alarms at the start)
                if abs(slope) < 1e-4 and final_m < 0.95:
                    return total_steps_this_call, True, final_m
                else:
                    self.updates_since_last_change = 0 
                    
        return total_steps_this_call, False, last_calculated_m

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