import numpy as np
import networkx as nx
from .core_sw import run_steps_sw, get_cluster_metrics_sw, compute_clustering_coefficient, compute_characteristic_path_length

class AxelrodSmallWorld:
    def __init__(self, N, k, p, F, q, iterationNumber, seed):
        self.N = N
        self.k = k        # Nearest neighbors to connect to initially (4 mimics 2D lattice)
        self.p = p        # Rewiring probability (0.0 = ring, 1.0 = random graph)
        self.F = F
        self.q = q
        self.iterationNumber = iterationNumber
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)
        self.grid = None
        self.updates_since_last_change = 0
        self.total_steps = 0 
        
        # Graph CSR (Compressed Sparse Row) Structures: !D array => entries next to each other in physical RAM (not the case for lists of lists); numba can only perform with basic data types adn numpy 1D arrays
        self.edge_ptrs = None   #pointers to assign edges in 1D self.edges list to agents [0,2,3]: agaent one has friends = edges[0:2] = edges[0], edges[1], agent two has one friend, edges[2]
        self.edges = None

    def _build_graph(self):
        """Generates Watts-Strogatz graph and converts to fast CSR format."""
        # Use exact same seed so topology is deterministically identical across resumes
        G = nx.watts_strogatz_graph(self.N, self.k, self.p, seed=self.seed) # at initialisation: With probability p, a connection gets unplugged and rewired it to a completely random agent across the world.
        
        # Convert NetworkX to Adjacency List
        adj_list = [list(G.neighbors(n)) for n in range(self.N)]
        degrees = np.array([len(nbrs) for nbrs in adj_list], dtype=np.int32)
        
        # Create CSR Arrays for Numba
        self.edge_ptrs = np.empty(self.N + 1, dtype=np.int32)
        self.edge_ptrs[0] = 0
        self.edge_ptrs[1:] = np.cumsum(degrees)
        
        self.edges = np.concatenate(adj_list).astype(np.int32)

    def initialize_new_simulation(self):
        self._build_graph()
        self.grid = self.rng.integers(0, self.q, size=(self.N, self.F), dtype=np.int16) # 2D array: array of FD array agents
        self.updates_since_last_change = 0
        self.total_steps = 0

    def run(self, additional_steps):
        if additional_steps <= 0: return 0, False
        threshold = self.N * 100
        
        steps_done, new_updates, is_frozen = run_steps_sw(
            self.grid, self.edge_ptrs, self.edges, additional_steps, self.F, self.N, 
            self.updates_since_last_change, threshold, self.rng
        )
        
        self.updates_since_last_change = new_updates
        self.total_steps += steps_done
        return steps_done, is_frozen

    def get_metrics(self):
        return get_cluster_metrics_sw(self.grid, self.edge_ptrs, self.edges, self.N, self.F)
        
    def get_network_metrics(self):
        """Returns the topological metrics (L, C) of the static Watts-Strogatz graph."""
        L = compute_characteristic_path_length(self.edge_ptrs, self.edges, self.N)
        C = compute_clustering_coefficient(self.edge_ptrs, self.edges, self.N)
        return L, C

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,
            "iterationNumber": self.iterationNumber,
            "total_steps": self.total_steps
        }

    def load_checkpoint_data(self, cp): 
        self._build_graph() # Rebuild graph deterministically instead of pickling 
        self.grid = cp["grid"]
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"]
        self.iterationNumber = cp['iterationNumber']
        self.total_steps = cp.get("total_steps", 0) 