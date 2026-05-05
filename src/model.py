import numpy as np
import random
import pickle
from .agent import AxelrodAgent
from .utils import calculate_similarity, get_different_traits

class AxelrodModel_regularLattice:
    def __init__(self, width=None, height=None, F=None, q=None, seed=None, config=None):
        cfg = config if config is not None else {}
        self.width = width if width is not None else cfg.get('width', 10)
        self.height = height if height is not None else cfg.get('height', 10)
        self.F = F if F is not None else cfg.get('features', 5)
        self.q = q if q is not None else cfg.get('traits', 10)
        self.seed = seed if seed is not None else cfg.get('seed')

        if None in [self.width, self.height, self.F, self.q, self.seed]:
            raise ValueError("Missing parameters or seed!")

        # Initialize RNGs
        self.array_rng = np.random.default_rng(self.seed)
        self.scalar_rng = random.Random(self.seed)
        
        # Internal state
        self.grid = np.empty((self.width, self.height), dtype=object)
        self.updates_since_last_change = 0

    def initialize_new_simulation(self):
        """Fills the grid with brand new random agents."""
        for x in range(self.width):
            for y in range(self.height):
                c_vector = self.array_rng.integers(0, self.q, size=self.F)
                self.grid[x, y] = AxelrodAgent(pos=(x, y), culture_vector=c_vector)

    def get_checkpoint_data(self):
        """Captures the entire state of the model for saving."""
        return {
            "culture_tensor": self.get_culture_tensor(),
            "updates_since_last_change": self.updates_since_last_change,
            "rng_numpy_state": self.array_rng.bit_generator.state,
            "rng_python_state": self.scalar_rng.getstate()
        }

    def load_checkpoint_data(self, cp):
        """Injects a saved state into the model (Resumes simulation)."""
        # 1. Restore RNGs
        self.array_rng.bit_generator.state = cp["rng_numpy_state"]
        self.scalar_rng.setstate(cp["rng_python_state"])
        
        # 2. Restore Counter
        self.updates_since_last_change = cp["updates_since_last_change"]
        
        # 3. Restore Grid
        tensor = cp["culture_tensor"]
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x, y] = AxelrodAgent(pos=(x, y), culture_vector=tensor[x, y].copy())

    def get_culture_tensor(self):
        tensor = np.zeros((self.width, self.height, self.F), dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                tensor[x, y] = self.grid[x, y].culture
        return tensor

    @property # accessable like class variable; but single source of truth: computed when called 
    def N(self):
        return self.width * self.height

    def get_random_neighbor(self, x, y):
        """Finds a neighbor using Von Neumann neighborhood (Up, Down, Left, Right)"""
        neighbors = []
        # Torus (wrap-around) logic: % self.width
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % self.width, (y + dy) % self.height
            neighbors.append(self.grid[nx, ny])
        return self.scalar_rng.choice(neighbors)

    def step(self):
        """A single iteration of the Axelrod algorithm"""
        # 1. Randomly pick an active agent
        x, y = self.scalar_rng.randint(0, self.width-1), self.scalar_rng.randint(0, self.height-1)
        agent_a = self.grid[x, y]

        # 2. Randomly pick one of its neighbors
        agent_b = self.get_random_neighbor(x, y)

        # 3. Calculate similarity
        sim = calculate_similarity(agent_a, agent_b, self.F)

        # 4. Interaction Rule
        # If similarity is 0 or 1, nothing happens.
        # Otherwise, they interact with probability equal to their similarity.
        if 0 < sim < 1:
            if self.scalar_rng.random() < sim:
                # Agent A adopts one trait from B that they don't share
                diff_indices = get_different_traits(agent_a, agent_b)
                target_trait = self.scalar_rng.choice(diff_indices)
                agent_a.culture[target_trait] = agent_b.culture[target_trait]
            self.updates_since_last_change = 0
            return True
        
        self.updates_since_last_change += 1
        return False

    
    def is_totally_frozen(self):
        """Prüft mathematisch exakt, ob noch irgendeine Interaktion möglich ist."""
        # Wir prüfen für jeden Agenten seine rechten und unteren Nachbarn (Torus)
        for x in range(self.width):
            for y in range(self.height):
                agent = self.grid[x, y]
                # Wir müssen nur zwei Richtungen prüfen, um alle Verbindungen abzudecken
                for dx, dy in [(0, 1), (1, 0)]:
                    nx, ny = (x + dx) % self.width, (y + dy) % self.height
                    neighbor = self.grid[nx, ny]
                    
                    if 0 < calculate_similarity(agent, neighbor, self.F) < 1:
                        return False
        return True
    
    