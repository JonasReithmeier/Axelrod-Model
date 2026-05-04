import numpy as np
import random
from .agent import AxelrodAgent
from .utils import calculate_similarity, get_different_traits

class AxelrodModel_regularLattice:
    def __init__(self, width=None, height=None, F=None, q=None, seed=None, config=None):
        """
        Initializes the model. Arguments passed directly override the config dictionary.
        """
        # Fallback logic: Use config if provided, otherwise empty dict
        cfg = config if config is not None else {}

        # Assign attributes: Priority: Argument > Config > Default
        self.width = width if width is not None else cfg.get('width', 10)
        self.height = height if height is not None else cfg.get('height', 10)
        self.F = F if F is not None else cfg.get('features', 5)
        self.q = q if q is not None else cfg.get('traits', 10)

        # 3. Validation
        if None in [self.width, self.height, self.F, self.q]:
            raise ValueError("Missing parameters! Provide either explicit arguments or a valid config.")

        # 4. Reproducibility 
        self.seed = seed if seed is not None else cfg.get('seed')
        if self.seed is None:
            raise ValueError("Missing seed!")
        else:
            # Create local, isolated RNG instances (two different RNG for max effectivity in their specific usecases)
            self.array_rng = np.random.default_rng(self.seed) # NumPy RNG (random number generator) for random arrays 
            self.scalar_rng = random.Random(self.seed) # Standard Python RNG for random scalars/ random list choices


        # Internal state for "Frozen" logic
        self.updates_since_last_change = 0

        # Create a 2D grid of Agent objects
        self.grid = np.empty((self.width, self.height), dtype=object)
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x, y] = AxelrodAgent((x, y), self.F, self.q, self.array_rng)

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
        else:
            self.updates_since_last_change += 1

    
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