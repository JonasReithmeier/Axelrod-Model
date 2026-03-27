import numpy as np
import random
from .agent import AxelrodAgent
from .utils import calculate_similarity, get_different_traits

class AxelrodModel:
    def __init__(self, config):
        self.width = config['width']
        self.height = config['height']
        self.F = config['features']
        self.q = config['traits']
        
        # Create a 2D grid of Agent objects
        self.grid = np.empty((self.width, self.height), dtype=object)
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x, y] = AxelrodAgent((x, y), self.F, self.q)

    def get_random_neighbor(self, x, y):
        """Finds a neighbor using Von Neumann neighborhood (Up, Down, Left, Right)"""
        neighbors = []
        # Torus (wrap-around) logic: % self.width
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % self.width, (y + dy) % self.height
            neighbors.append(self.grid[nx, ny])
        return random.choice(neighbors)

    def step(self):
        """A single iteration of the Axelrod algorithm"""
        # 1. Randomly pick an active agent
        x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
        agent_a = self.grid[x, y]

        # 2. Randomly pick one of its neighbors
        agent_b = self.get_random_neighbor(x, y)

        # 3. Calculate similarity
        sim = calculate_similarity(agent_a, agent_b, self.F)

        # 4. Interaction Rule
        # If similarity is 0 or 1, nothing happens.
        # Otherwise, they interact with probability equal to their similarity.
        if 0 < sim < 1:
            if random.random() < sim:
                # Agent A adopts one trait from B that they don't share
                diff_indices = get_different_traits(agent_a, agent_b)
                target_trait = random.choice(diff_indices)
                agent_a.culture[target_trait] = agent_b.culture[target_trait]