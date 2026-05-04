import numpy as np

#model metrics
def calculate_similarity(agent_a, agent_b, F):
    """Returns the fraction of shared cultural traits (0.0 to 1.0)"""
    shared = np.sum(agent_a.culture == agent_b.culture)
    return shared / F

def get_different_traits(agent_a, agent_b):
    """Returns a list of indices where the traits of A and B differ"""
    indices = np.where(agent_a.culture != agent_b.culture)[0]
    return indices

# Clustersize for phase transition analysis
def largest_cluster_normalized(model, N=None):
    """
    Returns the size of the largest cultural cluster 
    divided by the total number of agents (N).
    This is the standard 'Order Parameter' for the Axelrod model.
    """
    N = N if N is not None else model.N
    sizes = _get_cluster_sizes(model)
    if not sizes:
        return -1
    return max(sizes) / N

def average_cluster_normalized(model, N=None):
    """
    Returns the average size of all cultural clusters
    normalized by the total number of agents (N).
    """
    N = N if N is not None else model.N
    sizes = _get_cluster_sizes(model)
    if not sizes:
        return -1
    return np.mean(sizes) / N

def _get_cluster_sizes(model):
    """
    Helper function using BFS to find all cluster sizes.
    Accounts for periodic boundary conditions (Torus).
    """
    width, height = model.width, model.height
    visited = np.zeros((width, height), dtype=bool)
    cluster_sizes = []

    # Optimization: Pre-calculate culture hashes to avoid array comparisons in the loop
    # This is much faster than np.array_equal()
    culture_ids = np.array([[hash(tuple(model.grid[x, y].culture)) 
                             for y in range(height)] 
                            for x in range(width)])

    for x in range(width):
        for y in range(height):
            if not visited[x, y]:
                # Start a new BFS to find the full extent of this cluster
                size = 0
                target_id = culture_ids[x, y]
                queue = [(x, y)]
                visited[x, y] = True
                
                while queue:
                    cx, cy = queue.pop(0)
                    size += 1
                    
                    # Check Von Neumann neighbors (Torus wrap-around)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = (cx + dx) % width, (cy + dy) % height
                        
                        if not visited[nx, ny] and culture_ids[nx, ny] == target_id:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                cluster_sizes.append(size)
    
    return cluster_sizes



