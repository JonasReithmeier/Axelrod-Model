import numpy as np
from numba import njit

@njit
def calculate_similarity(agent_a, agent_b, F):
    shared = 0
    for i in range(F):
        if agent_a[i] == agent_b[i]:
            shared += 1
    return shared 


@njit
def check_frozen(grid, W, H, F):
    """Exhaustively checks if the grid is completely frozen."""
    for x in range(W):
        for y in range(H):
            # Check Right
            nx = (x + 1) % W
            sim = calculate_similarity(grid[x, y], grid[nx, y], F)
            if 0 < sim < F: return False
            
            # Check Down
            ny = (y + 1) % H
            sim = calculate_similarity(grid[x, y], grid[x, ny], F)
            if 0 < sim < F: return False
            
    return True

@njit
def run_steps(grid, max_steps, F, W, H, updates_since_change, threshold, rng):
    """Runs the simulation entirely inside compiled C-code with explicit RNG."""
    steps_done = 0
    diff_indices = np.empty(F, dtype=np.int32)
    
    while steps_done < max_steps:
        # Use the explicit RNG passed from Python
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        
        direction = rng.integers(0, 4)
        nx, ny = x, y
        if direction == 0: nx = (x + 1) % W
        elif direction == 1: nx = (x - 1) % W
        elif direction == 2: ny = (y + 1) % H
        else: ny = (y - 1) % H
            
        shared = 0
        diff_count = 0
        for i in range(F):
            if grid[x, y, i] == grid[nx, ny, i]:
                shared += 1
            else:
                diff_indices[diff_count] = i
                diff_count += 1
                
        # Interaction Rule
        if 0 < shared < F:
            prob = shared / F
            if rng.random() < prob:
                target_trait = diff_indices[rng.integers(0, diff_count)]
                grid[x, y, target_trait] = grid[nx, ny, target_trait]

            updates_since_change = 0

        else:
            updates_since_change += 1
            
        steps_done += 1
        
        # Check frozen condition
        if updates_since_change >= threshold:
            if check_frozen(grid, W, H, F):
                return steps_done, updates_since_change, True
            else:
                updates_since_change = 0
                
    return steps_done, updates_since_change, False

@njit
def get_cluster_metrics(grid, W, H, F):
    """Fast BFS algorithm to calculate cluster metrics."""
    visited = np.zeros((W, H), dtype=np.bool_)
    
    # Pre-allocate queue for BFS (max size is W*H)
    queue_x = np.empty(W * H, dtype=np.int32)
    queue_y = np.empty(W * H, dtype=np.int32)
    
    max_size = 0
    total_size = 0
    num_clusters = 0
    
    for i in range(W):
        for j in range(H):
            if not visited[i, j]:
                # Start new cluster BFS
                size = 0
                head = 0
                tail = 0
                
                # Push start node
                queue_x[tail] = i
                queue_y[tail] = j
                tail += 1
                visited[i, j] = True
                
                while head < tail:
                    cx = queue_x[head]
                    cy = queue_y[head]
                    head += 1
                    size += 1
                    
                    # Check 4 neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = (cx + dx) % W, (cy + dy) % H
                        if not visited[nx, ny]:
                            # Are they culturally identical?
                            identical = True
                            for f in range(F):
                                if grid[cx, cy, f] != grid[nx, ny, f]:
                                    identical = False
                                    break
                            
                            if identical:
                                visited[nx, ny] = True
                                queue_x[tail] = nx
                                queue_y[tail] = ny
                                tail += 1
                                
                if size > max_size:
                    max_size = size
                total_size += size
                num_clusters += 1
                
    N = W * H
    s_max = max_size / N
    s_mean = (total_size / num_clusters) / N
    if (total_size != W*H):
        ValueError("total_size != W*H")
    return s_max, s_mean