import numpy as np
from numba import njit

@njit
def check_frozen_as(grid, W, H, F, T, empty_locs, num_empty):
    """Exhaustive check: The model is frozen if NO ONE wants to move, and NO ONE can interact."""
    for x in range(W):
        for y in range(H):
            if grid[x, y, 0] == -1: # Skip empty cells
                continue
                
            valid_neighbors = 0
            diff_sum = 0.0
            can_interact = False
            
            # Check 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = (x + dx) % W, (y + dy) % H
                
                if grid[nx, ny, 0] != -1:
                    valid_neighbors += 1
                    shared = 0
                    for f in range(F):
                        if grid[x, y, f] == grid[nx, ny, f]:
                            shared += 1
                            
                    diff_sum += (F - shared) / F
                    
                    if 0 < shared < F:
                        can_interact = True
            
            # 1. Can it move?
            if valid_neighbors > 0:
                avg_diff = diff_sum / valid_neighbors
                if (1 > avg_diff > T) and (num_empty > 0):  
                    return False # Wants to move, therefore not frozen
                    
            # 2. Can it interact?
            if can_interact:
                return False
                
    return True

@njit
def run_steps_as(grid, W, H, F, empty_locs, num_empty, T, max_steps, updates, threshold, rng):
    steps_done = 0
    diff_indices = np.empty(F, dtype=np.int32)
    valid_neighbors_coords = np.empty((4, 2), dtype=np.int32)
    
    while steps_done < max_steps:
        x, y = rng.integers(0, W), rng.integers(0, H)
        
        if grid[x, y, 0] == -1: # Empty cell selected
            steps_done += 1
            updates += 1
            continue
            
        valid_n_count = 0
        diff_sum = 0.0
        
        # Gather neighbors and calculate dissatisfaction
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % W, (y + dy) % H
            if grid[nx, ny, 0] != -1:
                valid_neighbors_coords[valid_n_count, 0] = nx
                valid_neighbors_coords[valid_n_count, 1] = ny
                valid_n_count += 1
                
                shared = 0
                for f in range(F):
                    if grid[x, y, f] == grid[nx, ny, f]:
                        shared += 1
                diff_sum += (F - shared) / F
                
        # Calculate Average Difference
        avg_diff = 0.0
        if valid_n_count > 0:
            avg_diff = diff_sum / valid_n_count
            
        # SCHELLING PHASE: Move if unhappy or completely isolated
        if (avg_diff > T and num_empty > 0) or (valid_n_count == 0):
            # Pick random empty site
            e_idx = rng.integers(0, num_empty)
            ex, ey = empty_locs[e_idx, 0], empty_locs[e_idx, 1]
            
            # Teleport agent
            for f in range(F):
                grid[ex, ey, f] = grid[x, y, f]
                grid[x, y, f] = -1
                
            # Update empty lookup table
            empty_locs[e_idx, 0] = x
            empty_locs[e_idx, 1] = y
            
            updates = 0 # State changed
            
        # AXELROD PHASE: Interact if happy (and has neighbors)
        elif valid_n_count > 0:
            n_choice = rng.integers(0, valid_n_count)
            nx, ny = valid_neighbors_coords[n_choice, 0], valid_neighbors_coords[n_choice, 1]
            
            shared = 0
            diff_count = 0
            for f in range(F):
                if grid[x, y, f] == grid[nx, ny, f]:
                    shared += 1
                else:
                    diff_indices[diff_count] = f
                    diff_count += 1
                    
            if 0 < shared < F:
                if (rng.random() * F) < shared:
                    target_trait = diff_indices[rng.integers(0, diff_count)]
                    grid[x, y, target_trait] = grid[nx, ny, target_trait]
                    updates = 0
                else:
                    updates += 1
            else:
                updates += 1
        else:
            updates += 1 # No neighbors, stuck
            
        steps_done += 1
        
        # Frozen check
        if updates >= threshold:
            if check_frozen_as(grid, W, H, F, T, empty_locs, num_empty):
                return steps_done, updates, True
            else:
                updates = 0
                
    return steps_done, updates, False

@njit
def get_cluster_metrics_as(grid, W, H, F):
    visited = np.zeros((W, H), dtype=np.bool_)
    queue_x = np.empty(W * H, dtype=np.int32)
    queue_y = np.empty(W * H, dtype=np.int32)
    
    max_size = 0
    total_size = 0
    num_clusters = 0
    active_agents = 0
    
    for i in range(W):
        for j in range(H):
            if grid[i, j, 0] == -1: # Skip empty space
                continue
            active_agents += 1
                
            if not visited[i, j]:
                size = 0
                head = 0
                tail = 0
                
                queue_x[tail] = i
                queue_y[tail] = j
                tail += 1
                visited[i, j] = True
                
                while head < tail:
                    cx = queue_x[head]
                    cy = queue_y[head]
                    head += 1
                    size += 1
                    
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = (cx + dx) % W, (cy + dy) % H
                        if grid[nx, ny, 0] != -1 and not visited[nx, ny]:
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
                                
                if size > max_size: max_size = size
                total_size += size
                num_clusters += 1
                
    if active_agents == 0: return 0.0, 0.0
    
    # Normalize by number of ACTIVE agents, not grid cells
    s_max = max_size / active_agents
    s_mean = (total_size / num_clusters) / active_agents
    return s_max, s_mean