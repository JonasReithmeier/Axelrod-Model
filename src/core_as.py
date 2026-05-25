import numpy as np
from numba import njit

@njit
def linregress(y):
    """Simple linear regression helper to keep everything inside njit."""
    n = len(y)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    for i in range(n):
        x = float(i)
        sum_x += x
        sum_y += y[i]
        sum_xy += x * y[i]
        sum_xx += x * x
    
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    denominator = sum_xx - n * mean_x * mean_x
    if denominator == 0:
        return 0.0
    return (sum_xy - n * mean_x * mean_y) / denominator

@njit
def check_constant_as(grid, L, F, T, num_empty):
    """Exhaustive check: The model is constant if NO ONE wants to move, and NO ONE can interact."""
    for x in range(L):
        for y in range(L):
            if grid[x, y, 0] == -1: 
                continue
                
            valid_neighbors = 0
            diff_sum = 0.0
            can_interact = False
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = (x + dx) % L, (y + dy) % L
                
                if grid[nx, ny, 0] != -1:
                    valid_neighbors += 1
                    shared = 0
                    for f in range(F):
                        if grid[x, y, f] == grid[nx, ny, f]:
                            shared += 1
                            
                    diff_sum += (F - shared) / F
                    
                    if 0 < shared < F:
                        can_interact = True
            
            if valid_neighbors > 0:
                avg_diff = diff_sum / valid_neighbors
                if (avg_diff > T) and (num_empty > 0):   
                    return False 
                    
            if can_interact:
                return False
                
    return True

@njit
def calculate_mobility_as(grid, L, F, T, num_empty):
    if num_empty == 0:
        return 0.0
        
    movers = 0
    active_agents = 0
    
    for x in range(L):
        for y in range(L):
            if grid[x, y, 0] == -1:
                continue
            active_agents += 1
            
            valid_neighbors = 0
            diff_sum = 0.0
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = (x + dx) % L, (y + dy) % L
                if grid[nx, ny, 0] != -1:
                    valid_neighbors += 1
                    shared = 0
                    for f in range(F):
                        if grid[x, y, f] == grid[nx, ny, f]:
                            shared += 1
                    diff_sum += (F - shared) / F
            
            if valid_neighbors == 0:
                movers += 1
            else:
                avg_diff = diff_sum / valid_neighbors
                if avg_diff > T:
                    movers += 1
                    
    if active_agents == 0: return 0.0
    return movers / active_agents

@njit
def perform_agent_step(grid, L, F, empty_locs, num_empty, T, x, y, updates, rng, diff_indices, valid_neighbors_coords):
    """Executes a single agent step and returns updated 'updates' counter."""
    if grid[x, y, 0] == -1:
        return updates + 1
        
    valid_n_count = 0
    diff_sum = 0.0
    
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = (x + dx) % L, (y + dy) % L
        if grid[nx, ny, 0] != -1:
            valid_neighbors_coords[valid_n_count, 0] = nx
            valid_neighbors_coords[valid_n_count, 1] = ny
            valid_n_count += 1
            
            shared = 0
            for f in range(F):
                if grid[x, y, f] == grid[nx, ny, f]:
                    shared += 1
            diff_sum += (F - shared) / F
            
    avg_diff = 0.0
    if valid_n_count > 0:
        avg_diff = diff_sum / valid_n_count
        
    if (avg_diff > T and num_empty > 0) or (valid_n_count == 0 and num_empty > 0):
        e_idx = rng.integers(0, num_empty)
        ex, ey = empty_locs[e_idx, 0], empty_locs[e_idx, 1]
        
        for f in range(F):
            grid[ex, ey, f] = grid[x, y, f]
            grid[x, y, f] = -1
            
        empty_locs[e_idx, 0] = x
        empty_locs[e_idx, 1] = y
        return updates + 1
        
    else:
        if valid_n_count > 0:
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
                    return 0 # An interaction successfully updated the grid
                else:
                    return updates + 1
            else:
                return updates + 1
        else:
            return updates + 1

@njit
def run_mcs_chunk(grid, L, F, empty_locs, num_empty, T, max_mcs, updates, threshold, rng):
    """Executes pure sweeps without dropping into python."""
    N_cells = L * L
    diff_indices = np.empty(F, dtype=np.int32)
    valid_neighbors_coords = np.empty((4, 2), dtype=np.int32)
    
    for mcs in range(max_mcs):
        for _ in range(N_cells):
            x, y = rng.integers(0, L), rng.integers(0, L)
            updates = perform_agent_step(grid, L, F, empty_locs, num_empty, T, x, y, updates, rng, diff_indices, valid_neighbors_coords)
            
        if updates >= threshold:
            return mcs + 1, updates, True
            
    return max_mcs, updates, False

@njit
def run_monitoring_phase(grid, L, F, empty_locs, num_empty, T, rng, updates):
    """Runs 50k sweeps computing stability entirely on the compiled side."""
    monitoring_sweeps = 50000
    N_cells = L * L
    m_history = np.zeros(monitoring_sweeps, dtype=np.float64)
    
    diff_indices = np.empty(F, dtype=np.int32)
    valid_neighbors_coords = np.empty((4, 2), dtype=np.int32)
    
    for sweep in range(monitoring_sweeps):
        for _ in range(N_cells):
            x, y = rng.integers(0, L), rng.integers(0, L)
            updates = perform_agent_step(grid, L, F, empty_locs, num_empty, T, x, y, updates, rng, diff_indices, valid_neighbors_coords)
            
        m_history[sweep] = calculate_mobility_as(grid, L, F, T, num_empty)
        
    slope = linregress(m_history)
    final_m = np.mean(m_history)
    
    return monitoring_sweeps, final_m, slope, updates

@njit
def get_cluster_metrics_as(grid, L, F):
    visited = np.zeros((L, L), dtype=np.bool_)
    queue_x = np.empty(L * L, dtype=np.int32)
    queue_y = np.empty(L * L, dtype=np.int32)
    
    max_size = 0
    total_size = 0
    num_clusters = 0
    active_agents = 0
    
    for i in range(L):
        for j in range(L):
            if grid[i, j, 0] == -1: 
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
                        nx, ny = (cx + dx) % L, (cy + dy) % L
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
    s_max = max_size / active_agents
    s_mean = (total_size / num_clusters) / active_agents
    return s_max, s_mean