import numpy as np
from numba import njit

@njit
def linregress(y):
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
def check_constant_as(grid, N_cells, F, T, num_empty, edge_ptrs, edges):
    for node in range(N_cells):
        if grid[node, 0] == -1: 
            continue
            
        valid_neighbors = 0
        diff_sum_F = 0
        can_interact = False
        
        start = edge_ptrs[node]
        end = edge_ptrs[node+1]
        
        for e in range(start, end):
            n_node = edges[e]
            
            if grid[n_node, 0] != -1:
                valid_neighbors += 1
                shared = 0
                for f in range(F):
                    if grid[node, f] == grid[n_node, f]:
                        shared += 1
                        
                diff_sum_F += (F - shared)
                
                if 0 < shared < F:
                    can_interact = True
        
        if valid_neighbors > 0:
            # Eliminated division: Mathematical equivalent for avg_diff > T
            if (diff_sum_F > T * F * valid_neighbors) and (num_empty > 0):   
                return False 
                
        if can_interact:
            return False
                
    return True

@njit
def calculate_mobility_as(grid, N_cells, F, T, num_empty, edge_ptrs, edges):
    if num_empty == 0:
        return 0.0
        
    movers = 0
    active_agents = 0
    
    for node in range(N_cells):
        if grid[node, 0] == -1:
            continue
        active_agents += 1
        
        valid_neighbors = 0
        diff_sum_F = 0
        
        start = edge_ptrs[node]
        end = edge_ptrs[node+1]
        
        for e in range(start, end):
            n_node = edges[e]
            if grid[n_node, 0] != -1:
                valid_neighbors += 1
                shared = 0
                for f in range(F):
                    if grid[node, f] == grid[n_node, f]:
                        shared += 1
                diff_sum_F += (F - shared)
        
        if valid_neighbors == 0:
            movers += 1
        else:
            if diff_sum_F > T * F * valid_neighbors:
                movers += 1
                
    if active_agents == 0: return 0.0
    return movers / active_agents

@njit
def perform_agent_step(grid, F, empty_locs, num_empty, T, node, updates,
                       rng, diff_indices, valid_neighbors_coords,
                       edge_ptrs, edges):
    if grid[node, 0] == -1:
        return updates + 1

    start = edge_ptrs[node]
    end   = edge_ptrs[node + 1]

    # Collect occupied neighbours
    valid_n_count = 0
    for e in range(start, end):
        n_node = edges[e]
        if grid[n_node, 0] != -1:
            valid_neighbors_coords[valid_n_count] = n_node
            valid_n_count += 1

    # Isolated agent → moves with certainty 
    if valid_n_count == 0:
        if num_empty > 0:
            e_idx      = rng.integers(0, num_empty)
            empty_node = empty_locs[e_idx]
            for f in range(F):
                grid[empty_node, f] = grid[node, f]
                grid[node, f]       = -1
            empty_locs[e_idx] = node
        return updates + 1

    # Step 1: Axelrod interaction with one random neighbour
    n_choice = rng.integers(0, valid_n_count)
    n_node   = valid_neighbors_coords[n_choice]

    shared     = 0
    diff_count = 0
    for f in range(F):
        if grid[node, f] == grid[n_node, f]:
            shared += 1
        else:
            diff_indices[diff_count] = f
            diff_count += 1

    interacted = False
    if 0 < shared < F:
        if (rng.random() * F) < shared:
            target_trait = diff_indices[rng.integers(0, diff_count)]
            grid[node, target_trait] = grid[n_node, target_trait]
            interacted = True
            return 0   

    # Step 2: Schelling mobility — only if no imitation AND chosen
    if not interacted and shared != F:
        # Compute mean overlap over ALL occupied neighbours
        diff_sum_F = 0
        for i in range(valid_n_count):
            nb = valid_neighbors_coords[i]
            sh = 0
            for f in range(F):
                if grid[node, f] == grid[nb, f]:
                    sh += 1
            diff_sum_F += (F - sh)

        if diff_sum_F > T * F * valid_n_count and num_empty > 0:
            e_idx      = rng.integers(0, num_empty)
            empty_node = empty_locs[e_idx]
            for f in range(F):
                grid[empty_node, f] = grid[node, f]
                grid[node, f]       = -1
            empty_locs[e_idx] = node

    return updates + 1
@njit
def run_mcs_chunk(grid, N_cells, F, empty_locs, num_empty, T, max_mcs, updates, threshold, rng, edge_ptrs, edges, max_degree):
    diff_indices = np.empty(F, dtype=np.int32)
    valid_neighbors_coords = np.empty(max_degree, dtype=np.int32)
    
    for mcs in range(max_mcs):
        for _ in range(N_cells):
            node = rng.integers(0, N_cells) # Half the rng calls previously done since we use flat 1D space
            updates = perform_agent_step(grid, F, empty_locs, num_empty, T, node, updates, rng, diff_indices, valid_neighbors_coords, edge_ptrs, edges)
            
        if updates >= threshold:
            return mcs + 1, updates, True
            
    return max_mcs, updates, False

@njit
def run_monitoring_phase(grid, N_cells, F, empty_locs, num_empty, T, rng, updates, edge_ptrs, edges, max_degree):
    monitoring_sweeps = 50000
    m_history = np.zeros(monitoring_sweeps, dtype=np.float64)
    
    diff_indices = np.empty(F, dtype=np.int32)
    valid_neighbors_coords = np.empty(max_degree, dtype=np.int32)
    
    for sweep in range(monitoring_sweeps):
        for _ in range(N_cells):
            node = rng.integers(0, N_cells)
            updates = perform_agent_step(grid, F, empty_locs, num_empty, T, node, updates, rng, diff_indices, valid_neighbors_coords, edge_ptrs, edges)
            
        m_history[sweep] = calculate_mobility_as(grid, N_cells, F, T, num_empty, edge_ptrs, edges)
        
    slope = linregress(m_history)
    final_m = np.mean(m_history)
    
    return monitoring_sweeps, final_m, slope, updates

@njit
def get_cluster_metrics_as(grid, N_cells, F, edge_ptrs, edges):
    visited = np.zeros(N_cells, dtype=np.bool_)
    queue = np.empty(N_cells, dtype=np.int32)
    
    max_size = 0
    total_size = 0
    num_clusters = 0
    active_agents = 0
    
    for node in range(N_cells):
        if grid[node, 0] == -1: 
            continue
        active_agents += 1
            
        if not visited[node]:
            size = 0
            head = 0
            tail = 0
            
            queue[tail] = node
            tail += 1
            visited[node] = True
            
            while head < tail:
                curr = queue[head]
                head += 1
                size += 1
                
                start = edge_ptrs[curr]
                end = edge_ptrs[curr+1]
                
                for e in range(start, end):
                    n_node = edges[e]
                    if grid[n_node, 0] != -1 and not visited[n_node]:
                        identical = True
                        for f in range(F):
                            if grid[curr, f] != grid[n_node, f]:
                                identical = False
                                break
                        
                        if identical:
                            visited[n_node] = True
                            queue[tail] = n_node
                            tail += 1
                            
            if size > max_size: max_size = size
            total_size += size
            num_clusters += 1
            
    if active_agents == 0: return 0.0, 0.0
    s_max = max_size / active_agents
    s_mean = (total_size / num_clusters) / active_agents
    return s_max, s_mean