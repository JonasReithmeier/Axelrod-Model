import numpy as np
from numba import njit

@njit
def check_frozen_sw(grid, edge_ptrs, edges, N, F):
    """Checks if the Small World graph is frozen."""
    for i in range(N):
        start = edge_ptrs[i]
        end = edge_ptrs[i + 1]
        
        # Iterate over all neighbors of agent i
        for e in range(start, end):
            neighbor = edges[e]
            # Only check each pair once (i < neighbor)
            if i < neighbor:
                shared = 0
                for f in range(F):
                    if grid[i, f] == grid[neighbor, f]:
                        shared += 1
                if 0 < shared < F:
                    return False
    return True

@njit
def run_steps_sw(grid, edge_ptrs, edges, max_steps, F, N, updates_since_change, threshold, rng):
    steps_done = 0
    diff_indices = np.empty(F, dtype=np.int32)
    
    while steps_done < max_steps:
        # Pick random agent
        agent_idx = int(rng.random() * N)
        
        # Get its neighbors from the CSR array
        start = edge_ptrs[agent_idx]
        end = edge_ptrs[agent_idx + 1]
        degree = end - start
        
        if degree == 0:  # Safety check (rare but possible in high rewiring)
            steps_done += 1
            continue
            
        # Pick random neighbor
        neighbor_idx = edges[start + int(rng.random() * degree)]
            
        shared = 0
        diff_count = 0
        
        for i in range(F):
            if grid[agent_idx, i] == grid[neighbor_idx, i]:
                shared += 1
            else:
                diff_indices[diff_count] = i
                diff_count += 1
                
        # Interaction Rule
        if 0 < shared < F:
            if (rng.random() * F) < shared:
                target_idx = int(rng.random() * diff_count)
                target_trait = diff_indices[target_idx]
                grid[agent_idx, target_trait] = grid[neighbor_idx, target_trait]
                updates_since_change = 0
            else:
                updates_since_change += 1
        else:
            updates_since_change += 1
            
        steps_done += 1
        
        if updates_since_change >= threshold:
            if check_frozen_sw(grid, edge_ptrs, edges, N, F):
                return steps_done, updates_since_change, True
            else:
                updates_since_change = 0
                
    return steps_done, updates_since_change, False

@njit
def get_cluster_metrics_sw(grid, edge_ptrs, edges, N, F):
    visited = np.zeros(N, dtype=np.bool_)
    queue = np.empty(N, dtype=np.int32)
    
    max_size = 0
    total_size = 0
    num_clusters = 0
    
    for i in range(N):
        if not visited[i]:
            size = 0
            head = 0
            tail = 0
            
            queue[tail] = i
            tail += 1
            visited[i] = True
            
            while head < tail:
                current = queue[head]
                head += 1
                size += 1
                
                start = edge_ptrs[current]
                end = edge_ptrs[current + 1]
                
                # Check all neighbors
                for e in range(start, end):
                    nxt = edges[e]
                    if not visited[nxt]:
                        identical = True
                        for f in range(F):
                            if grid[current, f] != grid[nxt, f]:
                                identical = False
                                break
                        
                        if identical:
                            visited[nxt] = True
                            queue[tail] = nxt
                            tail += 1
                            
            if size > max_size:
                max_size = size
            total_size += size
            num_clusters += 1
            
    s_max = max_size / N
    s_mean = (total_size / num_clusters) / N
    return s_max, s_mean

@njit
def compute_clustering_coefficient(edge_ptrs, edges, N):
    """Calculates the average clustering coefficient C(p) of the network."""
    total_C = 0.0
    for i in range(N):
        start_i = edge_ptrs[i]
        end_i = edge_ptrs[i + 1]
        degree_i = end_i - start_i
        
        if degree_i < 2:
            continue
            
        existing_edges = 0
        # Check all pairs of neighbors of node i
        for u_idx in range(start_i, end_i):
            u = edges[u_idx]
            start_u = edge_ptrs[u]
            end_u = edge_ptrs[u + 1]
            
            for v_idx in range(start_i, end_i):
                v = edges[v_idx]
                if u < v:  # Only check each pair once
                    # Is v a neighbor of u?
                    for w_idx in range(start_u, end_u):
                        if edges[w_idx] == v:
                            existing_edges += 1
                            break
                            
        max_possible_edges = (degree_i * (degree_i - 1)) / 2.0
        total_C += existing_edges / max_possible_edges
        
    return total_C / N

@njit
def compute_characteristic_path_length(edge_ptrs, edges, N):
    """Calculates the average shortest path L(p) using all-pairs Breadth First Search."""
    total_path_length = 0
    reachable_pairs = 0
    
    # Pre-allocate queue and distance array to reuse across all nodes
    queue = np.empty(N, dtype=np.int32)
    dist = np.empty(N, dtype=np.int32)
    
    for source in range(N):
        # Reset distances to -1 (unvisited)
        for i in range(N):
            dist[i] = -1
        
        head = 0
        tail = 0
        
        # Start BFS from source
        queue[tail] = source
        tail += 1
        dist[source] = 0
        
        while head < tail:
            curr = queue[head]
            head += 1
            
            start = edge_ptrs[curr]
            end = edge_ptrs[curr + 1]
            
            for i in range(start, end):
                nxt = edges[i]
                if dist[nxt] == -1:  # If unvisited
                    dist[nxt] = dist[curr] + 1
                    queue[tail] = nxt
                    tail += 1
                    
                    # Accumulate distance metrics
                    total_path_length += dist[nxt]
                    reachable_pairs += 1
                    
    if reachable_pairs == 0:
        return 0.0
    return total_path_length / reachable_pairs