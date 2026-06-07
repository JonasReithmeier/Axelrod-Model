import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Weight function dispatch (int flag, fully compiled)
# ---------------------------------------------------------------------------
# Mode 0: linear         w(d) = alpha * d
# Mode 1: quadratic      w(d) = alpha * d * |d|          (signed square)
# Mode 2: biphasic       positive small d, negative mid, positive large d
# Mode 3: attraction     w(d) = -alpha * |d|             (always pulls together)
# d = dev_i - dev_j  (i's perspective; positive means i richer than j)

@njit(inline='always')
def weight(d, mode, alpha):
    if mode == 0:
        return alpha * d
    elif mode == 1:
        if d >= 0.0:
            return alpha * d * d
        else:
            return -alpha * d * d
    elif mode == 2:
        # Biphasic: concurrence with equals, friction with moderate diff,
        # fascination with very different. Tuned so zero-crossings at ~0.25 and ~0.75.
        # w(d) = alpha * d * (d - 0.25) * (d - 0.75) scaled to [-1, 1] roughly
        return alpha * 8.0 * d * (d - 0.25) * (d - 0.75)
    else:  # mode == 3: pure attraction across hierarchy
        if d >= 0.0:
            return -alpha * d
        else:
            return alpha * d

@njit(inline='always')
def total_dissatisfaction(shared, F, dev_i, dev_j, weight_mode, alpha):
    cultural = (F - shared) / F
    dev_diff = dev_i - dev_j
    raw = cultural + weight(dev_diff, weight_mode, alpha)
    # Clamp to [0, 1]
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw

# ---------------------------------------------------------------------------
# linregress (pure njit, no scipy)
# ---------------------------------------------------------------------------

@njit
def linregress_slope(y):
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
    denom = sum_xx - n * mean_x * mean_x
    if denom == 0.0:
        return 0.0
    return (sum_xy - n * mean_x * mean_y) / denom

# ---------------------------------------------------------------------------
# CSR helpers for mutable graph
# ---------------------------------------------------------------------------

@njit(inline='always')
def get_degree(edge_ptrs, padded_edges, node, max_degree):
    """Count actual (non-sentinel) edges for node."""
    start = node * max_degree
    count = 0
    for k in range(max_degree):
        if padded_edges[start + k] != -1:
            count += 1
    return count

@njit(inline='always')
def has_edge(adj_matrix, i, j, N):
    """O(1) connectivity check via boolean adjacency matrix."""
    return adj_matrix[i * N + j]
    

@njit(inline='always')
def add_edge(padded_edges, adj_matrix, i, j, N, max_degree):
    """Add undirected edge i-j. Returns False if max_degree exceeded."""
    # Find free slot for i
    start_i = i * max_degree
    slot_i = -1
    for k in range(max_degree):
        if padded_edges[start_i + k] == -1:
            slot_i = k
            break
    if slot_i == -1:
        return False  # i is full

    start_j = j * max_degree
    slot_j = -1
    for k in range(max_degree):
        if padded_edges[start_j + k] == -1:
            slot_j = k
            break
    if slot_j == -1:
        return False  # j is full

    padded_edges[start_i + slot_i] = j
    padded_edges[start_j + slot_j] = i
    adj_matrix[i * N + j] = True
    adj_matrix[j * N + i] = True
    return True

@njit(inline='always')
def remove_edge(padded_edges, adj_matrix, i, j, N, max_degree):
    """Remove undirected edge i-j."""
    start_i = i * max_degree
    for k in range(max_degree):
        if padded_edges[start_i + k] == j:
            padded_edges[start_i + k] = -1
            break
    start_j = j * max_degree
    for k in range(max_degree):
        if padded_edges[start_j + k] == i:
            padded_edges[start_j + k] = -1
            break
    adj_matrix[i * N + j] = False
    adj_matrix[j * N + i] = False

# ---------------------------------------------------------------------------
# Freeze check
# ---------------------------------------------------------------------------

@njit
def check_frozen_cultural(grid, padded_edges, N, F, max_degree):
    """True if no adjacent pair has 0 < shared < F (no possible culture interaction)."""
    for i in range(N):
        start = i * max_degree
        for k in range(max_degree):
            j = padded_edges[start + k]
            if j == -1:
                continue
            if i < j:  # each pair once
                shared = 0
                for f in range(F):
                    if grid[i, f] == grid[j, f]:
                        shared += 1
                if 0 < shared < F:
                    return False
    return True

# ---------------------------------------------------------------------------
# Capacity Helper
# ---------------------------------------------------------------------------

@njit(inline='always')
def is_node_full(padded_edges, node, max_degree):
    """Returns True if the node has no free neighbor slots."""
    start = node * max_degree
    for k in range(max_degree):
        if padded_edges[start + k] == -1:
            return False  # Found at least one free slot
    return True

# ---------------------------------------------------------------------------
# Reorganized Step Kernel
# ---------------------------------------------------------------------------

@njit
def step_dev_sw(grid, dev, padded_edges, adj_matrix, N, F, weight_mode, alpha,
                dis_threshold, rng, max_degree,
                neighbor_buf, dis_buf):
    """
    One elementary step with modified fallback logic.
    Returns (rewired_flag, interacted_flag, capacity_exceeded_flag).
    """
    agent_i = int(rng.random() * N)

    # 1. Collect neighbors and their dissatisfactions
    start = agent_i * max_degree
    n_neighbors = 0
    for k in range(max_degree):
        j = padded_edges[start + k]
        if j == -1:
            continue
        shared = 0
        for f in range(F):
            if grid[agent_i, f] == grid[j, f]:
                shared += 1
        d = total_dissatisfaction(shared, F, dev[agent_i], dev[j], weight_mode, alpha)
        neighbor_buf[n_neighbors] = j
        dis_buf[n_neighbors] = d
        n_neighbors += 1

    # --- FIRST: No neighbors logic ---
    if n_neighbors == 0:
        new_j = -1
        # Look for a random available non-full agent (up to 2*N attempts)
        for _ in range(N * 2):
            candidate = int(rng.random() * N)
            if candidate == agent_i:
                continue
            if has_edge(adj_matrix, agent_i, candidate, N):
                continue
            if is_node_full(padded_edges, candidate, max_degree):
                continue
            new_j = candidate
            break

        if new_j != -1:
            add_edge(padded_edges, adj_matrix, agent_i, new_j, N, max_degree)
            return True, False, False  # Rewired successfully
        else:
            return False, False, True  # No available node found (capacity exceeded)

    # --- SECOND: Try Axelrod interaction if differences exist ---
    choice_k = int(rng.random() * n_neighbors)
    agent_j = neighbor_buf[choice_k]
    dis_ij = dis_buf[choice_k]

    # Find differing traits
    diff_count = 0
    diff_buf_local = np.empty(F, dtype=np.int32)
    for f in range(F):
        if grid[agent_i, f] != grid[agent_j, f]:
            diff_buf_local[diff_count] = f
            diff_count += 1

    if diff_count > 0:
        prob_copy = 1.0 - dis_ij
        if rng.random() < prob_copy:
            # Axelrod interaction successful -> update culture and return immediately
            trait = diff_buf_local[int(rng.random() * diff_count)]
            grid[agent_i, trait] = grid[agent_j, trait]
            return False, True, False

    # --- THIRD: Interaction failed OR diff_count == 0 -> Trigger mobility check ---
    avg_dis = 0.0
    for k in range(n_neighbors):
        avg_dis += dis_buf[k]
    avg_dis /= n_neighbors

    if avg_dis > dis_threshold:
        worst_k = 0
        worst_dis = dis_buf[0]
        for k in range(1, n_neighbors):
            if dis_buf[k] > worst_dis:
                worst_dis = dis_buf[k]
                worst_k = k
        worst_j = neighbor_buf[worst_k]

        # Find new target connection
        new_j = -1
        for _ in range(N):
            candidate = int(rng.random() * N)
            if candidate == agent_i or has_edge(adj_matrix, agent_i, candidate, N):
                continue
            new_j = candidate
            break

        if new_j != -1:
            if is_node_full(padded_edges, new_j, max_degree):
                return False, False, True  # Blocked by capacity limit

            remove_edge(padded_edges, adj_matrix, agent_i, worst_j, N, max_degree)
            add_edge(padded_edges, adj_matrix, agent_i, new_j, N, max_degree)
            return True, False, False

    # Default fallback (e.g., interaction failed/skipped and average dissatisfaction too low)
    return False, False, False

# ---------------------------------------------------------------------------
# Chunk Runner (Bubbles capacity_exceeded_count up)
# ---------------------------------------------------------------------------

@njit
def run_steps_chunk(grid, dev, padded_edges, adj_matrix, N, F, weight_mode, alpha,
                    dis_threshold, max_steps, updates_since_change, threshold,
                    rng, max_degree):
    """
    Returns (steps_done, rewire_count, updates_since_change, hit_threshold, capacity_exceeded_count).
    """
    neighbor_buf = np.empty(max_degree, dtype=np.int32)
    dis_buf = np.empty(max_degree, dtype=np.float32)
    rewire_count = 0
    capacity_exceeded_count = 0

    for step in range(max_steps):
        rewired, interacted, cap_exceeded = step_dev_sw(
            grid, dev, padded_edges, adj_matrix, N, F, weight_mode, alpha,
            dis_threshold, rng, max_degree, neighbor_buf, dis_buf
        )

        if cap_exceeded:
            capacity_exceeded_count += 1
            updates_since_change += 1
        elif rewired:
            rewire_count += 1
            updates_since_change += 1  # Structural change does not reset freeze window
        elif interacted:
            updates_since_change = 0   # Reset culture freeze window on successful copy
        else:
            updates_since_change += 1

        if updates_since_change >= threshold:
            return step + 1, rewire_count, updates_since_change, True, capacity_exceeded_count

    return max_steps, rewire_count, updates_since_change, False, capacity_exceeded_count

# ---------------------------------------------------------------------------
# Monitoring Phase
# ---------------------------------------------------------------------------

@njit
def run_monitoring_phase(grid, dev, padded_edges, adj_matrix, N, F, weight_mode, alpha,
                         dis_threshold, rng, updates_since_change, max_degree):
    monitoring_steps = N * 500
    sweep_size = N
    n_sweeps = monitoring_steps // sweep_size

    rewire_rate_history = np.zeros(n_sweeps, dtype=np.float64)

    neighbor_buf = np.empty(max_degree, dtype=np.int32)
    dis_buf = np.empty(max_degree, dtype=np.float32)

    for sweep in range(n_sweeps):
        sweep_rewires = 0
        for _ in range(sweep_size):
            rewired, interacted, cap_exceeded = step_dev_sw(
                grid, dev, padded_edges, adj_matrix, N, F, weight_mode, alpha,
                dis_threshold, rng, max_degree, neighbor_buf, dis_buf
            )
            if cap_exceeded:
                updates_since_change += 1
            elif rewired:
                sweep_rewires += 1
                updates_since_change += 1
            elif interacted:
                updates_since_change = 0
            else:
                updates_since_change += 1

        rewire_rate_history[sweep] = sweep_rewires / sweep_size

    slope = linregress_slope(rewire_rate_history)
    final_rewire_rate = 0.0
    for i in range(n_sweeps):
        final_rewire_rate += rewire_rate_history[i]
    final_rewire_rate /= n_sweeps

    return n_sweeps * sweep_size, final_rewire_rate, slope, updates_since_change

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@njit
def compute_mean_dissatisfaction(grid, dev, padded_edges, N, F, weight_mode, alpha, max_degree):
    total = 0.0
    count = 0
    for i in range(N):
        start = i * max_degree
        for k in range(max_degree):
            j = padded_edges[start + k]
            if j == -1:
                continue
            if i < j:
                shared = 0
                for f in range(F):
                    if grid[i, f] == grid[j, f]:
                        shared += 1
                d = total_dissatisfaction(shared, F, dev[i], dev[j], weight_mode, alpha)
                total += d
                count += 1
    if count == 0:
        return 0.0
    return total / count

@njit
def compute_degree_variance(padded_edges, N, max_degree):
    degrees = np.empty(N, dtype=np.float64)
    for i in range(N):
        start = i * max_degree
        deg = 0
        for k in range(max_degree):
            if padded_edges[start + k] != -1:
                deg += 1
        degrees[i] = deg
    mean_deg = 0.0
    for i in range(N):
        mean_deg += degrees[i]
    mean_deg /= N
    var = 0.0
    for i in range(N):
        diff = degrees[i] - mean_deg
        var += diff * diff
    return var / N

@njit
def get_cluster_metrics(grid, padded_edges, N, F, max_degree):
    """Returns (s_max, num_clusters) over cultural identity clusters."""
    visited = np.zeros(N, dtype=np.bool_)
    queue = np.empty(N, dtype=np.int32)

    max_size = 0
    num_clusters = 0

    for i in range(N):
        if visited[i]:
            continue
        size = 0
        head = 0
        tail = 0
        queue[tail] = i
        tail += 1
        visited[i] = True

        while head < tail:
            curr = queue[head]
            head += 1
            size += 1

            start = curr * max_degree
            for k in range(max_degree):
                nxt = padded_edges[start + k]
                if nxt == -1 or visited[nxt]:
                    continue
                identical = True
                for f in range(F):
                    if grid[curr, f] != grid[nxt, f]:
                        identical = False
                        break
                if identical:
                    visited[nxt] = True
                    queue[tail] = nxt
                    tail += 1

        if size > max_size:
            max_size = size
        num_clusters += 1

    return max_size / N, num_clusters

@njit
def compute_clustering_coefficient(padded_edges, N, max_degree):
    total_C = 0.0
    for i in range(N):
        # Build neighbor list for i
        start_i = i * max_degree
        nbrs_i = np.empty(max_degree, dtype=np.int32)
        deg_i = 0
        for k in range(max_degree):
            j = padded_edges[start_i + k]
            if j != -1:
                nbrs_i[deg_i] = j
                deg_i += 1

        if deg_i < 2:
            continue

        existing = 0
        for a in range(deg_i):
            u = nbrs_i[a]
            start_u = u * max_degree
            for b in range(a + 1, deg_i):
                v = nbrs_i[b]
                # Is v a neighbor of u?
                for kk in range(max_degree):
                    if padded_edges[start_u + kk] == v:
                        existing += 1
                        break

        max_possible = (deg_i * (deg_i - 1)) / 2.0
        total_C += existing / max_possible

    return total_C / N

@njit
def compute_characteristic_path_length(padded_edges, N, max_degree):
    total_path = 0
    reachable = 0
    queue = np.empty(N, dtype=np.int32)
    dist = np.empty(N, dtype=np.int32)

    for source in range(N):
        for i in range(N):
            dist[i] = -1
        head = 0
        tail = 0
        queue[tail] = source
        tail += 1
        dist[source] = 0

        while head < tail:
            curr = queue[head]
            head += 1
            start = curr * max_degree
            for k in range(max_degree):
                nxt = padded_edges[start + k]
                if nxt == -1 or dist[nxt] != -1:
                    continue
                dist[nxt] = dist[curr] + 1
                queue[tail] = nxt
                tail += 1
                total_path += dist[nxt]
                reachable += 1

    if reachable == 0:
        return 0.0
    return total_path / reachable