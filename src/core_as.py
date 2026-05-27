import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Linear regression helper (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Convergence checks (unchanged logic)
# ---------------------------------------------------------------------------

@njit
def check_constant_as(grid, N_cells, F, T, num_empty, edge_ptrs, edges):
    for node in range(N_cells):
        if grid[node, 0] == -1:
            continue

        valid_neighbors = 0
        diff_sum_F = 0
        can_interact = False

        start = edge_ptrs[node]
        end   = edge_ptrs[node + 1]

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
        end   = edge_ptrs[node + 1]

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

    if active_agents == 0:
        return 0.0
    return movers / active_agents


# ---------------------------------------------------------------------------
# Core agent step — receives pre-generated random values directly.
#
# r_float : float — uniform [0,1) for the Axelrod interaction probability
# r_misc  : int64 — large uniform integer, consumed via modulo for both
#                   the empty-slot index and the neighbour/trait choice.
#                   Using one pre-generated value for all modulo uses within
#                   a single step is fine: the choices are mutually exclusive
#                   (only one branch executes) so there is no correlation.
# ---------------------------------------------------------------------------

@njit
def perform_agent_step(grid, F, empty_locs, num_empty, T, node, updates,
                       r_float, r_misc,
                       diff_indices, valid_neighbors_coords,
                       edge_ptrs, edges):
    if grid[node, 0] == -1:
        return updates + 1

    valid_n_count = 0
    diff_sum_F    = 0

    start = edge_ptrs[node]
    end   = edge_ptrs[node + 1]

    for e in range(start, end):
        n_node = edges[e]
        if grid[n_node, 0] != -1:
            valid_neighbors_coords[valid_n_count] = n_node
            valid_n_count += 1
            shared = 0
            for f in range(F):
                if grid[node, f] == grid[n_node, f]:
                    shared += 1
            diff_sum_F += (F - shared)

    is_unhappy = False
    if valid_n_count > 0:
        if diff_sum_F > T * F * valid_n_count:
            is_unhappy = True
    else:
        is_unhappy = True

    if is_unhappy and num_empty > 0:
        e_idx      = r_misc % num_empty
        empty_node = empty_locs[e_idx]

        for f in range(F):
            grid[empty_node, f] = grid[node, f]
            grid[node, f]       = -1

        empty_locs[e_idx] = node
        return updates + 1

    else:
        if valid_n_count > 0:
            n_node = valid_neighbors_coords[r_misc % valid_n_count]

            shared     = 0
            diff_count = 0
            for f in range(F):
                if grid[node, f] == grid[n_node, f]:
                    shared += 1
                else:
                    diff_indices[diff_count] = f
                    diff_count += 1

            if 0 < shared < F:
                if (r_float * F) < shared:
                    target_trait = diff_indices[r_misc % diff_count]
                    grid[node, target_trait] = grid[n_node, target_trait]
                    return 0
                else:
                    return updates + 1
            else:
                return updates + 1
        else:
            return updates + 1


# ---------------------------------------------------------------------------
# MCS chunk — pre-generates all random numbers for the entire chunk.
#
# Three parallel arrays of length (max_mcs * N_cells):
#   rand_nodes  : int64  in [0, N_cells)      — which agent to activate
#   rand_floats : float64 in [0, 1)           — Axelrod probability gate
#   rand_misc   : int64  large positive ints  — consumed via modulo inside
#                                               perform_agent_step
#
# Generated in model_as.py (Python side) and passed in, so Numba never
# calls the RNG wrapper inside the hot loop.
# ---------------------------------------------------------------------------

@njit
def run_mcs_chunk(grid, N_cells, F, empty_locs, num_empty, T,
                  max_mcs, updates, threshold,
                  rand_nodes, rand_floats, rand_misc,
                  edge_ptrs, edges, max_degree):

    diff_indices           = np.empty(F,          dtype=np.int32)
    valid_neighbors_coords = np.empty(max_degree, dtype=np.int32)

    idx = 0

    for mcs in range(max_mcs):
        for _ in range(N_cells):
            updates = perform_agent_step(
                grid, F, empty_locs, num_empty, T,
                rand_nodes[idx], updates,
                rand_floats[idx], rand_misc[idx],
                diff_indices, valid_neighbors_coords,
                edge_ptrs, edges,
            )
            idx += 1

        if updates >= threshold:
            return mcs + 1, updates, True

    return max_mcs, updates, False


# ---------------------------------------------------------------------------
# Monitoring phase — bulk RNG + sparse mobility sampling.
#
# Mobility is measured every MOBILITY_SAMPLE_INTERVAL sweeps (default 100)
# instead of every sweep → 100x fewer full-grid scans.
#
# Early-exit: once a 10-sample window has variance < 1e-8 the system is
# frozen; we short-circuit the remaining 50 000 - sweep iterations.
#
# rand_nodes / rand_floats / rand_misc must have length
# >= monitoring_sweeps * N_cells (allocated on the Python side).
# ---------------------------------------------------------------------------

MOBILITY_SAMPLE_INTERVAL = 100

@njit
def run_monitoring_phase(grid, N_cells, F, empty_locs, num_empty, T,
                         updates, edge_ptrs, edges, max_degree,
                         rand_nodes, rand_floats, rand_misc):

    monitoring_sweeps = 50_000
    n_samples         = monitoring_sweeps // MOBILITY_SAMPLE_INTERVAL
    m_history         = np.zeros(n_samples, dtype=np.float64)

    diff_indices           = np.empty(F,          dtype=np.int32)
    valid_neighbors_coords = np.empty(max_degree, dtype=np.int32)

    idx      = 0
    sample_i = 0

    for sweep in range(monitoring_sweeps):
        for _ in range(N_cells):
            updates = perform_agent_step(
                grid, F, empty_locs, num_empty, T,
                rand_nodes[idx], updates,
                rand_floats[idx], rand_misc[idx],
                diff_indices, valid_neighbors_coords,
                edge_ptrs, edges,
            )
            idx += 1

        if (sweep + 1) % MOBILITY_SAMPLE_INTERVAL == 0:
            m_history[sample_i] = calculate_mobility_as(
                grid, N_cells, F, T, num_empty, edge_ptrs, edges
            )
            sample_i += 1

            # Early exit once the last 10 samples are stationary
            if sample_i >= 10:
                mean_w = 0.0
                for v in m_history[sample_i - 10 : sample_i]:
                    mean_w += v
                mean_w /= 10.0

                var_w = 0.0
                for v in m_history[sample_i - 10 : sample_i]:
                    d = v - mean_w
                    var_w += d * d
                var_w /= 10.0

                if var_w < 1e-8:
                    for j in range(sample_i, n_samples):
                        m_history[j] = mean_w
                    return sweep + 1, mean_w, 0.0, updates

    slope   = linregress(m_history)
    final_m = 0.0
    for v in m_history:
        final_m += v
    final_m /= n_samples

    return monitoring_sweeps, final_m, slope, updates


# ---------------------------------------------------------------------------
# Cluster metrics (unchanged)
# ---------------------------------------------------------------------------

@njit
def get_cluster_metrics_as(grid, N_cells, F, edge_ptrs, edges):
    visited  = np.zeros(N_cells, dtype=np.bool_)
    queue    = np.empty(N_cells,  dtype=np.int32)

    max_size      = 0
    total_size    = 0
    num_clusters  = 0
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
            tail       += 1
            visited[node] = True

            while head < tail:
                curr  = queue[head]
                head += 1
                size += 1

                start = edge_ptrs[curr]
                end   = edge_ptrs[curr + 1]

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
                            queue[tail]     = n_node
                            tail           += 1

            if size > max_size:
                max_size = size
            total_size   += size
            num_clusters += 1

    if active_agents == 0:
        return 0.0, 0.0
    s_max  = max_size / active_agents
    s_mean = (total_size / num_clusters) / active_agents
    return s_max, s_mean