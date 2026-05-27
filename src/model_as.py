import hashlib
import numpy as np
from .core_as import (run_mcs_chunk, run_monitoring_phase,
                      check_constant_as, get_cluster_metrics_as,
                      MOBILITY_SAMPLE_INTERVAL)


def _make_rand_arrays(rng, n_steps, N_cells):
    """
    Pre-generate the three parallel random arrays consumed by the Numba
    hot loop.  Called once per chunk/phase on the Python side so that
    Numba never touches the RNG wrapper inside the critical loop.

    n_steps : total agent-step slots needed  (e.g. max_mcs * N_cells)

    Returns
    -------
    rand_nodes  : int64 array, shape (n_steps,), values in [0, N_cells)
    rand_floats : float64 array, shape (n_steps,), values in [0, 1)
    rand_misc   : int64 array, shape (n_steps,), large positive ints
                  consumed via modulo inside perform_agent_step for both
                  empty-slot selection and neighbour/trait choice.
                  Bound: N_cells * N_cells gives values large enough that
                  modulo bias is negligible for any realistic num_empty or
                  max_degree (both << N_cells).
    """
    rand_nodes  = rng.integers(0, N_cells,            size=n_steps, dtype=np.int64)
    rand_floats = rng.random(n_steps)
    rand_misc   = rng.integers(0, N_cells * N_cells,  size=n_steps, dtype=np.int64)
    return rand_nodes, rand_floats, rand_misc


class AxelrodSchellingModel:
    def __init__(self, L=40, F=5, q=10, h=0.1, T=0.5, m=1, master_seed=42):
        self.L = L
        self.F = F
        self.q = q
        self.h = h
        self.T = T
        self.m = m
        self.master_seed = master_seed

        self.N_cells  = self.L * self.L
        self.num_empty = int(self.N_cells * self.h)

        # Deterministic per-configuration seed
        seed_context = f"{master_seed}_{F}_{q}_{L}_{h}_{T}_{m}"
        self.seed = int(hashlib.md5(seed_context.encode()).hexdigest()[:16], 16)
        self.rng  = np.random.default_rng(self.seed)

        self.grid    = None
        self.empty_locs = None
        self.updates_since_last_change = 0
        self.total_mcs = 0

        self.edge_ptrs = None
        self.edges     = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """2-D periodic von Neumann neighbourhood → CSR."""
        self.edge_ptrs = np.arange(0, 4 * self.N_cells + 1, 4, dtype=np.int32)
        self.edges     = np.empty(4 * self.N_cells, dtype=np.int32)

        idx = 0
        for i in range(self.L):
            for j in range(self.L):
                self.edges[idx]     = i * self.L + ((j + 1) % self.L)   # right
                self.edges[idx + 1] = i * self.L + ((j - 1) % self.L)   # left
                self.edges[idx + 2] = ((i + 1) % self.L) * self.L + j   # down
                self.edges[idx + 3] = ((i - 1) % self.L) * self.L + j   # up
                idx += 4

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_new_simulation(self):
        self._build_graph()
        self.grid = self.rng.integers(
            0, self.q, size=(self.N_cells, self.F), dtype=np.int16
        )
        self.empty_locs = np.zeros(self.num_empty, dtype=np.int32)

        if self.num_empty > 0:
            flat_indices = self.rng.choice(
                self.N_cells, size=self.num_empty, replace=False
            )
            for i, idx in enumerate(flat_indices):
                self.grid[idx, :] = -1
                self.empty_locs[i] = idx

        self.updates_since_last_change = 0
        self.total_mcs = 0

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, additional_mcs, transient_mcs=None):
        if additional_mcs <= 0:
            return False, self.total_mcs, -1.0

        if transient_mcs is None:
            transient_mcs = 10_000_000 // self.N_cells

        threshold_updates = self.N_cells * 100_000
        mcs_left  = additional_mcs
        last_m    = -1.0
        max_degree = int(np.max(self.edge_ptrs[1:] - self.edge_ptrs[:-1]))

        # ---- Phase 1: uninterruptible transient -------------------------
        if self.total_mcs < transient_mcs:
            mcs_to_do = min(mcs_left, transient_mcs - self.total_mcs)
            n_steps   = mcs_to_do * self.N_cells

            rand_nodes, rand_floats, rand_misc = _make_rand_arrays(
                self.rng, n_steps, self.N_cells
            )

            mcs_done, self.updates_since_last_change, _ = run_mcs_chunk(
                self.grid, self.N_cells, self.F,
                self.empty_locs, self.num_empty, self.T,
                mcs_to_do, self.updates_since_last_change, np.iinfo(np.int64).max,
                rand_nodes, rand_floats, rand_misc,
                self.edge_ptrs, self.edges, max_degree,
            )
            self.total_mcs += mcs_done
            mcs_left       -= mcs_done

        # ---- Phase 2: interleaved convergence checking ------------------
        while mcs_left > 0:
            chunk   = min(mcs_left, 10_000)
            n_steps = chunk * self.N_cells

            rand_nodes, rand_floats, rand_misc = _make_rand_arrays(
                self.rng, n_steps, self.N_cells
            )

            mcs_done, self.updates_since_last_change, hit_threshold = run_mcs_chunk(
                self.grid, self.N_cells, self.F,
                self.empty_locs, self.num_empty, self.T,
                chunk, self.updates_since_last_change, threshold_updates,
                rand_nodes, rand_floats, rand_misc,
                self.edge_ptrs, self.edges, max_degree,
            )
            self.total_mcs += mcs_done
            mcs_left       -= mcs_done

            if hit_threshold:
                if check_constant_as(
                    self.grid, self.N_cells, self.F, self.T,
                    self.num_empty, self.edge_ptrs, self.edges
                ):
                    return True, self.total_mcs, 0.0

                # Pre-generate random arrays for the full monitoring phase.
                # We allocate for the worst case (50 000 sweeps); the early-
                # exit inside run_monitoring_phase will simply not consume
                # the tail of the arrays — no wasted computation, only a
                # small one-time allocation cost.
                mon_steps = 50_000 * self.N_cells
                rand_nodes_m, rand_floats_m, rand_misc_m = _make_rand_arrays(
                    self.rng, mon_steps, self.N_cells
                )

                mon_sweeps, final_m, slope, new_updates = run_monitoring_phase(
                    self.grid, self.N_cells, self.F,
                    self.empty_locs, self.num_empty, self.T,
                    self.updates_since_last_change,
                    self.edge_ptrs, self.edges, max_degree,
                    rand_nodes_m, rand_floats_m, rand_misc_m,
                )

                self.total_mcs += mon_sweeps
                mcs_left       -= mon_sweeps
                last_m          = final_m
                self.updates_since_last_change = new_updates

                if abs(slope) < 1e-4 and final_m < 0.95:
                    return True, self.total_mcs, final_m
                else:
                    self.updates_since_last_change = 0

        return False, self.total_mcs, last_m

    # ------------------------------------------------------------------
    # Metrics & checkpointing (unchanged)
    # ------------------------------------------------------------------

    def get_metrics(self):
        return get_cluster_metrics_as(
            self.grid, self.N_cells, self.F, self.edge_ptrs, self.edges
        )

    def get_checkpoint_data(self):
        return {
            "grid":       self.grid,
            "empty_locs": self.empty_locs,
            "edge_ptrs":  self.edge_ptrs,
            "edges":      self.edges,
            "updates":    self.updates_since_last_change,
            "rng_state":  self.rng.bit_generator.state,
            "total_mcs":  self.total_mcs,
        }

    def load_checkpoint_data(self, cp):
        self.grid = cp["grid"]
        if self.grid.ndim == 3:
            self.grid = self.grid.reshape(-1, self.F)

        self.empty_locs = cp["empty_locs"]
        if self.empty_locs.ndim == 2:
            self.empty_locs = np.array(
                [x * self.L + y for x, y in self.empty_locs], dtype=np.int32
            )

        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state   = cp["rng_state"]
        self.total_mcs = cp.get("total_mcs", 0)

        if "edge_ptrs" in cp and "edges" in cp:
            self.edge_ptrs = cp["edge_ptrs"]
            self.edges     = cp["edges"]
        else:
            self._build_graph()