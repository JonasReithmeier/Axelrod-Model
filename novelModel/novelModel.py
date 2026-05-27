import hashlib
import numpy as np
import networkx as nx

from .core_novelModel import (
    run_steps_chunk,
    run_monitoring_phase,
    check_frozen_cultural,
    get_cluster_metrics,
    compute_mean_dissatisfaction,
    compute_degree_variance,
    compute_clustering_coefficient,
    compute_characteristic_path_length,
    linregress_slope,
)
from .weight_functions_novelModel import sample_development


class AxelrodDevSmallWorld:
    def __init__(
        self,
        N=200,
        k=4,
        p=0.1,
        F=5,
        q=10,
        weight_mode=0,
        alpha=1.0,
        dis_threshold=0.5,
        dev_mode=0,
        dev_param=None,
        seed=42,
    ):
        """
        Parameters
        ----------
        N             : Number of agents
        k             : WS nearest-neighbor count
        p             : WS rewiring probability
        F             : Number of cultural features
        q             : Number of cultural traits per feature
        weight_mode   : 0=linear, 1=quadratic, 2=biphasic, 3=attraction
        alpha         : Strength of development weight function
        dis_threshold : Average dissatisfaction above which agent rewires
        dev_mode      : 0=uniform, 1=normal, 2=pareto, 3=bimodal
        dev_param     : Shape/sigma parameter for dev distribution
        seed          : Master seed (deterministic across resumes)
        """
        self.N = N
        self.k = k
        self.p = p
        self.F = F
        self.q = q
        self.weight_mode = weight_mode
        self.alpha = alpha
        self.dis_threshold = dis_threshold
        self.dev_mode = dev_mode
        self.dev_param = dev_param
        self.seed = seed

        # Deterministic seed derived from all params so two runs with the
        # same config are bit-identical even after a checkpoint resume.
        seed_context = (
            f"{seed}_{N}_{k}_{p}_{F}_{q}_{weight_mode}_{alpha}_"
            f"{dis_threshold}_{dev_mode}_{dev_param}"
        )
        derived_seed = int(hashlib.md5(seed_context.encode()).hexdigest()[:16], 16)
        self.rng = np.random.default_rng(derived_seed)

        # Generous max-degree ceiling: 8x initial k (handles topology drift)
        self.max_degree = min(k * 8, N - 1)

        # State
        self.grid = None          # (N, F) int16
        self.dev = None           # (N,)   float32
        self.padded_edges = None  # (N * max_degree,) int32, -1 = sentinel
        self.adj_matrix = None    # (N * N,) bool, flat row-major

        self.updates_since_last_change = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Watts-Strogatz → padded flat CSR + boolean adjacency matrix."""
        G = nx.watts_strogatz_graph(self.N, self.k, self.p, seed=self.seed)

        self.padded_edges = np.full(self.N * self.max_degree, -1, dtype=np.int32)
        self.adj_matrix = np.zeros(self.N * self.N, dtype=np.bool_)

        for i in range(self.N):
            neighbors = list(G.neighbors(i))
            if len(neighbors) > self.max_degree:
                neighbors = neighbors[: self.max_degree]
            slot = 0
            for j in neighbors:
                self.padded_edges[i * self.max_degree + slot] = j
                self.adj_matrix[i * self.N + j] = True
                slot += 1

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_new_simulation(self):
        self._build_graph()
        self.grid = self.rng.integers(0, self.q, size=(self.N, self.F), dtype=np.int16)
        self.dev = sample_development(self.N, self.dev_mode, self.dev_param, self.rng)
        self.updates_since_last_change = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, additional_steps, transient_steps=None):
        """
        Run the model for additional_steps elementary steps.

        Returns (is_converged, total_steps, final_rewire_rate).
        """
        if additional_steps <= 0:
            return False, self.total_steps, -1.0

        if transient_steps is None:
            transient_steps = self.N * 200  # 500 sweeps burn-in

        threshold = self.N * 100_000
        steps_left = additional_steps
        last_rewire_rate = -1.0

        # --- Phase 1: transient burn-in, no convergence checking ---
        if self.total_steps < transient_steps:
            to_do = min(steps_left, transient_steps - self.total_steps)
            done, _, self.updates_since_last_change, _ = run_steps_chunk(
                self.grid, self.dev, self.padded_edges, self.adj_matrix,
                self.N, self.F, self.weight_mode, self.alpha,
                self.dis_threshold, to_do,
                self.updates_since_last_change, np.iinfo(np.int64).max,
                self.rng, self.max_degree,
            )
            self.total_steps += done
            steps_left -= done

        # --- Phase 2: chunked run with convergence checks ---
        while steps_left > 0:
            chunk = min(steps_left, self.N * 10)  # 10-sweep chunks
            done, rewire_count, self.updates_since_last_change, hit = run_steps_chunk(
                self.grid, self.dev, self.padded_edges, self.adj_matrix,
                self.N, self.F, self.weight_mode, self.alpha,
                self.dis_threshold, chunk,
                self.updates_since_last_change, threshold,
                self.rng, self.max_degree,
            )
            self.total_steps += done
            steps_left -= done

            if hit:
                # Cultural freeze check
                if check_frozen_cultural(
                    self.grid, self.padded_edges, self.N, self.F, self.max_degree
                ):
                    return True, self.total_steps, 0.0

                # Structural convergence: monitor rewiring rate slope
                mon_done, final_rr, slope, new_updates = run_monitoring_phase(
                    self.grid, self.dev, self.padded_edges, self.adj_matrix,
                    self.N, self.F, self.weight_mode, self.alpha,
                    self.dis_threshold, self.rng,
                    self.updates_since_last_change, self.max_degree,
                )
                self.total_steps += mon_done
                steps_left -= mon_done
                last_rewire_rate = final_rr
                self.updates_since_last_change = new_updates

                if abs(slope) < 1e-4:
                    # Rewiring rate has stabilized — structural convergence
                    return True, self.total_steps, final_rr
                else:
                    self.updates_since_last_change = 0

        return False, self.total_steps, last_rewire_rate

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_culture_metrics(self):
        """Returns (s_max, num_clusters)."""
        return get_cluster_metrics(
            self.grid, self.padded_edges, self.N, self.F, self.max_degree
        )

    def get_network_metrics(self):
        """Returns (L, C, degree_variance)."""
        L = compute_characteristic_path_length(
            self.padded_edges, self.N, self.max_degree
        )
        C = compute_clustering_coefficient(
            self.padded_edges, self.N, self.max_degree
        )
        dv = compute_degree_variance(self.padded_edges, self.N, self.max_degree)
        return L, C, dv

    def get_dissatisfaction(self):
        """Returns mean dissatisfaction over all connected pairs."""
        return compute_mean_dissatisfaction(
            self.grid, self.dev, self.padded_edges,
            self.N, self.F, self.weight_mode, self.alpha, self.max_degree
        )

    def get_all_metrics(self):
        """
        Convenience: returns dict of all sweep-relevant metrics.
        Note: L and C are O(N²) — call at end of run, not mid-run.
        """
        s_max, num_clusters = self.get_culture_metrics()
        L, C, deg_var = self.get_network_metrics()
        mean_dis = self.get_dissatisfaction()
        return {
            "s_max": float(s_max),
            "num_clusters": int(num_clusters),
            "mean_dissatisfaction": float(mean_dis),
            "L": float(L),
            "C": float(C),
            "degree_variance": float(deg_var),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def get_checkpoint_data(self):
        return {
            "grid": self.grid,
            "dev": self.dev,
            "padded_edges": self.padded_edges,
            "adj_matrix": self.adj_matrix,
            "updates": self.updates_since_last_change,
            "rng_state": self.rng.bit_generator.state,
            "total_steps": self.total_steps,
        }

    def load_checkpoint_data(self, cp):
        # Graph is deterministically rebuildable — but checkpoint stores
        # the mutated runtime graph, so we load it directly.
        self.grid = cp["grid"]
        self.dev = cp["dev"]
        self.padded_edges = cp["padded_edges"]
        self.adj_matrix = cp["adj_matrix"]
        self.updates_since_last_change = cp["updates"]
        self.rng.bit_generator.state = cp["rng_state"]
        self.total_steps = cp.get("total_steps", 0)