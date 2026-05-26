"""
Weight function reference for the Development Small World model.
These are NOT compiled with Numba — they serve as documentation and
Python-level validation/plotting helpers.

The actual dispatch lives in core_dev_sw.py::weight() as an @njit if/elif chain.

d = dev_i - dev_j  (i's perspective; positive = i richer than j)
alpha controls the strength of the development effect.

Inject by passing weight_mode (int) and alpha (float) into the model.
"""

WEIGHT_LINEAR = 0
WEIGHT_QUADRATIC = 1
WEIGHT_BIPHASIC = 2
WEIGHT_ATTRACTION = 3

WEIGHT_DESCRIPTIONS = {
    WEIGHT_LINEAR: (
        "Linear: w(d) = alpha * d\n"
        "Rich agents feel more dissatisfied with poorer neighbors (and vice versa).\n"
        "Symmetric hierarchy aversion scaled linearly with gap."
    ),
    WEIGHT_QUADRATIC: (
        "Quadratic (signed): w(d) = alpha * d * |d|\n"
        "Same direction as linear but gap effects accelerate. Small differences\n"
        "matter little; large ones dominate dissatisfaction strongly."
    ),
    WEIGHT_BIPHASIC: (
        "Biphasic: w(d) = alpha * 8 * d * (d - 0.25) * (d - 0.75)\n"
        "Positive for small diffs (competition/concurrence with near-equals),\n"
        "negative for moderate diffs (interest/attraction to somewhat different),\n"
        "positive for large diffs (resentment/aversion to very different).\n"
        "Zero-crossings at d=0, d=0.25, d=0.75."
    ),
    WEIGHT_ATTRACTION: (
        "Pure attraction: w(d) = -alpha * |d|\n"
        "Development difference always reduces dissatisfaction — agents are\n"
        "drawn to those who are different from them in development.\n"
        "Models aspirational dynamics: poor seek rich, rich seek 'interesting' poor."
    ),
}


def python_weight(d, mode, alpha):
    """Python reference implementation for plotting/testing."""
    if mode == WEIGHT_LINEAR:
        return alpha * d
    elif mode == WEIGHT_QUADRATIC:
        return alpha * d * abs(d)
    elif mode == WEIGHT_BIPHASIC:
        return alpha * 8.0 * d * (d - 0.25) * (d - 0.75)
    elif mode == WEIGHT_ATTRACTION:
        return -alpha * abs(d)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")


# ---------------------------------------------------------------------------
# Development distribution modes
# These are used at Python level during initialization only (not in Numba).
# ---------------------------------------------------------------------------

DEV_UNIFORM = 0
DEV_NORMAL = 1      # truncated normal; param: sigma (mean always 0.5)
DEV_PARETO = 2      # Pareto / power law; param: shape (lower = more inequality)
DEV_BIMODAL = 3     # two Gaussians; param: sigma (separation fixed at 0.3/0.7)

DEV_DESCRIPTIONS = {
    DEV_UNIFORM: "Uniform [0,1]: equal probability across all development levels.",
    DEV_NORMAL:  "Truncated Normal (mean=0.5, sigma=param): middle-class-heavy society.",
    DEV_PARETO:  "Pareto (shape=param): realistic wealth — most poor, few very rich. Lower shape = more inequality.",
    DEV_BIMODAL: "Bimodal (two Gaussians at 0.3 and 0.7, sigma=param): polarized society.",
}


def sample_development(N, mode, param, rng):
    """
    Sample development values for N agents.
    rng: np.random.Generator
    Returns float32 array of shape (N,) in [0, 1].
    """
    if mode == DEV_UNIFORM:
        return rng.random(N).astype(np.float32)

    elif mode == DEV_NORMAL:
        sigma = param if param is not None else 0.15
        samples = rng.normal(0.5, sigma, size=N)
        return np.clip(samples, 0.0, 1.0).astype(np.float32)

    elif mode == DEV_PARETO:
        shape = param if param is not None else 1.5
        # Pareto on [0,1]: use bounded Pareto transformation
        raw = rng.pareto(shape, size=N)
        # Normalize to [0,1] by clipping at 99th percentile
        p99 = np.percentile(raw, 99)
        if p99 > 0:
            raw = raw / p99
        return np.clip(raw, 0.0, 1.0).astype(np.float32)

    elif mode == DEV_BIMODAL:
        sigma = param if param is not None else 0.1
        half = N // 2
        low = rng.normal(0.3, sigma, size=half)
        high = rng.normal(0.7, sigma, size=N - half)
        samples = np.concatenate([low, high])
        rng.shuffle(samples)
        return np.clip(samples, 0.0, 1.0).astype(np.float32)

    else:
        raise ValueError(f"Unknown dev_mode: {mode}")


import numpy as np  # noqa: E402 (kept at bottom to not confuse njit imports)