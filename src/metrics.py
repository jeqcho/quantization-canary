"""Pure metric functions: normalized edit distance, z-score, CUSUM.

All functions in this module are pure (no I/O, no global state) and fully
unit-testable. The rest of the pipeline only depends on these signatures.
"""

from rapidfuzz.distance import Levenshtein

# Epsilon floor for degenerate perfectly-stable prompts where calibration
# produces sigma == 0. Chosen many orders of magnitude below any realistic
# sigma for normalized edit distance (bounded in [0, 1]) so it only kicks in
# for genuinely-deterministic prompts.
_SIGMA_FLOOR = 1e-9


def normalized_char_edit_distance(a: str, b: str) -> float:
    """Normalized character-level Levenshtein distance in [0, 1].

    Returns 0.0 for identical strings (including both-empty) and 1.0 for
    completely different strings. Delegates to rapidfuzz's C implementation
    for speed during calibration (~5,700 pairwise distances).
    """
    return Levenshtein.normalized_distance(a, b)


def per_prompt_zscore(d: float, mu: float, sigma: float) -> float:
    """Standard z-score with an epsilon floor on sigma.

    For degenerate prompts that were perfectly stable in calibration
    (sigma == 0), any deviation from mu is a huge signal: we floor sigma
    to 1e-9 so z stays large-but-finite rather than NaN or inf.
    The d == mu case short-circuits to zero so "no deviation" always
    reads as z=0 regardless of sigma.
    """
    if d == mu:
        return 0.0
    effective_sigma = sigma if sigma > _SIGMA_FLOOR else _SIGMA_FLOOR
    return (d - mu) / effective_sigma


def cusum_step(
    state: float, deviation: float, k: float, h: float
) -> tuple[float, bool]:
    """One step of an upper-sided CUSUM control chart.

    Accumulates positive deviations (with slack k, which absorbs noise)
    and floors at zero. Alarms when the accumulated state strictly
    exceeds h.
    """
    new_state = max(0.0, state + deviation - k)
    alarmed = new_state > h
    return new_state, alarmed
