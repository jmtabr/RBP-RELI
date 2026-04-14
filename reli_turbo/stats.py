"""
reli_turbo.stats -- Statistical functions for RELI enrichment analysis.

Implements the Z-score, P-value, and Bonferroni correction logic from
RELI_impl.cpp ``cal_stats()`` (line 573).  Key behaviours replicated:

1. Mean is computed over ALL 2001 values (observed + 2000 null),
   matching the C++ code which sums ``statsvec[0..2000]`` and divides
   by ``statsvec.size() == 2001``.

2. Standard deviation uses Bessel's correction (ddof=1), dividing by
   ``statsvec.size() - 1 == 2000``.

3. The 5% significance guard forces Z = 0 when:
   (a) SD == 0 (all permutations gave the same count), or
   (b) observed < ceil(N_loci * 0.05).

4. P-value is the upper tail of the standard normal: Q(z).

5. Bonferroni correction: corrected_p = min(p * n_targets, 1.0).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# Significance guard threshold fraction (RELI_impl.cpp: sig_pct = 0.05)
SIG_PCT = 0.05


def compute_z_score(
    observed: float | np.ndarray,
    mean: float | np.ndarray,
    std: float | np.ndarray,
    n_loci: int | None = None,
) -> float | np.ndarray:
    """Compute Z-score with the RELI 5% guard.

    Parameters
    ----------
    observed : scalar or array
        Observed overlap count(s) from iteration 0.
    mean : scalar or array
        Mean of all 2001 overlap counts (observed + null).
    std : scalar or array
        Sample standard deviation (ddof=1) of all 2001 counts.
    n_loci : int, optional
        Total number of query loci.  If provided, the 5% guard is
        applied: Z is forced to 0 when observed < ceil(n_loci * 0.05).

    Returns
    -------
    z : same shape as inputs
        Z-score.  Zero where the guard triggers.
    """
    observed = np.asarray(observed, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    # Guard 1: zero variance
    guard = std == 0.0

    # Guard 2: observed below 5% threshold
    if n_loci is not None:
        threshold = math.ceil(n_loci * SIG_PCT)
        guard = guard | (observed < threshold)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(guard, 0.0, (observed - mean) / std)
    return float(z) if z.ndim == 0 else z


def compute_p_value(z_score: float | np.ndarray) -> float | np.ndarray:
    """Compute one-sided P-value from the upper tail of the standard normal.

    Equivalent to ``gsl_cdf_ugaussian_Q(z)`` used in RELI_impl.cpp
    ``cal_stats()`` (line ~600).

    P = Q(z) = 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))

    Parameters
    ----------
    z_score : scalar or array
        Z-score(s).

    Returns
    -------
    p : same shape as input
        Upper-tail P-value(s).
    """
    z = np.asarray(z_score, dtype=np.float64)
    p = norm.sf(z)  # survival function = 1 - CDF = upper tail
    return float(p) if p.ndim == 0 else p


def bonferroni_correct(
    p_values: float | np.ndarray,
    n_targets: int,
) -> float | np.ndarray:
    """Apply Bonferroni correction.

    Matches RELI_impl.cpp: ``corrected_pval = min(pval * corr, 1.0)``

    Parameters
    ----------
    p_values : scalar or array
        Raw P-value(s).
    n_targets : int
        Number of targets tested (Bonferroni multiplier).

    Returns
    -------
    corrected : same shape as input
        Corrected P-value(s), capped at 1.0.
    """
    p = np.asarray(p_values, dtype=np.float64)
    corrected = np.minimum(p * n_targets, 1.0)
    return float(corrected) if corrected.ndim == 0 else corrected


def compute_enrichment(
    observed: float | np.ndarray,
    mean: float | np.ndarray,
) -> float | np.ndarray:
    """Compute relative enrichment (Relative Risk).

    enrichment = observed / mean  (0 if mean == 0)

    Parameters
    ----------
    observed, mean : scalar or array

    Returns
    -------
    enrichment : same shape as inputs
    """
    observed = np.asarray(observed, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean > 0, observed / mean, 0.0)


def compute_stats_from_counts(
    all_counts: np.ndarray,
    n_loci: int,
    n_targets_corr: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute full RELI statistics from a 2-D overlap count matrix.

    This is the GPU-side equivalent of ``cal_stats()`` but usable on
    CPU numpy arrays as well.

    Parameters
    ----------
    all_counts : np.ndarray, shape (N_targets, 2001)
        Overlap counts.  Column 0 is observed; columns 1-2000 are null.
    n_loci : int
        Total number of query loci.
    n_targets_corr : int, optional
        Bonferroni correction factor.  Defaults to ``all_counts.shape[0]``.

    Returns
    -------
    dict with keys: observed, mean, sd, zscore, pval, corr_pval, enrichment
    """
    counts_f = all_counts.astype(np.float64)
    observed = counts_f[:, 0]

    # Mean over all 2001 values (matches C++ which includes observed)
    mean = np.mean(counts_f, axis=1)

    # Sample SD with Bessel's correction (ddof=1, denominator = 2000)
    sd = np.std(counts_f, axis=1, ddof=1)

    # Z-score with guards
    zscore = compute_z_score(observed, mean, sd, n_loci=n_loci)

    # P-value
    pval = compute_p_value(zscore)

    # Bonferroni
    if n_targets_corr is None:
        n_targets_corr = all_counts.shape[0]
    corr_pval = bonferroni_correct(pval, n_targets_corr)

    # Enrichment
    enrichment = compute_enrichment(observed, mean)

    return {
        "observed": observed.astype(int),
        "mean": mean,
        "sd": sd,
        "zscore": zscore,
        "pval": pval,
        "corr_pval": corr_pval,
        "enrichment": enrichment,
    }
