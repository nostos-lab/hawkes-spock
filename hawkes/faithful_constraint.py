"""Faithful reproduction of the Mishra et al. (2016) constrained marked-power-law fit.

Reference: Mishra, Rizoiu, and Xie (2016), CIKM, "Feature Driven and Point
Process Approaches for Popularity Prediction." This module mirrors the
closed-form branching-factor relation and the `n* < 1` inequality constraint
that the original work imposed via Ipopt; here the constraint is enforced by
`scipy.optimize.minimize(method='SLSQP', ...)`.

Public functions:

- `fit_one_cascade(t, m, T, seed=...)` -- multi-start constrained MLE on a
  single cascade. Returns the SLSQP `OptimizeResult` of the best random
  restart (lowest NLL among finite-objective runs), or `None` if every
  restart failed.
- `branching_factor_closed_form(K, beta, c, theta, alpha=2.016)` -- the
  Mishra (2016) closed form `K * (alpha - 1) / (alpha - 1 - beta) /
  (theta * c^theta)`, returning `inf` outside the validity region
  `beta < alpha - 1`.
- `hill_alpha(marks, m_min=1.0)` -- the Hill estimator for the mark Pareto
  exponent.
- `fit_active(csv_path, n_cascades=80, T_obs=600, seed=7)` -- orchestrates
  the per-cascade fit on a sample of ACTIVE cascades stratified by size; each
  record contains the original cascade index, the number of events used, the
  constrained `n_hat`, the fitted `beta`, and the empirical Hill `alpha`.

The default Pareto exponent `alpha = 2.016` matches the value hard-coded in
Mishra et al. (2016)'s released code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


ALPHA_ASSUMED_DEFAULT = 2.016


@dataclass
class CascadeFit:
    cascade_index: int
    N: int
    n_hat: float
    beta: float
    alpha_emp: float


def _nll_marked_powerlaw(p: np.ndarray, t: np.ndarray, m: np.ndarray, T: float) -> float:
    """Negative log-likelihood used by Mishra et al. (2016): mu = 0,
    parameters [K, beta, c, theta].
    """
    K, beta, c, theta = p
    if K <= 0 or c <= 0 or theta <= 0 or beta < 0:
        return 1e12
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mb = m ** beta
        if not np.all(np.isfinite(mb)):
            return 1e12
        # Compensator
        comp = K * np.sum(mb * (1.0 / (theta * c ** theta) - 1.0 / (theta * (T + c - t) ** theta)))
        if not np.isfinite(comp):
            return 1e12
        # log lambda(t_i^-) for i >= 1 (i = 0 is the seed)
        dt = t[:, None] - t[None, :] + c
        W = np.tril(dt ** (-(1.0 + theta)), k=-1)
        lam = (K * (W * mb[None, :]).sum(axis=1))[1:]
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            return 1e12
        ll = np.sum(np.log(lam))
    val = comp - ll
    return val if np.isfinite(val) else 1e12


def branching_factor_closed_form(K: float, beta: float, c: float, theta: float,
                                  alpha: float = ALPHA_ASSUMED_DEFAULT) -> float:
    """Mishra et al. (2016) closed-form n_star.

    n_star = K * (alpha - 1) / (alpha - 1 - beta) / (theta * c^theta),
    valid for beta < alpha - 1 and theta > 0. Returns inf outside that region.
    """
    if beta >= alpha - 1 or theta <= 0:
        return float("inf")
    return K * (alpha - 1) / (alpha - 1 - beta) / (theta * c ** theta)


def _log_n_constraint(p: np.ndarray, alpha: float = ALPHA_ASSUMED_DEFAULT) -> float:
    """log(n_star) using the same hard-coded alpha and small offset as the
    Mishra et al. (2016) reference implementation.
    """
    K, beta, c, theta = p
    return (np.log(K)
            + np.log(alpha - 1 + 0.0846)
            - np.log(alpha - 1 - beta)
            - np.log(theta)
            - theta * np.log(c))


def fit_one_cascade(t: np.ndarray, m: np.ndarray, T: float,
                    seed: int = 0, n_starts: int = 10,
                    alpha: float = ALPHA_ASSUMED_DEFAULT):
    """Multi-start SLSQP fit of [K, beta, c, theta] under `n_star < 1`.

    Returns the best (lowest-NLL) `OptimizeResult` across `n_starts` random
    restarts, or `None` if no restart returned a finite objective.
    """
    rng = np.random.default_rng(seed)
    ub_log_n = np.log(1 - 1e-9)
    lb_log_n = np.log(1e-12)
    cons = [
        {"type": "ineq", "fun": lambda p, a=alpha: ub_log_n - _log_n_constraint(p, a)},
        {"type": "ineq", "fun": lambda p, a=alpha: _log_n_constraint(p, a) - lb_log_n},
    ]
    bounds = [
        (1e-8, 1.0),              # K <= 1
        (1e-8, alpha - 1 - 1e-6), # beta < alpha - 1
        (1e-8, 1e6),              # c
        (1e-8, 1e2),              # theta
    ]
    best = None
    for _ in range(n_starts):
        x0 = np.array([
            10 ** rng.uniform(-3, 0),
            rng.uniform(0.05, 1.0),
            10 ** rng.uniform(-2, 3.3),
            rng.uniform(0.1, 2.0),
        ])
        try:
            r = minimize(_nll_marked_powerlaw, x0, args=(t, m, T),
                         method="SLSQP", bounds=bounds, constraints=cons,
                         options={"maxiter": 500, "ftol": 1e-9})
        except Exception:
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


def hill_alpha(marks: np.ndarray, m_min: float = 1.0) -> float:
    """Hill estimator of the Pareto exponent above `m_min`.

    Returns NaN when fewer than five values lie above the threshold.
    """
    mm = marks[marks >= m_min]
    if len(mm) < 5:
        return float("nan")
    return 1 + len(mm) / np.sum(np.log(mm / m_min))


def _read_active_cascades(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read the ACTIVE CSV and locate cascade boundaries by `time == 0` rows."""
    times: list[float] = []
    marks: list[float] = []
    bounds: list[int] = []
    with csv_path.open() as f:
        f.readline()  # header: time,magnitude
        for k, line in enumerate(f):
            a, b = line.split(",")
            if float(a) == 0.0:
                bounds.append(k)
            times.append(float(a))
            marks.append(float(b))
    bounds.append(len(times))
    return np.asarray(times), np.asarray(marks), bounds


def _stratified_pick(sizes: np.ndarray, rng: np.random.Generator,
                     strata: Sequence[tuple[int, int, int]]) -> list[int]:
    """Sample cascade indices stratified by size.

    `strata` is a sequence of `(low, high, k)` tuples: pick at most `k`
    cascades whose size lies in `[low, high)`.
    """
    picks: list[int] = []
    for low, high, k in strata:
        idx = np.where((sizes >= low) & (sizes < high))[0]
        if len(idx) == 0:
            continue
        picks += list(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return picks


def fit_active(csv_path: str | Path,
               n_cascades: int = 80,
               T_obs: float = 600.0,
               seed: int = 7,
               alpha: float = ALPHA_ASSUMED_DEFAULT,
               strata: Sequence[tuple[int, int, int]] = (
                   (30, 50, 20),
                   (50, 100, 25),
                   (100, 200, 20),
                   (200, 400, 15),
               )) -> list[CascadeFit]:
    """Faithful constrained MLE on a sample of ACTIVE cascades.

    Cascades are stratified by size and sampled per stratum; each cascade is
    optionally truncated to its first `T_obs` seconds (otherwise its full
    length). The fit is the multi-start SLSQP constrained MLE from
    `fit_one_cascade`.

    `n_cascades` is the (approximate) target total; the actual sample size is
    determined by the strata specification.
    """
    csv_path = Path(csv_path)
    times, marks, bounds = _read_active_cascades(csv_path)
    sizes = np.diff(bounds)

    rng_pick = np.random.default_rng(seed)
    picks = _stratified_pick(sizes, rng_pick, strata)
    # If the strata oversample, trim down to the requested total.
    if len(picks) > n_cascades:
        picks = picks[:n_cascades]

    out: list[CascadeFit] = []
    for k, ci in enumerate(picks):
        a, b = bounds[ci], bounds[ci + 1]
        t = times[a:b] - times[a]
        m = marks[a:b]
        mask = t <= T_obs
        if mask.sum() >= 15:
            t_used, m_used, T_used = t[mask], m[mask], T_obs
        else:
            t_used, m_used, T_used = t, m, float(t.max())
        if len(t_used) < 10 or T_used <= 0:
            continue
        result = fit_one_cascade(t_used, m_used, T_used, seed=k, alpha=alpha)
        if result is None:
            continue
        K_, beta_, c_, theta_ = result.x
        n_hat = branching_factor_closed_form(K_, beta_, c_, theta_, alpha=alpha)
        a_emp = hill_alpha(marks[a:b])
        out.append(CascadeFit(
            cascade_index=int(ci),
            N=int(len(t_used)),
            n_hat=float(n_hat),
            beta=float(beta_),
            alpha_emp=float(a_emp),
        ))
    return out
