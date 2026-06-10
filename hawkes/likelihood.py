"""Negative log-likelihood (NLL) for marked power-law Hawkes processes on [0, T].

Standard form (Daley & Vere-Jones 2003):

    log L = sum_i log lambda(t_i) - integral_0^T lambda(s) ds

Marked power-law compensator in closed form:

      integral_0^T lambda(s) ds = mu * T
                                + (kappa / theta) * sum_{t_i <= T} m_i^beta
                                    * [c^{-theta} - (T - t_i + c)^{-theta}]

`nll_marked_powerlaw` is the stationary form with mu > 0 (every event, i = 0
included, contributes to the log-lambda sum). `nll_cascade_marked_powerlaw` is
the Rizoiu (2017) cascade form with mu = 0, where the seed event t_0 is a fixed
exogenous trigger excluded from the log-lambda sum.
"""
from __future__ import annotations

import numpy as np


_BIG = 1e12


def nll_marked_powerlaw(params: np.ndarray, times: np.ndarray, marks: np.ndarray,
                        T: float) -> float:
    """params = [mu, kappa, beta, c, theta], all > 0."""
    mu, kappa, beta, c, theta = params
    if min(mu, kappa, beta, c, theta) <= 0.0:
        return _BIG
    s = 0.0
    for i in range(len(times)):
        if i == 0:
            lam = mu
        else:
            tau = times[i] - times[:i]
            lam = mu + kappa * np.sum((marks[:i] ** beta) * (tau + c) ** (-(1.0 + theta)))
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        s += np.log(lam)
    dT = T - times
    comp_kernel = (kappa / theta) * np.sum((marks ** beta) *
                                           (c ** (-theta) - (dT + c) ** (-theta)))
    comp = mu * T + comp_kernel
    val = -(s - comp)
    if not np.isfinite(val):
        return _BIG
    return val


def nll_cascade_marked_powerlaw(params, times, marks, T):
    """Cascade marked power-law Hawkes (Rizoiu 2017 form, μ=0).

    seed event (i=0) fixed as exogenous trigger — except from log λ sum.
    λ(t) = Σ_{t_i<t} κ m_i^β (t − t_i + c)^{−(1+θ)}
    log L = Σ_{i≥1} log λ(t_i^-) − (κ/θ) Σ_i m_i^β [c^{−θ} − (T − t_i + c)^{−θ}]
    params = [κ, β, c, θ], all > 0
    """
    kappa, beta, c, theta = params
    if min(kappa, beta, c, theta) <= 0.0:
        return _BIG
    s = 0.0
    for i in range(1, len(times)):
        tau = times[i] - times[:i]
        lam = kappa * np.sum((marks[:i] ** beta) * (tau + c) ** (-(1.0 + theta)))
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        s += np.log(lam)
    dT = T - times[times < T]
    m_w = marks[times < T]
    comp = (kappa / theta) * np.sum((m_w ** beta) *
                                    (c ** (-theta) - (dT + c) ** (-theta)))
    val = -(s - comp)
    return val if np.isfinite(val) else _BIG
