"""Negative log-likelihood (NLL) for stationary Hawkes processes on an observation window [0, T].

Standard form (Daley & Vere-Jones 2003):

    log L = sum_i log lambda(t_i) - integral_0^T lambda(s) ds

Compensators in closed form:

- Unmarked power-law:
      integral_0^T lambda(s) ds = mu * T
                                + (alpha / eta) * sum_{t_i <= T} [delta^{-eta} - (T - t_i + delta)^{-eta}]
- Marked power-law:
      integral_0^T lambda(s) ds = mu * T
                                + (kappa / theta) * sum_{t_i <= T} m_i^beta
                                    * [c^{-theta} - (T - t_i + c)^{-theta}]

The synthetic-recovery experiments assume mu > 0 (no seed event), so every
event contributes to the log-lambda sum (i = 0 included).
"""
from __future__ import annotations

import numpy as np


_BIG = 1e12


def nll_unmarked_powerlaw(params: np.ndarray, times: np.ndarray, T: float) -> float:
    """params = [mu, alpha, delta, eta], all > 0."""
    mu, alpha, delta, eta = params
    if min(mu, alpha, delta, eta) <= 0.0:
        return _BIG
    s = 0.0
    for i in range(len(times)):
        if i == 0:
            lam = mu
        else:
            tau = times[i] - times[:i]
            lam = mu + alpha * np.sum((tau + delta) ** (-(1.0 + eta)))
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        s += np.log(lam)
    dT = T - times
    comp_kernel = (alpha / eta) * np.sum(delta ** (-eta) - (dT + delta) ** (-eta))
    comp = mu * T + comp_kernel
    val = -(s - comp)
    if not np.isfinite(val):
        return _BIG
    return val


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


def nll_unmarked_exp(params: np.ndarray, times: np.ndarray, T: float) -> float:
    """Exponential-kernel NLL via O(N) recursive state update.

    params = [mu, alpha, delta]. With
        S_i = sum_{j <= i} alpha * exp(-delta * (t_i - t_j))
    the recursion is
        S_0 = alpha,
        S_i = alpha + exp(-delta * (t_i - t_{i-1})) * S_{i-1},
    and for i >= 1, lambda(t_i^-) = mu + decay(S_{i-1}, t_i - t_{i-1}).
    Compensator: mu * T + (alpha / delta) * sum_i (1 - exp(-delta * (T - t_i))).
    """
    mu, alpha, delta = params
    if min(mu, alpha, delta) <= 0.0:
        return _BIG
    N = len(times)
    if N == 0:
        return mu * T  # background only
    log_lik = np.log(mu)  # lambda(t_0^-) = mu (no past events)
    S = alpha  # state right after event 0
    for i in range(1, N):
        gap = times[i] - times[i - 1]
        decay = np.exp(-delta * gap)
        lam = mu + S * decay
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        log_lik += np.log(lam)
        S = alpha + decay * S  # state right after event i
    comp = mu * T + (alpha / delta) * np.sum(1.0 - np.exp(-delta * (T - times)))
    val = -(log_lik - comp)
    if not np.isfinite(val):
        return _BIG
    return val


def nll_cascade_exp(params: np.ndarray, times: np.ndarray, T: float) -> float:
    """Cascade exponential Hawkes (Rizoiu form with mu = 0).

    The seed event (i = 0) is treated as a fixed trigger and is excluded from
    the log-lambda sum.

        lambda(t) = sum_{t_i < t} alpha * exp(-delta * (t - t_i))
        log L     = sum_{i >= 1} log lambda(t_i^-)
                    - (alpha / delta) * sum_i [1 - exp(-delta * (T - t_i))]

    O(N) recursion: S_i = alpha + exp(-delta * gap) * S_{i-1}, with
    S_0 = alpha (state right after the seed).
    """
    alpha, delta = params
    if min(alpha, delta) <= 0.0:
        return _BIG
    N = len(times)
    if N < 2:
        return _BIG  # a cascade needs at least seed + 1 endogenous event
    log_lik = 0.0
    S = alpha  # state right after the seed (i = 0)
    for i in range(1, N):
        gap = times[i] - times[i - 1]
        decay = np.exp(-delta * gap)
        lam = S * decay  # mu = 0
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        log_lik += np.log(lam)
        S = alpha + decay * S
    comp = (alpha / delta) * np.sum(1.0 - np.exp(-delta * (T - times)))
    val = -(log_lik - comp)
    if not np.isfinite(val):
        return _BIG
    return val


def nll_cascade_exp_piecewise_mu(params: np.ndarray, times: np.ndarray, T: float,
                                  K: int = 3) -> float:
    """Non-stationary cascade exponential Hawkes with a piecewise-constant background mu(t).

        lambda(t) = mu_k(t) + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

    where mu_k(t) = mu_k for t in [k * T / K, (k + 1) * T / K), k = 0..K-1.

    params = [mu_0, ..., mu_{K-1}, alpha, beta] (K + 2 parameters, all > 0).

    The seed event t_0 = 0 contributes to the log-lambda sum because the
    background is strictly positive.
    """
    K_int = int(K)
    if len(params) != K_int + 2:
        return _BIG
    mus = params[:K_int]
    alpha, beta = params[K_int], params[K_int + 1]
    if min(*mus, alpha, beta) <= 0.0:
        return _BIG
    if T <= 0:
        return _BIG

    interval_width = T / K_int
    N = len(times)
    log_lik = 0.0
    S = 0.0  # post-event sum_alpha * exp(-beta * (t - t_i)) state
    for i in range(N):
        t_i = times[i]
        k_idx = int(t_i / interval_width)
        if k_idx >= K_int:
            k_idx = K_int - 1
        mu_t = mus[k_idx]
        if i == 0:
            lam = mu_t  # no past events
        else:
            gap = t_i - times[i - 1]
            decay = np.exp(-beta * gap)
            lam = mu_t + S * decay
            S = decay * S  # decay before adding self-contribution
        if not np.isfinite(lam) or lam <= 0.0:
            return _BIG
        log_lik += np.log(lam)
        S = S + alpha  # add this event's contribution

    comp_bg = sum(mus) * interval_width
    comp_kernel = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    comp = comp_bg + comp_kernel

    val = -(log_lik - comp)
    if not np.isfinite(val):
        return _BIG
    return val


def nll_marked_powerlaw_fixed_beta(params4: np.ndarray, times: np.ndarray,
                                   marks: np.ndarray, T: float,
                                   beta_fixed: float = 1.0) -> float:
    """4-parameter [mu, kappa, c, theta] NLL when beta is held fixed at `beta_fixed`."""
    mu, kappa, c, theta = params4
    return nll_marked_powerlaw(np.array([mu, kappa, beta_fixed, c, theta]),
                                times, marks, T)

def nll_marked_powerlaw_cascade(params: np.ndarray, times: np.ndarray,
                                marks: np.ndarray, T: float) -> float:
    """Negative log-likelihood of the marked power-law Hawkes CASCADE model (§3.2):
    exogenous seed at times[0], background mu=0, seed excluded from the log-sum.
    Matches Rizoiu/Mishra marked_hawkes.R neg.log.likelihood (sum over time[-1], no
    background rate) and faithful_constraint._nll_marked_powerlaw (verified identical,
    NLL diff ~1e-14). Differs from nll_marked_powerlaw, which keeps log(mu) for the
    seed and fits mu>0 — that one does NOT implement the §3.2 cascade model.
    params = [kappa, beta, c, theta], all > 0.
    """
    K, beta, c, theta = params
    if K <= 0 or c <= 0 or theta <= 0 or beta < 0:
        return 1e12
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mb = marks ** beta
        if not np.all(np.isfinite(mb)):
            return 1e12
        comp = K * np.sum(mb * (1.0 / (theta * c ** theta)
                                - 1.0 / (theta * (T + c - times) ** theta)))
        if not np.isfinite(comp):
            return 1e12
        dt = times[:, None] - times[None, :] + c
        W = np.tril(dt ** (-(1.0 + theta)), k=-1)
        lam = (K * (W * mb[None, :]).sum(axis=1))[1:]   # [1:] excludes the seed
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            return 1e12
        ll = np.sum(np.log(lam))
    val = comp - ll
    return val if np.isfinite(val) else 1e12