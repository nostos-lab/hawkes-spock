"""Ogata thinning simulators for stationary Hawkes processes (unmarked and marked power-law, plus exponential).

Algorithm (Ogata 1981):

1. At the current time t, hold an upper bound `lam_bar` on the intensity
   (the post-jump value right after the previous accepted event).
2. Sample a candidate gap `dt ~ Exp(lam_bar)` and set `t_cand = t + dt`.
3. Evaluate the true intensity `lam(t_cand)` and accept with probability
   `lam(t_cand) / lam_bar`.
4. On accept, append the event and update `lam_bar` to `lam + phi_max(0+)`.
5. On reject, the intensity is only decreasing, so `lam_bar` remains a valid
   bound.

The marked variant has the same structure but additionally samples a mark
when an event is accepted.
"""
from __future__ import annotations

import numpy as np

from . import kernels


def simulate_unmarked_powerlaw(mu: float, alpha: float, delta: float, eta: float,
                                T: float, rng: np.random.Generator,
                                max_events: int = 1_000_000) -> np.ndarray:
    """Simulate an unmarked power-law Hawkes process. Returns event times up to T."""
    events: list[float] = []
    phi0 = kernels.phi_unmarked_max(alpha, delta, eta)
    t = 0.0
    lam_bar = mu  # initial upper bound on the intensity (background only)
    while True:
        u = rng.uniform()
        if u <= 0:
            continue
        dt = -np.log(u) / lam_bar
        t_cand = t + dt
        if t_cand >= T:
            break
        # Evaluate the true intensity at the candidate time.
        if events:
            ts = np.asarray(events)
            lam = mu + alpha * np.sum((t_cand - ts + delta) ** (-(1.0 + eta)))
        else:
            lam = mu
        if rng.uniform() * lam_bar <= lam:
            events.append(t_cand)
            # Intensity jump right after the new event: lam + phi(0+).
            lam_bar = lam + phi0
            if len(events) >= max_events:
                break
        # On reject: no new event, the intensity only decays, so lam_bar stays valid.
        t = t_cand
    return np.asarray(events)


def simulate_marked_powerlaw(mu: float, kappa: float, beta: float, c: float, theta: float,
                              T: float, rng: np.random.Generator,
                              mark_mu_log10: float = 2.5, mark_sigma_log10: float = 1.0,
                              max_events: int = 1_000_000
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a marked power-law Hawkes process. Returns (times, marks).

    Mark distribution: log10(m) ~ Normal(mark_mu_log10, mark_sigma_log10).
    """
    times: list[float] = []
    marks: list[float] = []
    t = 0.0
    lam_bar = mu
    while True:
        u = rng.uniform()
        if u <= 0:
            continue
        dt = -np.log(u) / lam_bar
        t_cand = t + dt
        if t_cand >= T:
            break
        if times:
            ts = np.asarray(times)
            ms = np.asarray(marks)
            lam = mu + kappa * np.sum((ms ** beta) * (t_cand - ts + c) ** (-(1.0 + theta)))
        else:
            lam = mu
        if rng.uniform() * lam_bar <= lam:
            m_new = float(10.0 ** rng.normal(mark_mu_log10, mark_sigma_log10))
            times.append(t_cand)
            marks.append(m_new)
            jump = kernels.phi_marked_max_for(m_new, kappa, beta, c, theta)
            lam_bar = lam + jump
            if len(times) >= max_events:
                break
        t = t_cand
    return np.asarray(times), np.asarray(marks)


def simulate_unmarked_exp(mu: float, alpha: float, delta: float,
                           T: float, rng: np.random.Generator,
                           max_events: int = 1_000_000) -> np.ndarray:
    """Simulate an unmarked exponential-kernel Hawkes process via Ogata thinning.

    phi(tau) = alpha * exp(-delta * tau). The state
    S = sum_{t_j < t} alpha * exp(-delta * (t - t_j)) supports O(1) intensity
    updates: S += alpha after a new event, S *= exp(-delta * dt) over a gap dt.
    """
    events: list[float] = []
    t = 0.0
    S = 0.0  # contribution of past events to the current intensity
    while True:
        lam_bar = mu + S
        if lam_bar <= 0:
            lam_bar = max(mu, 1e-12)
        u = rng.uniform()
        if u <= 0:
            continue
        dt = -np.log(u) / lam_bar
        t_cand = t + dt
        if t_cand >= T:
            break
        S_at_cand = S * np.exp(-delta * dt)
        lam_at_cand = mu + S_at_cand
        if rng.uniform() * lam_bar <= lam_at_cand:
            events.append(t_cand)
            S = S_at_cand + alpha
            if len(events) >= max_events:
                break
        else:
            S = S_at_cand
        t = t_cand
    return np.asarray(events)


def apply_burnin(times: np.ndarray, T_total: float,
                 fraction: float = 0.1) -> tuple[np.ndarray, float]:
    """Drop the leading `fraction` of events to mitigate edge effects, then re-zero time.

    Returns (times_shifted, T_effective).
    """
    if len(times) == 0:
        return times, T_total
    n_burn = int(fraction * len(times))
    if n_burn == 0:
        return times, T_total
    t0 = times[n_burn]
    return times[n_burn:] - t0, T_total - t0


def apply_burnin_marked(times: np.ndarray, marks: np.ndarray, T_total: float,
                        fraction: float = 0.1
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Marked analogue of `apply_burnin`. Returns (times_shifted, marks, T_effective)."""
    if len(times) == 0:
        return times, marks, T_total
    n_burn = int(fraction * len(times))
    if n_burn == 0:
        return times, marks, T_total
    t0 = times[n_burn]
    return times[n_burn:] - t0, marks[n_burn:], T_total - t0
