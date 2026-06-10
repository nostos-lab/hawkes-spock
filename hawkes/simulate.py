"""Ogata thinning simulator for the marked power-law Hawkes process.

Algorithm (Ogata 1981):

1. At the current time t, hold an upper bound `lam_bar` on the intensity
   (the post-jump value right after the previous accepted event).
2. Sample a candidate gap `dt ~ Exp(lam_bar)` and set `t_cand = t + dt`.
3. Evaluate the true intensity `lam(t_cand)` and accept with probability
   `lam(t_cand) / lam_bar`.
4. On accept, append the event (and a sampled mark) and update `lam_bar` to
   `lam + phi_max(0+)`.
5. On reject, the intensity is only decreasing, so `lam_bar` remains a valid
   bound.
"""
from __future__ import annotations

import numpy as np

from . import kernels


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


def apply_burnin_marked(times: np.ndarray, marks: np.ndarray, T_total: float,
                        fraction: float = 0.1
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Drop the leading `fraction` of events to mitigate edge effects, then re-zero time.

    Returns (times_shifted, marks, T_effective).
    """
    if len(times) == 0:
        return times, marks, T_total
    n_burn = int(fraction * len(times))
    if n_burn == 0:
        return times, marks, T_total
    t0 = times[n_burn]
    return times[n_burn:] - t0, marks[n_burn:], T_total - t0
