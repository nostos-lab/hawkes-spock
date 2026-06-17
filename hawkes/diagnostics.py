"""Model-fit diagnostics: time-rescaling residuals and the Hardiman-Bouchaud sub-window estimator.

(1) Time-rescaling residuals (Ogata 1988; Brown et al. 2002):
    tau_k = integral_0^{t_k} lambda_hat(s) ds. When the fit is correct,
    {tau_k - tau_{k-1}} are i.i.d. Exp(1); a KS test against Exp(1) yields a
    goodness-of-fit statistic and p-value.

(2) Hardiman-Bouchaud (2014, arXiv:1312.6231) sub-window estimator:
    Fano factor F = Var(N_{Delta t}) / E(N_{Delta t}).
    For a stationary Hawkes process, F approaches 1 / (1 - n_star)^2,
    so n_hat_HB = 1 - 1 / sqrt(F) = 1 - sqrt(E / V).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest


def _compensator_unmarked(t_eval: float, t_hist: np.ndarray,
                          mu: float, alpha: float, delta: float, eta: float) -> float:
    past = t_hist < t_eval
    tp = t_hist[past]
    return mu * t_eval + (alpha / eta) * np.sum(
        delta ** (-eta) - (t_eval - tp + delta) ** (-eta))


def _compensator_cascade_exp(t_eval: float, t_hist: np.ndarray,
                              alpha: float, delta: float) -> float:
    """Cascade exponential Hawkes (mu = 0) compensator:

        integral_0^t lambda(s) ds = (alpha / delta) * sum_{t_i < t}
                                    [1 - exp(-delta * (t - t_i))]

    Assumes the seed event t_0 = 0 is included in `t_hist`.
    """
    past = t_hist < t_eval
    tp = t_hist[past]
    return (alpha / delta) * np.sum(1.0 - np.exp(-delta * (t_eval - tp)))


def time_rescaling_xi_cascade_exp(times: np.ndarray,
                                   params: np.ndarray) -> np.ndarray:
    """Cascade exponential Hawkes time-rescaling residuals.

    Returns the xi array of length len(times) - 1 (the seed event sets
    tau_0 = 0, so the first residual is tau_1 - 0).
    """
    alpha, delta = params
    taus = np.array([_compensator_cascade_exp(t, times, alpha, delta)
                     for t in times[1:]])  # skip seed (tau_0 = 0)
    if len(taus) == 0:
        return np.zeros(0)
    return np.diff(np.concatenate([[0.0], taus]))


def _compensator_marked(t_eval: float, t_hist: np.ndarray, m_hist: np.ndarray,
                        mu: float, kappa: float, beta: float, c: float, theta: float) -> float:
    past = t_hist < t_eval
    tp = t_hist[past]
    mp = m_hist[past]
    return mu * t_eval + (kappa / theta) * np.sum(
        (mp ** beta) * (c ** (-theta) - (t_eval - tp + c) ** (-theta)))


def time_rescaling_xi_unmarked(times: np.ndarray,
                                params: np.ndarray) -> np.ndarray:
    """xi_k = tau_k - tau_{k-1}, with tau_0 = 0. Returns array of length len(times)."""
    mu, alpha, delta, eta = params
    taus = np.array([_compensator_unmarked(t, times, mu, alpha, delta, eta)
                     for t in times])
    return np.diff(np.concatenate([[0.0], taus]))


def _compensator_unmarked_exp(t_eval: float, t_hist: np.ndarray,
                               mu: float, alpha: float, delta: float) -> float:
    past = t_hist < t_eval
    tp = t_hist[past]
    return mu * t_eval + (alpha / delta) * np.sum(1.0 - np.exp(-delta * (t_eval - tp)))


def time_rescaling_xi_unmarked_exp(times: np.ndarray,
                                    params: np.ndarray) -> np.ndarray:
    mu, alpha, delta = params
    taus = np.array([_compensator_unmarked_exp(t, times, mu, alpha, delta)
                     for t in times])
    return np.diff(np.concatenate([[0.0], taus]))


def time_rescaling_xi_marked(times: np.ndarray, marks: np.ndarray,
                              params: np.ndarray) -> np.ndarray:
    mu, kappa, beta, c, theta = params
    taus = np.array([_compensator_marked(t, times, marks, mu, kappa, beta, c, theta)
                     for t in times])
    return np.diff(np.concatenate([[0.0], taus]))


def ks_test_exp1(xi: np.ndarray) -> tuple[float, float]:
    """KS test of xi against Exp(1). Returns (statistic, p_value)."""
    if len(xi) == 0:
        return float("nan"), float("nan")
    res = kstest(xi, "expon")
    return float(res.statistic), float(res.pvalue)


def hardiman_bouchaud(times: np.ndarray, T: float,
                      delta_t: float) -> dict:
    """Sub-window estimator for the branching factor.

        n_hat_HB = 1 - sqrt(E[N_{Delta t}] / Var[N_{Delta t}])

    Returns a dict with diagnostic fields. `n_hb` is NaN when the estimator
    is inapplicable (e.g., Var <= E indicates a non-clustered, sub-Poisson,
    or trivial regime).
    """
    n_windows = int(np.floor(T / delta_t))
    out: dict = {"delta_t": delta_t, "T": T, "n_windows": n_windows,
                 "E_N": float("nan"), "Var_N": float("nan"),
                 "Fano": float("nan"), "n_hb": float("nan"),
                 "note": ""}
    if n_windows < 2:
        out["note"] = "fewer than 2 windows"
        return out
    bins = np.arange(0, (n_windows + 1) * delta_t, delta_t)
    counts, _ = np.histogram(times, bins=bins)
    E_N = float(counts.mean())
    V_N = float(counts.var(ddof=1))
    out["E_N"] = E_N
    out["Var_N"] = V_N
    if E_N <= 0:
        out["note"] = "E_N <= 0, no events"
        return out
    Fano = V_N / E_N
    out["Fano"] = Fano
    if Fano <= 1.0:
        out["note"] = "Fano <= 1 -- sub-Poisson or Poisson; H-B is not informative for clustered regimes"
        return out
    out["n_hb"] = 1.0 - 1.0 / np.sqrt(Fano)
    return out


def hb_sensitivity_sweep(times: np.ndarray, T: float,
                         delta_ts: list[float]) -> list[dict]:
    """Compute the H-B estimator across a list of window widths `delta_ts`."""
    return [hardiman_bouchaud(times, T, dt) for dt in delta_ts]
