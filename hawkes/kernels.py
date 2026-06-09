"""Hawkes triggering kernels, branching-factor closed forms, and log-normal mark helpers.

Definitions (paper §2.1, §3.1):

- Unmarked power-law kernel:
      phi(tau) = alpha * (tau + delta)^{-(1 + eta)}
      n_star  = alpha * delta^{-eta} / eta
- Marked power-law kernel:
      phi_m(tau) = kappa * m^beta * (tau + c)^{-(1 + theta)}
      n_star     = kappa * E[m^beta] * c^{-theta} / theta
- Mark distribution: log10(m) ~ Normal(mu_log10, sigma_log10).
  E[m^beta] follows analytically from the Gaussian moment-generating function.
"""
from __future__ import annotations

import numpy as np


def phi_unmarked(tau: np.ndarray, alpha: float, delta: float, eta: float) -> np.ndarray:
    """Unmarked power-law kernel phi(tau) = alpha * (tau + delta)^{-(1 + eta)} for tau >= 0."""
    return alpha * (tau + delta) ** (-(1.0 + eta))


def branching_factor_unmarked(alpha: float, delta: float, eta: float) -> float:
    """n_star = integral_0^inf alpha * (tau + delta)^{-(1 + eta)} dtau = alpha * delta^{-eta} / eta."""
    return alpha * (delta ** (-eta)) / eta


def phi_marked(tau: np.ndarray, mark: float | np.ndarray,
               kappa: float, beta: float, c: float, theta: float) -> np.ndarray:
    """Marked power-law kernel phi_m(tau) = kappa * m^beta * (tau + c)^{-(1 + theta)} for tau >= 0."""
    return kappa * (mark ** beta) * (tau + c) ** (-(1.0 + theta))


def branching_factor_marked(kappa: float, beta: float, c: float, theta: float,
                            E_m_beta: float) -> float:
    """n_star = kappa * E[m^beta] * c^{-theta} / theta (plug-in via mark-distribution moment)."""
    return kappa * E_m_beta * (c ** (-theta)) / theta


def lognormal_E_m_beta(beta: float,
                       mu_log10: float = 2.5, sigma_log10: float = 1.0) -> float:
    """Analytic E[m^beta] when log10(m) ~ Normal(mu_log10, sigma_log10).

    log(m) = ln(10) * log10(m) ~ Normal(ln10 * mu_log10, (ln10 * sigma_log10)^2),
    so E[m^beta] = E[exp(beta * ln m)]
                = exp(beta * ln10 * mu_log10 + 0.5 * (beta * ln10 * sigma_log10)^2).
    """
    ln10 = np.log(10.0)
    return float(np.exp(beta * ln10 * mu_log10 + 0.5 * (beta * ln10 * sigma_log10) ** 2))


def sample_lognormal_marks(n: int, rng: np.random.Generator,
                           mu_log10: float = 2.5, sigma_log10: float = 1.0) -> np.ndarray:
    """Draw n marks with log10(m) ~ Normal(mu_log10, sigma_log10)."""
    log10_m = rng.normal(mu_log10, sigma_log10, size=n)
    return 10.0 ** log10_m


def phi_unmarked_max(alpha: float, delta: float, eta: float) -> float:
    """Maximum of phi(tau) at tau = 0+, used as the Ogata-thinning jump bound."""
    return alpha * (delta ** (-(1.0 + eta)))


# ----- Exponential kernel (Phase 0 baseline; identifiable two-parameter kernel) -----

def phi_exp(tau: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Exponential kernel phi(tau) = alpha * exp(-delta * tau) for tau >= 0."""
    return alpha * np.exp(-delta * tau)


def branching_factor_exp(alpha: float, delta: float) -> float:
    """n_star = integral_0^inf alpha * exp(-delta * tau) dtau = alpha / delta."""
    return alpha / delta


def phi_exp_max(alpha: float) -> float:
    """Maximum of phi(tau) at tau = 0 for the exponential kernel = alpha."""
    return alpha


def phi_marked_max_for(mark: float, kappa: float, beta: float,
                       c: float, theta: float) -> float:
    """Maximum of phi_m(tau) at tau = 0+ for a given mark, used as the Ogata-thinning jump bound."""
    return kappa * (mark ** beta) * (c ** (-(1.0 + theta)))
