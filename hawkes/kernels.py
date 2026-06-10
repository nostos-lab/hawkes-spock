"""Marked Hawkes triggering kernel, branching-factor closed form, and log-normal mark helper.

Definitions (paper §2.1, §3.1):

- Marked power-law kernel:
      phi_m(tau) = kappa * m^beta * (tau + c)^{-(1 + theta)}
      n_star     = kappa * E[m^beta] * c^{-theta} / theta
- Mark distribution: log10(m) ~ Normal(mu_log10, sigma_log10).
  E[m^beta] follows analytically from the Gaussian moment-generating function.
"""
from __future__ import annotations

import numpy as np


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


def phi_marked_max_for(mark: float, kappa: float, beta: float,
                       c: float, theta: float) -> float:
    """Maximum of phi_m(tau) at tau = 0+ for a given mark, used as the Ogata-thinning jump bound."""
    return kappa * (mark ** beta) * (c ** (-(1.0 + theta)))
