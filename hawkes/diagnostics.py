"""Model-fit diagnostics: time-rescaling residuals.

Time-rescaling residuals (Ogata 1988; Brown et al. 2002):
    tau_k = integral_0^{t_k} lambda_hat(s) ds. When the fit is correct,
    {tau_k - tau_{k-1}} are i.i.d. Exp(1); a KS test against Exp(1) yields a
    goodness-of-fit statistic and p-value.
"""
from __future__ import annotations

import numpy as np


def time_rescaling_xi_cascade_marked(times: np.ndarray, marks: np.ndarray,
                                     params: np.ndarray) -> np.ndarray:
    """Cascade marked power-law Hawkes (mu = 0) time-rescaling residuals.

    The seed event (i = 0) is a fixed exogenous trigger, so tau_0 = 0 and the
    residuals are computed on events 1..N-1:

        tau_k = (kappa / theta) * sum_{t_i < t_k} m_i^beta
                * [c^{-theta} - (t_k - t_i + c)^{-theta}]
        xi_k  = tau_k - tau_{k-1}

    params = [kappa, beta, c, theta]. Returns the xi array of length
    len(times) - 1.
    """
    kappa, beta, c, theta = params

    def _comp(t_eval):
        past = times < t_eval
        tp = times[past]
        mp = marks[past]
        return (kappa / theta) * np.sum(
            (mp ** beta) * (c ** (-theta) - (t_eval - tp + c) ** (-theta)))

    taus = np.array([_comp(tk) for tk in times[1:]])  # skip seed (tau_0 = 0)
    if len(taus) == 0:
        return np.zeros(0)
    return np.diff(np.concatenate([[0.0], taus]))
