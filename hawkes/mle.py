"""Multi-start L-BFGS-B MLE wrapper.

Strategy:

- Hawkes negative log-likelihoods are non-convex, so we sample N initial
  points from log-uniform distributions in each free parameter.
- For each start we minimise the NLL with `scipy.optimize.minimize`
  (method = 'L-BFGS-B', jac = '2-point' finite-difference gradient).
- The fit with the lowest NLL among the converged runs is returned as
  `best`; all attempts are kept in `all_fits`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    params: list[float]
    nll: float
    converged: bool
    n_iter: int
    init: list[float]
    message: str = ""


@dataclass
class MultiStartResult:
    best: FitResult | None
    all_fits: list[FitResult] = field(default_factory=list)
    n_converged: int = 0
    n_total: int = 0


def fit_multi_start(nll_fn: Callable,
                    nll_args: tuple,
                    init_sampler: Callable[[np.random.Generator], np.ndarray],
                    bounds: Sequence[tuple[float, float]],
                    n_inits: int = 20,
                    rng: np.random.Generator | None = None,
                    options: dict | None = None) -> MultiStartResult:
    """L-BFGS-B multi-start fit.

    nll_fn(params, *nll_args) -> float
    init_sampler(rng) -> initial parameter vector (np.ndarray matching `bounds`)
    """
    rng = rng if rng is not None else np.random.default_rng()
    options = options or {"ftol": 1e-10, "gtol": 1e-8, "maxiter": 5000}

    fits: list[FitResult] = []
    for _ in range(n_inits):
        x0 = init_sampler(rng)
        try:
            res = minimize(nll_fn, x0, args=nll_args,
                           method="L-BFGS-B", jac="2-point",
                           bounds=list(bounds), options=options)
        except Exception as exc:  # noqa: BLE001
            fits.append(FitResult(params=x0.tolist(), nll=float("inf"),
                                  converged=False, n_iter=0,
                                  init=x0.tolist(), message=f"raised:{exc}"))
            continue
        if not np.isfinite(res.fun):
            fits.append(FitResult(params=res.x.tolist(), nll=float("inf"),
                                  converged=False, n_iter=int(res.nit),
                                  init=x0.tolist(), message="nonfinite_nll"))
            continue
        fits.append(FitResult(
            params=res.x.tolist(), nll=float(res.fun),
            converged=bool(res.success), n_iter=int(res.nit),
            init=x0.tolist(), message=str(res.message)))

    fits.sort(key=lambda r: (not r.converged, r.nll))
    converged = [f for f in fits if f.converged]
    best = converged[0] if converged else (fits[0] if fits else None)
    return MultiStartResult(best=best, all_fits=fits,
                            n_converged=len(converged), n_total=len(fits))


# ----- Initial-point samplers -----

def sample_init_marked_free_beta(rng: np.random.Generator) -> np.ndarray:
    """[mu, kappa, beta, c, theta] for beta-free marked Hawkes."""
    mu = 10.0 ** rng.uniform(np.log10(0.01), np.log10(5.0))
    kappa = 10.0 ** rng.uniform(-3.0, 1.0)
    beta = rng.uniform(0.3, 2.0)
    c = 10.0 ** rng.uniform(np.log10(0.5), np.log10(2000.0))
    theta = rng.uniform(0.3, 3.0)
    return np.array([mu, kappa, beta, c, theta])

def sample_init_cascade_marked(rng):
    """[κ, β, c, θ] for cascade marked Hawkes (μ=0)."""
    kappa = 10.0 ** rng.uniform(-2.0, 1.0)
    beta  = rng.uniform(0.1, 3.0)
    c     = 10.0 ** rng.uniform(0.0, np.log10(5000.0))
    theta = rng.uniform(0.05, 3.0)
    return np.array([kappa, beta, c, theta])