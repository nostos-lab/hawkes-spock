# Results

Pre-computed artifacts produced by `notebooks/main.ipynb`. Re-running the
notebook overwrites these files. The committed values are the frozen
snapshots referenced by the top-level README and by the paper.

## Real-data fit (§4.1–4.2)

| File | Description |
|---|---|
| `spock_fit.json` | Free-MLE fit on the Spock cascade (4-parameter seed-excluded `mu = 0` model) plus the top-five ridge fits and the constrained fit. The unconstrained MLE does not recover the reported parameters: free fit `(κ, β, c, θ) = (7.6e4, 2.58, 1278, 4.69)`, plug-in `n* = 1.18` (NLL 147.34), versus the constrained fit reproducing the reported `n* = 0.92` (closed-form, NLL 148.11). |
| `fig01_spock_cascade.png` | Spock cascade event stem-plot (full timeline + 600 s observation window, `N_obs = 43`). **(fig 1)** |
| `fig02_intensity_fit.png` | Fitted marked-Hawkes intensity overlaid on observed events (peak ≈ 0.16). **(fig 2)** |
| `fig03_residual_qq.png` | Time-rescaled residuals Q–Q vs. `Exp(1)`; KS not rejected (p = 0.306) but with systematic upper-tail deviation. **(fig 3)** |
| `fig04_profile_kappa.png` | κ profile likelihood: fixing `κ` over six decades (`10²–10⁸`) leaves `ΔNLL ≤ 0.012` (left), and the plug-in `n*` stays supercritical (`1.17–1.20`) along the profile (right). The kernel scale is non-identified. **(fig 4)** |
| `fig05_fisher_spectrum.png` | Observed Fisher information at the unconstrained optimum: eigenvalue spectrum with condition number `≈ 1.2e15` (left), and the softest eigenvector aligned with `κ` while the stiffest mixes `β, θ` (right). **(fig 5)** |

## Synthetic recovery and cross-validation (§4.3–4.7)

Frozen experimental snapshots (loaded as input by `main.ipynb` when
`FORCE_RECOMPUTE = False`):

| File | Description |
|---|---|
| `phase0_E{1,2,3}_results.json` | 20-rep recovery of an *exponential* Hawkes kernel at `n* = 0.3 / 0.7 / 0.9`. Implementation-validity check (`n̂*` median 0.299 / 0.677 / 0.888; KS 20/20). |
| `phase1_R{1,2,3}_results.json` | 20-rep recovery of an *unmarked power-law* Hawkes kernel at the same `n*`. Exposes the `(α, δ, η)` ridge; `n̂*` median 0.281 / 0.510 / 96.6, worsening toward criticality. |
| `hb_spock.json` | Hardiman–Bouchaud `Δt` sub-window sweep on the Spock cascade (`Δt ∈ {60, 120, 300, 600, 1800}` s). Independent cross-check; `n̂_HB = 0.51–0.88`, below 1 across all `Δt`. |

Derived outputs (produced when the notebook runs):

| File | Description |
|---|---|
| `phase2_marked.json` | Marked power-law recovery, three regimes × 10 reps (seeds 424242 / 70707 / 909090). `n*` medians 0.16 → 2.69 → 18.4; `β` medians 1.14 → 1.65 → 1.63. |
| `active80_faithful.json` | ACTIVE faithful-constraint fits (80 cascades, SLSQP `n* < 1`). 91 % subcritical, ~46 % at the `n* ≈ 0.92` boundary; empirical mark `α` median 1.18; closed form void (`β ≥ α − 1`) in 55 %. |
| `diag2_alpha.json` | Per-cascade Hill-estimated mark Pareto exponent `α` for the 80 ACTIVE cascades. |
| `final_summary.json` | Aggregated verdict combining the Spock free / constrained fits, Phase 0 / 1 / 2, Hardiman–Bouchaud, and ACTIVE. |
| `fig06_phase0_distribution.png` | Phase 0 `n̂*` histograms per regime. **(fig 6)** |
| `fig07_phase1_distribution.png` | Phase 1 `n̂*` histograms per regime. **(fig 7)** |
| `fig08_hb_spock.png` | Hardiman–Bouchaud `Δt` sweep (`Δt`-dependent estimate vs. single-value MLE reference lines). **(fig 8)** |
| `fig09_phase2_marked.png` | Phase 2: `n̂*` worsening toward criticality + `β` over-estimation. **(fig 9)** |
| `fig10_active_constraint.png` | ACTIVE: subcriticality depends jointly on the `n* < 1` constraint and the `α = 2.016` mark assumption. **(fig 10)** |

## Files generated locally (not pre-included)

The following are produced when you run the notebook with the data in place,
and are **not** committed here because they require the external Spock CSV:

* `fig01_spock_cascade.png`, `fig02_intensity_fit.png`, `fig03_residual_qq.png`
* `fig04_profile_kappa.png`, `fig05_fisher_spectrum.png`
* `fig06_phase0_distribution.png`, `fig07_phase1_distribution.png`
* `final_summary.json` (aggregated verdict)

All other JSON and PNG artifacts referenced above are included.

## Frozen dates

* Real-data cascade fit refrozen 2026-06-17 (`spock_fit.json`, `final_summary.json`,
  4-parameter seed-excluded `mu = 0` schema)
* H-B cross-check 2026-05-24
* Phase 0 / Phase 1 supplement 2026-05-24
* Phase 2 and ACTIVE generated in this study
