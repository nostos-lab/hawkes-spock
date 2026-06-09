# Results

Pre-computed artifacts produced by `notebooks/main.ipynb`. Re-running the
notebook overwrites these files. The committed values are the frozen
snapshots referenced by the top-level README and by the paper.

## Real-data fit (§4.1–4.2)

| File | Description |
|---|---|
| `spock_fit.json` | Best free-MLE fit on the Spock cascade plus the top-five ridge fits. The unconstrained MLE does not recover the reported parameters (unmarked plug-in `n* ~ 1.19` vs. reported `n* = 0.92`). |
| `fig01_spock_cascade.png` | Spock cascade event stem-plot (full timeline + 600 s observation window). **(fig 1)** |
| `fig02_intensity_fit.png` | Fitted marked-Hawkes intensity overlaid on observed events. **(fig 2)** |
| `fig03_residual_qq.png` | Time-rescaled residuals Q–Q vs. `Exp(1)`; KS not rejected (p = 0.305) but with systematic upper-tail deviation. **(fig 3)** |
| `fig04_likelihood_ridge.png` | The likelihood ridge: five near-identical fits span `κ` over ~6 orders of magnitude (`ΔNLL = 0.17`); the reported point lies on the ridge. **(fig 4)** |

## Synthetic recovery and cross-validation (§4.3–4.7)

Frozen experimental snapshots (loaded as input by `main.ipynb` when
`FORCE_RECOMPUTE = False`):

| File | Description |
|---|---|
| `phase0_E{1,2,3}_results.json` | 20-rep recovery of an *exponential* Hawkes kernel at `n* = 0.3 / 0.7 / 0.9`. Implementation-validity check (E2 and E3 within tolerance). |
| `phase1_R{1,2,3}_results.json` | 20-rep recovery of an *unmarked power-law* Hawkes kernel at the same `n*`. Exposes the `(α, δ, η)` ridge; worsens toward criticality. |
| `hb_spock.json` | Hardiman–Bouchaud `Δt` sub-window sweep on the Spock cascade (`Δt ∈ {60, 120, 300, 600, 1800}` s). Independent cross-check; `n* < 1` across all `Δt`. |

Derived outputs (produced when the notebook runs):

| File | Description |
|---|---|
| `phase2_marked.json` | Marked power-law recovery, three regimes × 10 reps (seeds 424242 / 70707 / 909090). `n*` medians 0.16 → 2.69 → 18.4; `β` medians 1.14 → 1.65 → 1.63. |
| `active80_faithful.json` | ACTIVE faithful-constraint fits (80 cascades, SLSQP `n* < 1`). 94 % subcritical, ~44 % at the `n* ≈ 0.92` boundary; empirical `α` median 1.18. |
| `diag2_alpha.json` | Per-cascade Hill-estimated mark Pareto exponent `α` for the 80 ACTIVE cascades. |
| `final_summary.json` | Aggregated verdict combining the Spock free / constrained fits, Phase 0 / 1 / 2, Hardiman–Bouchaud, and ACTIVE. |
| `fig05_phase0_distribution.png` | Phase 0 `n̂*` histograms per regime. **(fig 5)** |
| `fig06_phase1_distribution.png` | Phase 1 `n̂*` histograms per regime. **(fig 6)** |
| `fig07_hb_spock.png` | Hardiman–Bouchaud `Δt` sweep (`Δt`-dependent estimate vs. single-value MLE reference lines). **(fig 7)** |
| `fig08_phase2_marked.png` | Phase 2: `n̂*` worsening toward criticality + `β` over-estimation. **(fig 8)** |
| `fig09_active_constraint.png` | ACTIVE: subcriticality depends jointly on the `n* < 1` constraint and the `α = 2.016` mark assumption. **(fig 9)** |

## Files generated locally (not pre-included)

The following are produced when you run the notebook with the data in place,
and are **not** committed here because they require the external Spock CSV:

* `fig01_spock_cascade.png`, `fig02_intensity_fit.png`, `fig03_residual_qq.png`
* `fig05_phase0_distribution.png`, `fig06_phase1_distribution.png`
* `final_summary.json` (aggregated verdict)

All other JSON and PNG artifacts referenced above are included.
