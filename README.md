# Assumption-Dependence of the Branching Ratio in a Power-Law Hawkes Cascade — A Reproduction Audit

[![DOI](https://zenodo.org/badge/1264160587.svg)](https://doi.org/10.5281/zenodo.20615710)

Code and frozen result artifacts for a reproduction audit of the marked
power-law Hawkes fit reported for the "Spock" retweet cascade (Rizoiu *et al.*
2017). Re-estimating the model under the seed-excluded, zero-background
specification that the worked example defines, the reported subcritical
branching ratio (`n* = 0.92`) is found to be determined by estimation
assumptions rather than by the cascade data. The release also includes the
synthetic recovery and cross-validation experiments that separate
*implementation validity* from *model identifiability*: exponential and
power-law Hawkes kernel recovery on synthetic data with known ground truth, a
Hardiman–Bouchaud cross-check on the Spock cascade, and a constraint-faithful
reproduction on the ACTIVE dataset.

> **Data is not redistributed in this repository.** The analyses consume one
> external CSV (`example_book.csv`) and the ACTIVE dataset, both from the
> upstream Rizoiu/Mishra repositories under their own (GPLv3 / CC BY-NC 4.0)
> terms. See [Data](#data) for how to obtain them. Everything else in this
> repository is original work released under the MIT License.

## Repository layout

```
.
├── README.md                       # this file
├── LICENSE                         # MIT (all code, notebooks, and results)
├── CITATION.cff                    # citation metadata
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── hawkes/                         # Importable Python toolkit (MIT)
│   ├── __init__.py
│   ├── kernels.py                  # phi(tau), n* closed-forms, log-normal mark helpers
│   ├── simulate.py                 # Ogata thinning simulators (unmarked / marked / exp)
│   ├── likelihood.py               # negative log-likelihoods (power-law, exp, cascade mu=0)
│   ├── mle.py                      # multi-start L-BFGS-B wrapper + init samplers
│   ├── diagnostics.py              # time-rescaling residuals + Hardiman–Bouchaud
│   └── faithful_constraint.py      # Mishra-closed-form constrained MLE (SLSQP n*<1)
│
├── notebooks/
│   └── main.ipynb                  # single notebook reproducing every result and figure
│
├── scripts/
│   └── verify_against_rizoiu.R     # R cross-check: our cascade NLL vs marked_hawkes.R
│
├── data/
│   └── README.md                   # how to obtain example_book.csv and ACTIVE (NOT redistributed)
│
└── results/
    ├── README.md                   # describes every artifact below
    ├── spock_fit.json              # free-MLE fit on Spock + top-5 ridge fits (4-param cascade)
    ├── final_summary.json          # aggregated verdict across all experiments
    ├── phase0_E{1,2,3}_results.json  # exponential kernel synthetic recovery
    ├── phase1_R{1,2,3}_results.json  # unmarked power-law synthetic recovery
    ├── phase2_marked.json          # marked power-law synthetic recovery
    ├── active80_faithful.json      # ACTIVE 80-cascade faithful-constraint fits
    ├── diag2_alpha.json            # per-cascade Hill mark-distribution exponent
    ├── hb_spock.json               # Hardiman–Bouchaud Δt sweep on Spock
    └── fig{01..10}_*.png           # ten figures referenced by the paper
```

## What this release demonstrates

1. **The reported value is a product of estimation assumptions, not the data.**
   Re-fitting the Spock cascade under the seed-excluded, `mu = 0` cascade
   likelihood that the example text specifies, three assumptions each control
   the subcritical-versus-supercritical verdict independently:
   * the imposed `n* < 1` constraint — the constrained fit reproduces the
     reported `n* = 0.92` (closed-form, Pareto `α = 2.016`), but removing the
     constraint lowers the negative log-likelihood (148.11 → 147.34);
   * the validity window `β < α − 1 = 1.016` of the closed-form ratio — the
     unconstrained fit lands at `β = 2.58`, where the closed-form ratio is
     undefined, and this holds across the entire non-identified ridge;
   * the assumed mark distribution — at the *same* constrained parameters,
     replacing the Pareto exponent with the empirical mark moment moves `n*`
     from 0.92 (subcritical) to 1.28 (supercritical), a factor of 1.38.

2. **The mechanism is non-identifiability of the kernel scale `κ`.**
   * **Profile likelihood** — fixing `κ` across six decades (`10²–10⁸`) and
     re-optimizing `(β, c, θ)` leaves the NLL flat to within `ΔNLL ≤ 0.012`;
     `κ` is matched at essentially equal likelihood by a compensating
     `(β, c, θ)`. On the plug-in mark scale the profiled region stays
     supercritical throughout (`n* ≈ 1.17–1.20`).
   * **Fisher information** — at the unconstrained optimum the 4×4 observed
     Fisher has condition number `≈ 1.2 × 10¹⁵`, and the eigenvector of its
     smallest eigenvalue is aligned with the `κ` axis (`κ` component `≈ 1.0`,
     others `≤ 10⁻⁴`). `κ` is non-identified; `(β, c, θ)` are identified.

3. **Implementation vs. model separation.**
   * **Exponential kernel synthetic recovery** (`n* = 0.3 / 0.7 / 0.9`) confirms
     the likelihood / multi-start / time-rescaling code recovers ground truth.
   * **Unmarked power-law synthetic recovery** at the same regimes — the
     `(α, δ, η)` ridge collapses individual identification and worsens toward
     criticality, even at hundreds–thousands of events.
   * **Marked power-law synthetic recovery** at `n* = 0.3 / 0.7 / 0.9`, `β =
     1.3` — the marked kernel is likewise non-identifiable; the medians of
     `n*` diverge (0.16 → 2.69 → 18.4) via systematic over-estimation of `β`.
   * **Hardiman–Bouchaud Δt sweep** on the Spock cascade — places `n*` below 1
     across all observed window widths (0.51–0.88), a path independent of the
     likelihood ridge.
   * **ACTIVE faithful-constraint check** (80 cascades) — faithfully
     reproducing the `n* < 1` constraint of Mishra *et al.* (2016) keeps 91 %
     subcritical, yet ~46 % pile up at the constraint boundary (`n* ≈ 0.92`),
     and the empirical mark exponent (median 1.18) mismatches the assumed
     `α = 2.016`, voiding the closed form in 55 % of cascades.

4. **Conclusion.** The reported subcritical branching ratio of this cascade is
   selected by the estimation assumptions — the `n* < 1` constraint, the
   `β < α − 1` validity window, and the Pareto mark exponent — rather than
   determined by the data alone; the underlying mechanism is the
   non-identifiability of the kernel scale. The generalization claim is that
   branching-ratio criticality claims for power-law Hawkes cascades require the
   identifiability of the relevant parameters to be checked, not that every
   cascade is supercritical.

## Data

This repository does **not** redistribute any dataset. Two external sources
are required to run the notebook end-to-end:

| Dataset | Source | Upstream license |
|---|---|---|
| `example_book.csv` (Spock cascade, 219 events) | [s-mishra/featuredriven-hawkes](https://github.com/s-mishra/featuredriven-hawkes/blob/master/code/example_book.csv) | GNU GPL v3 |
| ACTIVE (`data.csv`, ~23k cascades) | Rizoiu group cascade-influence repository | CC BY-NC 4.0 |

Download each file from its upstream source and place it where the notebook
expects it (see `data/README.md`). The data files and their upstream licenses
are **not** included here, so the MIT license of this repository applies
cleanly to all distributed content.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then obtain the data (see `data/README.md`) and run:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```

Optionally, cross-check the Python cascade likelihood against the original
authors' R implementation (requires R and the upstream `marked_hawkes.R`):

```bash
Rscript scripts/verify_against_rizoiu.R data/example_book.csv path/to/marked_hawkes.R
```

**Paths.** The notebook reads the Spock CSV from a `DATA_PATH` variable and
the ACTIVE CSV from an `ACTIVE_CSV` variable, and writes figures and JSON to
a `RESULTS_DIR`. The defaults already point to `../data/` and `../results/`,
so dropping the data files into `data/` is sufficient to run.

The free MLE fit on Spock is deterministic given `RNG_SEED = 20260523`. The
synthetic recovery experiments use fixed per-regime seeds so the figures
reproduce. With the default `FORCE_RECOMPUTE = False`, the notebook loads
frozen JSON snapshots for the Phase 0 / Phase 1 / H-B experiments and finishes
in under five minutes; setting `FORCE_RECOMPUTE = True` recomputes everything
and takes roughly thirty minutes (Phase 2 marked recovery is ~10 minutes and
the ACTIVE SLSQP fits ~5 minutes).

## Licensing

All code, notebooks, and derived results in this repository are released under
the **MIT License** (see `LICENSE`). No third-party data is redistributed, so
no upstream license applies to the contents here. If you separately obtain the
datasets listed under [Data](#data), their upstream terms (GPLv3 / CC BY-NC
4.0) govern those files in your own working copy.

## Citations

If you build on this work, please cite the upstream Hawkes-process literature
that the reproduction targets:

* Rizoiu, M.-A., Lee, Y., Mishra, S. & Xie, L. (2017). *A Tutorial on Hawkes
  Processes for Events in Social Media.* arXiv:1708.06401. (The worked Spock
  example audited here.)
* Mishra, S., Rizoiu, M.-A. & Xie, L. (2016). *Feature Driven and Point
  Process Approaches for Popularity Prediction.* CIKM 2016.
  doi:[10.1145/2983323.2983812](https://doi.org/10.1145/2983323.2983812)
  (Source of the closed-form branching-ratio expression and the `n* < 1`
  constraint.)
* Filimonov, V. & Sornette, D. (2015). *Apparent criticality and calibration
  issues in the Hawkes self-excited point process model.* Quantitative Finance,
  15(8), 1293–1314.
* Hardiman, S. J. & Bouchaud, J.-P. (2014). *Branching-ratio approximation for
  the self-exciting Hawkes process.* Physical Review E, 90, 062807.
