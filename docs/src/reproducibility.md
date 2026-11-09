# Reproducibility

The vignettes in `docs/src/vignettes/` reproduce the analyses of the manuscript *Discrete dynamic
survival analysis of epidemic case line lists*. This page explains how to rerun them and how the
results compare with the manuscript's published results.

## Rerunning the manuscript analyses

| Manuscript Analysis | Vignette |
|:--|:--|
| What each likelihood term contributes (synthetic outbreak) | [data availability](vignettes/data_availability.md) |
| DDSA versus ODE count models (synthetic outbreak)          | [DDSA vs ODE](vignettes/ddsa_vs_ode.md) |
| 2009 H1N1 influenza, Washington State University           | [H1N1](vignettes/h1n1_wsu.md) |
| 2018–2020 Ebola virus disease, DRC                         | [Ebola](vignettes/ebola_drc.md) |
| 1861 measles, Hagelloch: three-group model                 | [Hagelloch](vignettes/measles_hagelloch.md) |

With the environment variable `DDSA_FULL=true`, the vignettes use the manuscript's settings:
- 4 chains of 2,000 draws after 1,000 NUTS adaptation steps, with 4,000 draws for the scenario B
  NegBin and Boxcar fits of the data-availability example;
- in the DDSA-vs-ODE comparison, 2,000 adaptation steps for the ODE count models, with target
  acceptance 0.95 (0.8 for the ``\gamma``-fixed model).

The Hagelloch vignette differs in one setting: all its fits use target acceptance 0.95 instead of
the manuscript's 0.8, because its Coxian fit, which goes beyond the manuscript, needs the smaller
steps to avoid divergent transitions.

The pages in this documentation were rendered with these settings and are shown without being
rerun. Without `DDSA_FULL=true` the scripts use fewer chains and draws (as few as 2 chains of
100 draws) with shorter adaptation, and the Ebola vignette fits only wave 2. That mode only checks that
the code works, and its estimates do not replicate the manuscript's results.

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs --threads=10 docs/render_vignettes.jl              # re-render the pages, ≈15 min
julia --project=docs --threads=10 scripts/reproduce_manuscript.jl       # rerun and compare, ≈15 min
julia --project=docs --threads=10 scripts/reproduce_manuscript.jl h1n1_wsu ebola_drc
DDSA_FULL=true julia --project=docs --threads=10 docs/src/vignettes/h1n1_wsu.jl   # one vignette
```

Each vignette sets the manuscript's priors, ``r`` support, ``K_\text{age}``, NUTS settings (apart from
the Hagelloch target acceptance) and numerical guards.

The two Boxcar analyses truncate the infectious period ``W`` at ``K_\text{age}`` age classes in different ways:
- **Data availability:** the hazard of ``W`` conditioned on ``W \le K_\text{age}``, which is
  `truncation = :condition` in [`discrete_hazard`](@ref);
- **Hagelloch:** the exact hazard of ``\min(W, K_\text{age})``, which is `truncation = :absorb`.

**Other notes:**
- **Data:** the synthetic data set of the data-availability and DDSA-vs-ODE examples is
  regenerated: `simulate_linelist(Xoshiro(1234), dph_erlang_from_mean(4, 4.0, 1.0), 0.5,
  0.01, 99, 40)` gives the same 91 cases, infection histogram and intervals.
- **Predictive checks:** [`posterior_predictive`](@ref) and [`recovery_predictive`](@ref) use draws that are evenly spaced across all combined chains.

## Comparison with the manuscript

`scripts/reproduce_manuscript.jl` runs each vignette with the full settings. It compares the
posterior medians and 95% limits with the manuscript's posterior tables
(`test/reference/manuscript_posteriors.toml`). A quantity passes if one of two rules holds:
- **Posterior SD:** measured in the posterior sd of the new run, its median is within 0.1 sd and
  its limits within 0.2 sd. 
- **Monte Carlo error:** a median or limit outside that tolerance still passes if it is within 2
  combined Monte Carlo standard errors. The combined error is ``\sqrt{2} \times \mathrm{MCSE}`` of the new run's
  estimate (`MCMCDiagnosticTools.mcse`). The script reports these quantities as "pass (MCSE)".

Agreement is within Monte Carlo error.

Results (Julia 1.11.6, 10 threads, Apple M-series laptop):

| Analysis | Quantities passing | Of which only by the Monte Carlo rule | Wall time (whole vignette) |
|:--|:--|:--|:--|
| Data availability (8 fits) | 25 / 25 | 3 | 67 s |
| DDSA vs ODE (5 fits + Poisson identity) | 18 / 18 | 0 | 140 s |
| H1N1 | 8 / 8 | 0 | 20 s |
| Ebola waves 1–3 | 24 / 24 | 0 | 52 s |
| Hagelloch, GLCT and Boxcar | 19 / 19 | 3 | 574 s |
| **Total** | **94 / 94** | **6** | 14 min |

The wall times include compilation and plotting, and for Hagelloch the Coxian fit, which has no
counterpart in the manuscript. All medians are within the sd tolerance, and so
are the mode and distribution of ``r``, the integer group sizes and the Poisson identity (to 4
decimals).

## Regression tests

The regression fixtures come from the Quarto-notebook implementation of the analyses, 
archived as version 1.0.0 of [epirecipes/DDSA-manuscript](https://github.com/epirecipes/DDSA-manuscript) 
([doi:10.5281/zenodo.23924833](https://doi.org/10.5281/zenodo.23924833)). 
They were generated from commit [8ea70c9](https://github.com/epirecipes/DDSA-manuscript/commit/8ea70c9865a13fdaa4601d6cf40995656e987330). 
`test/reference/generate_reference.jl` ran that implementation and saved its outputs in `test/reference/*.toml`. The tests compare the
package against them:

| Comparison | Largest relative difference |
|:--|:--|
| Data preprocessing (`linelist` vs the notebook preprocessing) | exact |
| Phase-type pmf/cdf/logpdf/mean/var, hazards | round-off (≤ 1e-14) |
| Forward maps: ``\tau``, ``f``, ``S`` (GLCT, Boxcar, three-group) | ≤ 1e-12 (``\tau``, ``S``); ``f`` ≤ 1e-10 where infections are tiny |
| Data-availability models (8), logjoint/loglikelihood | ≤ 1.9e-15 |
| DDSA vs ODE, DDSA model | 2.7e-16 |
| H1N1 | 2.2e-14 |
| Ebola waves 1–3 | ≤ 4.1e-14 |
| Hagelloch GLCT / Boxcar three-group models | ≤ 1.6e-15 |
| Synthetic data (`simulate_linelist`, seed 1234) | exact |

The package builds ``\tau`` and ``f`` from the new infections recorded at each step. Where new infections
are tiny (``R_0 < 1``), this is more precise than differences of the susceptible fraction, which
explains the small differences in `f`. A few phase-type functions also differ from the notebook
implementation at edge cases. For example `negbin_total_logpdf` at ``p = 1``, and `pmf` and `cdf` for ``n < 1``.

## Environments

- **Package:** `Project.toml` declares compatibility bounds only. The tested versions differ
  slightly from those used for the manuscript: Distributions 0.25.132 (manuscript 0.25.131),
  StatsFuns 2.2.1 (1.5.3), MCMCChains 7.8.1 (7.7.0). Turing is 0.49.0 in both.
- **Documentation and vignettes:** `docs/Project.toml`. Plots, StatsPlots and OrdinaryDiffEq
  resolve to the versions used for the manuscript.
- **Julia:** ≥ 1.11. Tested with 1.11.6, the version used for the manuscript.
