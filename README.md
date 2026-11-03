# DiscreteDynamicSurvivalAnalysis.jl

*Fit SIR models to case line lists, in discrete time.*

DiscreteDynamicSurvivalAnalysis.jl fits deterministic, mean-field SIR models, for one population
or several interacting groups, to individual infection and removal times from a closed
population. It gives posterior distributions for the transmission rates and R₀, the parameters of
a geometric, negative binomial or other discrete phase-type infectious period, the initially
infected fraction and, when it is unknown, the size of the population at risk. Records may be
incomplete: a case can lack its infection or its removal time, or still be infectious when
observation ends.

**Discrete dynamic survival analysis (DDSA)** treats the infection and removal times of a line list
as survival data whose distributions come from a deterministic epidemic model. A discrete-time SIR
map, run on the calendar grid of the data, gives the expected fraction `τ` of the population
infected during the observation window and the distribution `f` of infection times. The
likelihood combines three terms:

- a final-size term, `K ~ Binomial(N, τ)`;
- an infection-time term, the histogram of infection steps `~ Multinomial(K, f)`;
- a recovery term, the observed infectious periods under a discrete phase-type distribution.

Right-censored intervals and cases with only a removal time add their own terms. Integer
parameters (the number of infectious stages, an unknown population size) are summed out exactly,
and the continuous parameters are sampled with NUTS in [Turing.jl](https://turinglang.org).

**Scope:**
- single-population models with geometric or negative binomial infectious periods, propagated by
  GLCT (phase-type stage) or Boxcar (infection-age) maps;
- multi-group maps with any force of infection, and likelihood building blocks for custom models;
- posterior summaries and predictive checks, plot recipes, and simulators.

For the assumptions of DDSA and when it applies, see the manuscript.

## Installation

Requires Julia ≥ 1.11. Install the package from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/epirecipes/DiscreteDynamicSurvivalAnalysis.jl")
```

## Quickstart

This fits the synthetic outbreak of the manuscript's data-availability example: 99 susceptibles,
one initial infective, an Erlang-4 infectious period of mean 4 days on a daily grid, and β = 0.5.

```julia
using DiscreteDynamicSurvivalAnalysis, Distributions, Random

sim = simulate_linelist(Xoshiro(1234), dph_erlang_from_mean(4, 4.0, 1.0), 0.5, 0.01, 99, 40)
spec = DDSAModel(infectious_period = NegBinGLCT(1:10),          # r ∈ 1:10 stages, summed out
                 priors = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1)),
                 population = 99)
fit = fit_ddsa(Xoshiro(1), spec, sim.data; ndraws = 500, nchains = 2)
posterior_summary(fit)        # β, ρ, p, r, N, τ, mean_ip, R0 (truth: R0 = 3.16, mean_ip = 6.33)

using Plots
plot(posterior_predictive(Xoshiro(2), fit))
```

The default settings are the manuscript's: 4 chains of 2,000 draws after 1,000 NUTS adaptation
steps. Start Julia with `--threads` to run the chains in parallel.

## Documentation

The documentation is built from `docs/`:
- [Guide](docs/src/guide.md): the data schema, infectious-period models, likelihood terms,
  guards, population options, and the multi-group interface;
- [API reference](docs/src/api.md);
- vignettes that reproduce the manuscript's analyses ([data availability](docs/src/vignettes/data_availability.jl),
  [DDSA vs ODE](docs/src/vignettes/ddsa_vs_ode.jl), [H1N1](docs/src/vignettes/h1n1_wsu.jl),
  [Ebola](docs/src/vignettes/ebola_drc.jl), [Hagelloch measles](docs/src/vignettes/measles_hagelloch.jl)),
  and a [walk-through for your own line list](docs/src/vignettes/external_data.jl);
- [Reproducibility](docs/src/reproducibility.md): how to rerun the manuscript analyses and how the
  results compare with the manuscript.

The manuscript vignettes are shown pre-rendered with the manuscript settings, so the docs build
itself takes about a minute.

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl                                        # build the docs
julia --project=docs --threads=10 docs/render_vignettes.jl [name ...]    # re-render the vignette pages, ≈15 min
```

## Package layout

| path | contents |
|:--|:--|
| `src/PhaseTypeDistributions/` | submodule: continuous and discrete phase-type distributions, NegBin/Erlang builders, discrete hazards |
| `src/IntegerParameters.jl` | submodule: exact summation of integer parameters (`NGrid`, `marginal_r_loglik`, …) |
| `src/data.jl` | `LineList`, `linelist` |
| `src/forward_map.jl` | GLCT and Boxcar forward maps, one or several groups |
| `src/likelihood.jl` | likelihood terms, `NumericalGuards`, `ddsa_loglikelihood` |
| `src/models.jl`, `src/inference.jl` | `DDSAModel`, the Turing model, `fit_ddsa` |
| `src/posterior.jl` | summaries and posterior predictive checks |
| `src/simulate.jl` | `simulate_linelist`, `simulate_chain_binomial` |
| `ext/` | plot recipes (loaded with Plots.jl/RecipesBase) |
| `data/` | the manuscript's line lists ([provenance](data/README.md)) |
| `test/` | tests; `test/reference/` holds fixtures from the notebook implementation of the analyses |
| `scripts/reproduce_manuscript.jl` | reruns the vignettes with the manuscript settings and compares with the published posteriors |
| `docs/render_vignettes.jl` | renders the manuscript vignettes with the manuscript settings for the documentation |

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests compare the package with fixtures from the notebook implementation of the analyses
([archived on Zenodo](https://doi.org/10.5281/zenodo.23924833); `test/reference/generate_reference.jl`). 
They cover the phase-type distributions, forward maps, likelihoods and data preprocessing, with logjoint 
agreement to ≤ 4e-14 relative. The suite also includes a short end-to-end fit and Aqua.jl checks, 
and takes about 4 minutes.

To reproduce the manuscript's posterior tables with the full settings (about 15 minutes on 10
threads):

```sh
julia --project=docs --threads=10 scripts/reproduce_manuscript.jl
```

## Citation

The methods are described in *Discrete dynamic survival analysis of epidemic case line lists*
(manuscript in preparation). The code used for the manuscript's analyses is [archived on Zenodo](https://doi.org/10.5281/zenodo.23924833).
The vignettes in `docs/src/vignettes/` reproduce those analyses with this package.

## Data

`data/` contains the line lists analysed in the manuscript:
- 2009 H1N1 influenza at Washington State University;
- the three waves of the 2018–2020 Ebola epidemic in the DRC;
- the 1861 measles outbreak in Hagelloch.

They are included so that the manuscript analyses can be reproduced. Their sources are listed in
[data/README.md](data/README.md).

## Licence and authors

MIT licence, © 2026 epirecipes. Authors: Sandra Montes-Olivas and Simon Frost.
