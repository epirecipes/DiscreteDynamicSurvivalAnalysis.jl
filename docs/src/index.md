# DiscreteDynamicSurvivalAnalysis.jl

*Fit SIR models to case line lists, in discrete time.*

DiscreteDynamicSurvivalAnalysis.jl fits deterministic, mean-field SIR models, for one population
or several interacting groups, to individual infection and removal times from a closed
population. It gives posterior distributions for the transmission rates and ``R_0``, the
parameters of a geometric, negative binomial or other discrete phase-type infectious period, the
initially infected fraction and, when it is unknown, the size of the population at risk. Records
may be incomplete: a case can lack its infection or its removal time, or still be infectious when
observation ends.

**Discrete dynamic survival analysis (DDSA)** treats the individual infection and removal times of
a line list as survival data whose distributions come from a deterministic epidemic model. A
discrete-time SIR map, run on the calendar grid of the data (e.g. days), gives:

- the expected fraction ``\tau`` of the population infected during the observation window;
- the distribution ``f`` of the infection times over the steps of the window.

The log-likelihood of the line list is then

```math
\ell = \underbrace{\log \mathrm{Binomial}(K;\, N, \tau)}_{\text{final size}}
     + \underbrace{\log \mathrm{Multinomial}(\mathbf{n};\, K, f)}_{\text{infection times}}
     + \underbrace{\textstyle\sum_i \log P(W = \delta_i)}_{\text{recovery intervals}},
```

with ``K`` cases, infection-step counts ``\mathbf{n}``, recovery intervals ``\delta_i``, and an infectious period ``W``
with a discrete phase-type (DPH) distribution: geometric, negative binomial (Erlang-like), or any
DPH. Right-censored intervals and cases with only a removal time add their own terms. Integer
parameters (the number of infectious stages ``r``, an unknown population size ``N``) are summed out
exactly. The continuous parameters (transmission rate ``\beta`` or ``R_0``, the infectious-period
parameter, the initially infected fraction ``\rho``) are sampled with NUTS through
[Turing.jl](https://turinglang.org).

DDSA is the discrete-time counterpart of the continuous-time dynamic survival analysis (DSA) of
KhudaBukhsh et al., Di Lauro et al. and Vossler et al. This package implements the methods of the manuscript
*Discrete dynamic survival analysis of epidemic case line lists* (in preparation), and its
vignettes reproduce the manuscript's analyses.

## What the package covers

- **Data:** binning of raw event times into a [`LineList`](@ref) ([`linelist`](@ref)), with
  missing times, right-censoring and removal-only cases.
- **Forward maps:** the GLCT (phase-type stages) and Boxcar (infection-age) SIR maps, for one or
  several groups with any force of infection ([`forward_map`](@ref)).
- **Likelihood:** the DDSA likelihood and its terms ([`ddsa_loglikelihood`](@ref)).
- **Models:** single-population models with geometric or negative binomial infectious periods
  ([`DDSAModel`](@ref)), and building blocks for custom Turing models (see the
  [Hagelloch vignette](vignettes/measles_hagelloch.md), which also fits a Coxian infectious period
  on the GLCT map).
- **Inference:** [`fit_ddsa`](@ref) samples the posterior, then draws ``r`` and ``N`` exactly.
  [`posterior_summary`](@ref), [`posterior_predictive`](@ref) and
  [`recovery_predictive`](@ref) report the results; plot recipes become available when Plots.jl
  is loaded.
- **Simulation:** synthetic line lists from the DDSA working model and from a stochastic chain
  binomial ([`simulate_linelist`](@ref), [`simulate_chain_binomial`](@ref)).
- **Submodules:** `PhaseTypeDistributions` (continuous and discrete phase-type distributions)
  and `IntegerParameters` (exact marginalisation of integer parameters). Their exports are
  re-exported.
- **Example data:** the manuscript's data sets ([`example_data_path`](@ref)).

## Installation

The package requires Julia ≥ 1.11. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/epirecipes/DiscreteDynamicSurvivalAnalysis.jl")
```

## Quickstart

Simulate the synthetic outbreak of the manuscript's data-availability example: 99 susceptibles
and one initial infective, with an Erlang-4 infectious period of mean 4 days discretised to
daily steps, ``\beta = 0.5`` per day, over 40 days.

```@example quickstart
using DiscreteDynamicSurvivalAnalysis, Distributions, Random

sim = simulate_linelist(Xoshiro(1234), dph_erlang_from_mean(4, 4.0, 1.0), 0.5, 0.01, 99, 40)
data = sim.data
```

Fit a model with a negative binomial infectious period, with ``r \in \{1, \dots, 10\}`` stages summed out, and a
known population of 99:

```@example quickstart
spec = DDSAModel(infectious_period = NegBinGLCT(1:10),
                 priors = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1)),
                 population = 99)
fit = fit_ddsa(Xoshiro(1), spec, data; ndraws = 500, nchains = 2)
```

```@example quickstart
using DataFrames
DataFrame(posterior_summary(fit))
```

The true values are ``R_0 = 3.16`` and a mean infectious period of 6.33 days. The documentation
build uses 2 chains of 500 draws, while the manuscript uses 4 chains of 2,000 draws.

```@example quickstart
using Plots
plot(posterior_predictive(Xoshiro(2), fit); title = "posterior predictive infection curve")
```

## Where to go next

- The [Guide](guide.md) describes the data schema, the infectious-period models, the likelihood
  terms, the guards, the population options and the multi-group interface.
- The vignettes reproduce the manuscript's analyses:
  - [data availability](vignettes/data_availability.md);
  - [DDSA vs ODE count models](vignettes/ddsa_vs_ode.md);
  - [H1N1](vignettes/h1n1_wsu.md);
  - [Ebola](vignettes/ebola_drc.md);
  - [Hagelloch measles](vignettes/measles_hagelloch.md).

  [Using your own line list](vignettes/external_data.md) walks through a new data set.
- The [API reference](api.md) documents every function.
- [Reproducibility](reproducibility.md) explains how to rerun the manuscript analyses and how the
  results compare with the manuscript.
