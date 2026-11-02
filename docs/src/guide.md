# Guide

This page describes the pieces of a DDSA analysis and the conventions the package uses. The
[vignettes](vignettes/external_data.md) show them in complete analyses.

```@setup guide
using DiscreteDynamicSurvivalAnalysis, Distributions, Turing, Random
```

## The time grid and the data

DDSA works on a grid of `nsteps` steps of length ``\Delta t`` (e.g. days), measured from the start of
observation. Step ``k`` covers the interval ``[(k-1)\Delta t, k\Delta t)``: with ``\Delta t = 1`` day, step 1 is day 0.
The forward map runs over the same grid, so its infection-time distribution ``f`` has one entry per
step.

A [`LineList`](@ref) holds one population's (or one group's) data on that grid:

| Field | Meaning | Likelihood term |
|:--|:--|:--|
| `infection_counts` | Number of cases infected (onset) in each step | Infection times |
| `final_size` | Number of cases ``K``, including those with an unknown infection time | Final size |
| `recovery_intervals` | Infection-to-removal intervals in steps (``\geq 1``) | Recovery |
| `censored` | `true` where an interval is right-censored (still infectious at the end) | Recovery |
| `removal_only_steps` | Removal step of cases with an unknown infection time | Removal only |

[`linelist`](@ref) builds one from raw event times (`missing` = unknown):

- Infection step ``\lfloor t / \Delta t \rfloor + 1``;
- Recovery interval ``\max\{1, \operatorname{round}((t_\text{rem} - t_\text{inf}) / \Delta t)\}``;
- Removals at or after `censor_at` are right-censored at `censor_at`;
- A case with no infection time and a known removal time becomes a removal-only case;
- A case with no removal time contributes only its infection time.

The grid ends at `nsteps` if given, otherwise at the step of the last infection. Invalid input
raises an `ArgumentError` or `DimensionMismatch` that names the problem: negative times, a
removal before the infection, both times missing, or an infection beyond the grid.

```@example guide
data = linelist([0.2, 1.5, 2.1, missing, 4.8], [4.0, missing, 7.9, 6.3, 9.5]; nsteps = 8, censor_at = 8.0)
```

```@example guide
(data.infection_counts, data.recovery_intervals, data.censored, data.removal_only_steps)
```

Calendar dates (`Date` or `DateTime`) need `origin`, the date at which step 1 starts. Each date
becomes its number of days since `origin`, so ``\Delta t`` is in days, and `censor_at` can be a date.
A date before `origin` is an error.

```@example guide
using Dates
day0 = Date(2024, 3, 4)
onset = [day0, day0 + Day(2), missing]
removal = [day0 + Day(5), day0 + Day(9), day0 + Day(6)]
linelist(onset, removal; origin = day0, nsteps = 7, censor_at = day0 + Day(7))
```

Already-binned data go straight into the keyword constructor,
`LineList(counts; final_size, recovery_intervals, censored, removal_only_steps, Δt)`.

## The forward map

The deterministic SIR map tracks the susceptible fraction ``S`` and the infected population. Each
step, the new infections are

```math
\lambda_t = \left(1 - e^{-\beta I_t \Delta t}\right) S_t, \qquad S_{t+1} = S_t - \lambda_t,
```

with ``I`` the infected fraction and ``S_0 = 1 - \rho``. The infected progress to removal in one of two
ways (an [`InfectionStructure`](@ref)):

- [`GLCT`](@ref)`(dph)`, the generalised linear chain trick. The infected occupy the transient
  states of a discrete phase-type distribution: ``x \leftarrow T^\top x + \alpha \lambda``, with ``x`` the
  infected fraction in each state, ``T`` the sub-transition matrix (`dph.S`) and ``\alpha`` the
  initial distribution (`dph.α`). A geometric infectious period (one state) gives the classical discrete SIR map.
- [`Boxcar`](@ref)`(hazard)`, infection-age classes. A fraction `hazard[k]` of the infected of
  age ``k`` is removed each step, and the rest age by one step.
  [`discrete_hazard`](@ref)`(dph, K; truncation)` builds the hazard from the DPH of the infectious
  period ``W``. The map tracks only ages ``1, \dots, K_\text{age}`` (``K_\text{age}`` = `K`), so the
  probability that ``W`` exceeds ``K_\text{age}`` has to go somewhere, and `truncation` decides where:
  - `:absorb` (default) keeps the removal probability of ``W`` at every age below
    ``K_\text{age}`` and removes everyone still infected at age ``K_\text{age}``: the map's
    infectious period is ``\min(W, K_\text{age})``. The map is exact for every age it tracks, and
    exact overall when ``K_\text{age}`` is at least the number of steps, because no one reaches
    age ``K_\text{age}`` within the observation window. 
  - `:condition` spreads that probability over all ages in proportion: the map's infectious
    period is ``W`` conditioned on ``W \le K_\text{age}``, so the removal probability rises
    slightly at every age. 

  Both give the same map when ``P(W > K_\text{age})`` is negligible, which is the usual aim: choose
  ``K_\text{age}`` beyond the longest infectious period the prior allows (e.g. its 99.9th
  percentile). In a [`NegBinBoxcar`](@ref) model the recovery term always uses the untruncated ``W``.

[`forward_map`](@ref) returns a [`MapSolution`](@ref) holding:
- ``\tau``, the attack fraction (the new infections over the window, ``S_0 - S_T``);
- ``f``, the new infections per step normalised to sum to one;
- the susceptible history.

Both ``\tau`` and ``f`` are computed from the recorded new infections, so they keep full precision when
infections are tiny. The map is generic in its element type, so ForwardDiff differentiates
through it.

```@example guide
sol = forward_map(GLCT(dph_negative_binomial(4, 0.6)), 0.5, 0.01, 40)
(attack_fraction(sol), sum(infection_pmf(sol)))
```

## Infectious-period models and parameterisations

A [`DDSAModel`](@ref) takes one of three [`InfectiousPeriodModel`](@ref)s:

| Model | Distribution of ``W`` (steps) | Forward map | Integer parameter |
|:--|:--|:--|:--|
| [`GeometricIP`](@ref)`(; parameter)` | ``\mathrm{Geometric}(p)`` on ``\{1, 2, \dots\}`` | GLCT, one state | none |
| [`NegBinGLCT`](@ref)`(r_support; parameter)` | negative binomial, ``r`` stages, support ``\{r, r+1, \dots\}`` | GLCT | ``r`` in `r_support` |
| [`NegBinBoxcar`](@ref)`(r_support, K_age; parameter, truncation)` | as above | Boxcar with `K_age` classes | ``r`` in `r_support` |

The sampled infectious-period parameter is chosen with `parameter`:

- `:p`: the per-step progression probability, ``p \in (0, 1)``;
- `:γ`: a removal rate ``\gamma`` per unit time, with ``p = 1 - e^{-\gamma \Delta t}`` (geometric only);
- `:mean_ip`: the mean infectious period ``\mu`` in time units, with ``p = r \Delta t / \mu``.

Transmission is parameterised by ``\beta``, a rate per unit time, or with `transmission = :R0` by
``R_0``, where ``\beta = R_0 / \mu``. The latter needs `parameter = :mean_ip`. The derived
reproduction number reported by [`posterior_draws`](@ref) is ``R_0 = \beta \mu = \beta r \Delta t / p``.

## Likelihood terms

The `terms` keyword of [`DDSAModel`](@ref) selects the terms; each is also a function you can call
directly.

| Term | Function | Data | Needs the map |
|:--|:--|:--|:--|
| `:final_size` | [`final_size_loglik`](@ref) | ``K`` | yes (``\tau``) |
| `:infection_times` | [`infection_time_loglik`](@ref) | `infection_counts` | yes (``f``) |
| `:recovery` | [`recovery_loglik`](@ref) | `recovery_intervals`, `censored` | no |
| `:removal_only` | [`removal_only_loglik`](@ref) | `removal_only_steps` | yes (``f``) |

[`ddsa_loglikelihood`](@ref) adds the selected terms for one `MapSolution` and `LineList`. A model
samples only the parameters that its terms need: a recovery-only model (`terms = (:recovery,)`)
has just the infectious-period parameter, and needs no ``\rho``, transmission parameter or population.

## Numerical guards

[`NumericalGuards`](@ref)`(; τ_bounds, f_floor)` clamps the attack fraction and floors ``f`` before
the likelihood is evaluated. The package default is no guarding, `τ_bounds = (0, 1)` and
`f_floor = 0`, which is the exact likelihood. The guards only matter at parameter values where the
map produces attack fractions or infection probabilities at those extremes, typically far out in
the tails of the posterior.

## Population size

The final-size term ``K \sim \mathrm{Binomial}(N, \tau)`` needs the number at risk ``N`` (`population`):

- an integer: ``N`` is known;
- a range `N_min:N_max`: ``N`` is unknown with a discrete uniform prior;
- an [`NGrid`](@ref)`(K, N_min, N_max; logprior)`: ``N`` is unknown with any prior.

An unknown ``N`` is summed out of the likelihood exactly ([`marginal_N_loglik`](@ref)). After
sampling, [`fit_ddsa`](@ref) draws ``N`` from its exact conditional posterior for each MCMC draw
([`sample_N`](@ref)) and adds it to the chain as column `:N`.

## Integer parameters

The number of stages ``r`` changes the forward map, so it cannot enter through a closed-form sum.
The model evaluates the log-likelihood ``\ell_r`` for every ``r`` in `r_support` and adds
``\log \sum_r \pi(r)\, e^{\ell_r}`` ([`marginal_r_loglik`](@ref)), where ``\pi(r)`` is the prior of
``r`` (uniform on `r_support`). Values of ``r`` above the shortest observed recovery interval have
zero likelihood and cost nothing. The model returns the per-``r`` log-likelihoods, and
[`fit_ddsa`](@ref) draws ``r`` for each MCMC draw ([`sample_index`](@ref)),
adding column `:r`. Everything else is sampled using No-U-Turn Sampler (NUTS).

The same tools handle joint grids. In the Hagelloch example, ``r`` and the size ``N_2`` of the group with
the index case both change the map. The pairs ``(r, N_2)`` are summed jointly, then drawn with `sample_index` over the cells.

## Fitting and results

[`fit_ddsa`](@ref)`(rng, spec, data; ndraws, nchains, sampler, parallel, initial_params)` runs
NUTS through Turing. The defaults are the manuscript's settings: 4 chains of 2,000 draws after
1,000 adaptation steps ([`default_sampler`](@ref)), threaded chains, and jittered data-based
starting values ([`data_based_inits`](@ref)). A seeded `rng` makes the run reproducible.
[`posterior_draws`](@ref) and [`posterior_summary`](@ref) give the parameters, ``r``, ``N``, ``\tau`` and
derived quantities. [`posterior_predictive`](@ref) and [`recovery_predictive`](@ref) give
posterior predictive checks, and `plot(pp)` draws them when Plots.jl is loaded.

The infection-curve check plots the observed infection counts per step as bars. They count only
the ``K_\text{known}`` cases whose infection time is recorded, out of ``K`` cases in total. The
curve is the map's expected number of new infections per step, ``N \tau f``, at posterior draws,
rescaled according to `scale`:

| `scale` | Curve | What it checks |
|:--|:--|:--|
| `:expected_known` (default) | ``(K_\text{known} / K)\, N \tau f``, the expected infections among cases with a recorded infection time | The size and the timing of the epidemic, assuming the same fraction of infection times is missing at every step |
| `:expected` | ``N \tau f``, all expected new infections | The same as the default when every infection time is recorded; otherwise it lies above the bars by the factor ``K / K_\text{known}`` |
| `:observed_total` | ``K_\text{known} f``, the infection-time distribution scaled to the total of the bars | The timing only: the curve has the same total as the bars whatever the predicted epidemic size |

For a given draw the three curves differ only by a constant factor.

To use other Turing settings or samplers, build the model with [`turing_model`](@ref) and call
`sample` yourself. The model returns `(lps, τ)` per value of ``r``.

## Several groups: the force-of-infection interface

For ``G`` groups, `forward_map(structure, foi, ρ::Vector, nsteps)` runs one SIR map per group,
coupled through the force of infection. `foi` is either a ``G \times G`` matrix ``B``, giving rates
``B I``, or any callable `foi(I)` returning the ``G`` rates from the vector ``I`` of infected
fractions. A closure over the parameters is the natural way to write a user model, and it must
accept ForwardDiff `Dual`s. Each group then gets its own `LineList` and its own call of
[`ddsa_loglikelihood`](@ref) (`group = g`). Shared terms, such as a recovery term pooled across
groups, are added once.

A minimal two-group model, with transmission within groups and a weaker cross-group term:

```@example guide
@model function two_groups(data1, data2, N1, N2, delta)
    β_within ~ Exponential(0.5)
    β_cross ~ Exponential(0.1)
    p ~ Beta(1, 1)
    foi(I) = [β_within * I[1] + β_cross * I[2], β_cross * I[1] + β_within * I[2]]
    dph = dph_geometric(p)
    sol = forward_map(GLCT(dph), foi, [0.01, 0.0], data1.nsteps)
    ll = ddsa_loglikelihood(sol, data1, dph; population = N1, group = 1,
                            terms = (:final_size, :infection_times)).loglik
    ll += ddsa_loglikelihood(sol, data2, dph; population = N2, group = 2,
                             terms = (:final_size, :infection_times)).loglik
    Turing.@addlogprob! ll + recovery_loglik(dph, delta)
end

d1 = LineList([1, 2, 4, 6, 5, 3, 1, 0, 0, 0])
d2 = LineList([0, 0, 1, 1, 3, 4, 3, 2, 1, 0])
m = two_groups(d1, d2, 40, 30, [3, 4, 2, 5, 3, 4])
logjoint(m, (β_within = 0.8, β_cross = 0.1, p = 0.3))
```

Sample it with `sample(m, NUTS(), MCMCThreads(), 1000, 4)`. 