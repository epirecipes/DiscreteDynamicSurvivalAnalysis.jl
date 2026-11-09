# API reference

```@meta
CurrentModule = DiscreteDynamicSurvivalAnalysis
```

```@docs
DiscreteDynamicSurvivalAnalysis
```

## Workflow

### Data

```@docs
LineList
LineList(::AbstractVector{<:Integer})
linelist
example_data_path
```

### Model specification

```@docs
DDSAModel
InfectiousPeriodModel
GeometricIP
NegBinGLCT
NegBinBoxcar
NumericalGuards
turing_model
```

### Inference

```@docs
fit_ddsa
DDSAFit
default_sampler
data_based_inits
```

### Results

```@docs
posterior_draws
posterior_summary
posterior_predictive
PosteriorPredictive
recovery_predictive
RecoveryPredictive
predictive_quantiles
```

## Building blocks

### Forward maps

```@docs
InfectionStructure
GLCT
Boxcar
forward_map
forward_map!
MapSolution
attack_fraction
infection_pmf
susceptible_fraction
```

### Likelihood

```@docs
ddsa_loglikelihood
final_size_loglik
infection_time_loglik
recovery_loglik
removal_only_loglik
```

## Simulation

```@docs
simulate_linelist
simulate_chain_binomial
```

## PhaseTypeDistributions

```@docs
PhaseTypeDistributions
```

### Discrete phase-type distributions

```@docs
DPH
dph_geometric
dph_negative_binomial
dph_negative_binomial_min_support
dph_prepend_deterministic_delay
dph_coxian
dph_erlang
dph_erlang_from_mean
dph_erlang_from_rates
dph_exponential
dph_from_rates
dph_negative_binomial_from_rates
dph_negative_binomial_from_mean
discrete_hazard
PhaseTypeDistributions.pmf
mean(::DPH)
var(::DPH)
cdf(::DPH, ::Integer)
ccdf(::DPH, ::Integer)
logccdf(::DPH, ::Integer)
logpdf(::DPH, ::Integer)
Base.rand(::Random.AbstractRNG, ::DPH)
```

### Smooth negative binomial

```@docs
SmoothDPH
dph_negative_binomial_smooth
logpdf(::SmoothDPH, ::Integer)
Base.rand(::Random.AbstractRNG, ::SmoothDPH)
```

### Continuous phase-type distributions

```@docs
PhaseType
ph_exponential
ph_erlang
ph_erlang_by_mean
```

### Helper functions

Not exported: call them with the module name, e.g. `PhaseTypeDistributions.rate_to_proportion(0.2)`.

```@docs
PhaseTypeDistributions.rate_to_proportion
PhaseTypeDistributions.negbin_total_logpdf
```

## IntegerParameters

```@docs
IntegerParameters
NGrid
marginal_N_loglik
N_posterior_weights
sample_N
marginal_r_loglik
posterior_weights
sample_index
```
