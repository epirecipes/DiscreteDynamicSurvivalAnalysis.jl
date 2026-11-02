"""
    PhaseTypeDistributions

Continuous and discrete phase-type distributions.

- **Continuous phase-type** (`PhaseType`): time to absorption of a continuous-time Markov chain
  with initial distribution `α` and transient generator `A`; density `f(t) = α' exp(A t) q`.
- **Discrete phase-type** (`DPH`): number of steps to absorption of a discrete-time Markov
  chain with initial distribution `α` and sub-stochastic transition matrix `S`;
  `P(W = n) = α' S^(n-1) s` for `n ≥ 1`, with absorption vector `s = 1 - S 1`.
- **Smooth negative binomial** (`SmoothDPH`): mixture of the `⌊r⌋`- and `⌊r⌋+1`-stage
  negative binomial DPHs, so that the number of stages `r` can be real-valued.

All types are `Distributions.Distribution`s: `pdf`, `logpdf`, `cdf`, `mean`, `var`, `rand`, …
are methods of the `Distributions`/`Statistics`/`Random` functions (re-exported here), so they
work with or without this module's prefix and compose with Turing.jl. Element types are generic,
so ForwardDiff `Dual` numbers pass through the constructors and the discrete-PH methods.

# Loading

```julia
using DiscreteDynamicSurvivalAnalysis                          # re-exports everything below
using DiscreteDynamicSurvivalAnalysis.PhaseTypeDistributions   # this submodule only
```

# Example

```julia
d = dph_negative_binomial(4, 0.6)   # 4 stages, per-step progression probability 0.6
pdf(d, 7), mean(d)                  # P(W = 7), r/p = 6.67 steps
erl = ph_erlang(3, 1.5)             # continuous Erlang, 3 phases, rate 1.5
cdf(erl, 2.0)
```

# References

- Hurtado, P. J. & Richards, C. (2021). Building mean field ODE models using the generalized
  linear chain trick & Markov chain theory. Journal of Biological Dynamics, 15(sup1), S248–S272.
  https://doi.org/10.1080/17513758.2021.1912418
"""
module PhaseTypeDistributions

using LinearAlgebra
using Random
using ForwardDiff
using SpecialFunctions: logabsbinomial, loggamma
import Distributions
using Distributions: ContinuousUnivariateDistribution, DiscreteUnivariateDistribution
import Distributions: pdf, logpdf, cdf, logcdf, ccdf, logccdf, support, insupport
import Statistics: mean, var

# Numerical constants
const DEFAULT_TOLERANCE = 1e-10   # Erlang-matrix detection and DPH α-sum check
const MAX_PHASES = 50             # upper bound on the number of stages of the smooth NegBin

include("continuous.jl")
include("discrete.jl")
include("builders.jl")
include("smooth.jl")

# Continuous phase-type distributions
export PhaseType, ph_exponential, ph_erlang, ph_erlang_by_mean

# Discrete phase-type distributions
export DPH, dph_geometric, dph_negative_binomial, dph_erlang
export dph_from_rates, dph_exponential, dph_erlang_from_rates, dph_erlang_from_mean
export dph_negative_binomial_from_rates, dph_negative_binomial_from_mean
export dph_prepend_deterministic_delay, dph_negative_binomial_min_support
export discrete_hazard

# Smooth interpolations
export dph_negative_binomial_smooth, SmoothDPH

# Common methods (pmf is this module's own; the others are the Distributions/Statistics functions)
export mean, var, pmf, cdf, pdf, logpdf, ccdf, logccdf

# Documented but not exported
public negbin_total_logpdf, rate_to_proportion, DEFAULT_TOLERANCE, MAX_PHASES

end # module PhaseTypeDistributions
