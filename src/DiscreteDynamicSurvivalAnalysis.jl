"""
    DiscreteDynamicSurvivalAnalysis

Discrete Dynamic Survival Analysis (DDSA): Bayesian inference for epidemic models from case line
lists. A deterministic discrete-time SIR map on the calendar grid of the data gives the expected
attack fraction `τ` and infection-time distribution `f`; individual infection and removal times
are then treated as survival data, with log-likelihood

    ℓ = ℓ_constr (final size, K ~ Binomial(N, τ))
      + ℓ_I      (infection times, Multinomial(K, f))
      + ℓ_R      (recovery intervals, discrete phase-type infectious period)

plus right-censoring and removal-only terms where the data need them. Integer-valued parameters
(the number of infectious stages `r`, an unknown population size `N`) are summed out exactly, so
the continuous parameters are sampled with NUTS (Turing.jl).

Main entry points: [`linelist`](@ref)/[`LineList`](@ref) (data), [`DDSAModel`](@ref) (model),
[`fit_ddsa`](@ref) (inference), [`posterior_summary`](@ref) and [`posterior_predictive`](@ref)
(results). Building blocks for custom models: [`forward_map`](@ref), [`ddsa_loglikelihood`](@ref)
and the term functions, and the submodules `PhaseTypeDistributions` and `IntegerParameters`, whose
exports are re-exported.
"""
module DiscreteDynamicSurvivalAnalysis

using Dates: Dates, Date, DateTime
using LinearAlgebra
using Logging: Logging, AbstractLogger, LogLevel, current_logger, with_logger
using Random
using Statistics: mean, median, quantile, std
using Distributions: Distributions, Binomial, Multinomial, UnivariateDistribution, logpdf, pdf
using StatsFuns: binomlogpdf
using MCMCChains: Chains, setrange
using MCMCDiagnosticTools: ess_rhat
using Turing: Turing, @model, NUTS, sample, returned, InitFromParams, InitFromPrior, MCMCThreads

include("PhaseTypeDistributions/PhaseTypeDistributions.jl")
include("IntegerParameters.jl")
using .PhaseTypeDistributions
using .PhaseTypeDistributions: rate_to_proportion
using .IntegerParameters

include("utils.jl")
include("data.jl")
include("forward_map.jl")
include("likelihood.jl")
include("models.jl")
include("inference.jl")
include("posterior.jl")
include("simulate.jl")
include("datasets.jl")

# submodules (re-exported)
export PhaseTypeDistributions, IntegerParameters
export PhaseType, ph_exponential, ph_erlang, ph_erlang_by_mean
export DPH, dph_geometric, dph_negative_binomial, dph_erlang
export dph_from_rates, dph_exponential, dph_erlang_from_rates, dph_erlang_from_mean
export dph_negative_binomial_from_rates, dph_negative_binomial_from_mean
export dph_prepend_deterministic_delay, dph_negative_binomial_min_support
export discrete_hazard, dph_negative_binomial_smooth, SmoothDPH
export mean, var, pmf, cdf, pdf, logpdf, ccdf, logccdf
export NGrid, marginal_N_loglik, N_posterior_weights, sample_N,
       marginal_r_loglik, posterior_weights, sample_index

# data
export LineList, linelist, example_data_path

# forward map
export InfectionStructure, GLCT, Boxcar, MapSolution, forward_map, forward_map!,
       attack_fraction, infection_pmf, susceptible_fraction

# likelihood
export NumericalGuards, final_size_loglik, infection_time_loglik, recovery_loglik,
       removal_only_loglik, ddsa_loglikelihood

# model specification and inference
export InfectiousPeriodModel, GeometricIP, NegBinGLCT, NegBinBoxcar, DDSAModel, turing_model,
       DDSAFit, fit_ddsa, default_sampler, data_based_inits

# results
export posterior_draws, posterior_summary, PosteriorPredictive, posterior_predictive,
       RecoveryPredictive, recovery_predictive, predictive_quantiles

# simulation
export simulate_linelist, simulate_chain_binomial

end # module DiscreteDynamicSurvivalAnalysis
