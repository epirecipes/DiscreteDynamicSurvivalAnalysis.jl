# Model specification for the standard single-population DDSA workflow, and its Turing model.

"""
    InfectiousPeriodModel

Infectious-period (IP) model of a [`DDSAModel`](@ref): which distribution the recovery intervals
follow and how infected individuals progress in the forward map. Subtypes:
[`GeometricIP`](@ref), [`NegBinGLCT`](@ref), [`NegBinBoxcar`](@ref).

Each has a `parameter` field naming the sampled IP parameter:
- `:p`: per-step progression (removal, for one stage) probability `p ∈ (0, 1)`;
- `:γ`: removal rate per unit time, `p = 1 − exp(−γ Δt)` (geometric only);
- `:mean_ip`: mean infectious period `μ` in time units, `p = r Δt / μ`.
"""
abstract type InfectiousPeriodModel end

"""
    GeometricIP(; parameter = :p)

Geometric infectious period (one stage, `r ≡ 1`): removal with probability `p` per step, so
`P(W = n) = p (1 − p)^(n−1)` and the mean is `1/p` steps. The map is the classical discrete-time
SIR map. `parameter` is `:p`, `:γ` (rate per unit time) or `:mean_ip`.
"""
struct GeometricIP <: InfectiousPeriodModel
    parameter::Symbol
    function GeometricIP(parameter::Symbol)
        parameter in (:p, :γ, :mean_ip) ||
            throw(ArgumentError("GeometricIP parameter must be :p, :γ or :mean_ip; got :$parameter"))
        return new(parameter)
    end
end
GeometricIP(; parameter::Symbol = :p) = GeometricIP(parameter)

"""
    NegBinGLCT(r_support; parameter = :p)

Negative binomial infectious period with `r ∈ r_support` geometric stages (support `{r, r+1, …}`,
mean `r/p` steps), propagated through the GLCT forward map. The number of stages is summed out of
the likelihood with a discrete uniform prior on `r_support` (one forward-map run per value), and
drawn afterwards from its exact conditional posterior. Values of `r` above the shortest observed
(uncensored) recovery interval have zero likelihood. `parameter` is `:p` or `:mean_ip`.
"""
struct NegBinGLCT <: InfectiousPeriodModel
    r_support::UnitRange{Int}
    parameter::Symbol
    function NegBinGLCT(r_support::AbstractUnitRange{<:Integer}, parameter::Symbol)
        (!isempty(r_support) && first(r_support) ≥ 1) ||
            throw(ArgumentError("r_support must be a non-empty range of integers ≥ 1; got $r_support"))
        parameter in (:p, :mean_ip) || throw(ArgumentError("NegBinGLCT parameter must be :p or :mean_ip; got :$parameter"))
        return new(UnitRange{Int}(r_support), parameter)
    end
end
NegBinGLCT(r_support::AbstractUnitRange{<:Integer}; parameter::Symbol = :p) = NegBinGLCT(r_support, parameter)

"""
    NegBinBoxcar(r_support, K_age; parameter = :p, truncation = :absorb)

The negative binomial infectious period of [`NegBinGLCT`](@ref) propagated through a Boxcar
(infection-age) map with `K_age` age classes, whose removal probabilities are the discrete hazard
of the distribution truncated at `K_age` ([`discrete_hazard`](@ref); `truncation = :absorb` or
`:condition`). The recovery term uses the untruncated distribution.
"""
struct NegBinBoxcar <: InfectiousPeriodModel
    r_support::UnitRange{Int}
    K_age::Int
    parameter::Symbol
    truncation::Symbol
    function NegBinBoxcar(r_support::AbstractUnitRange{<:Integer}, K_age::Integer, parameter::Symbol, truncation::Symbol)
        (!isempty(r_support) && first(r_support) ≥ 1) ||
            throw(ArgumentError("r_support must be a non-empty range of integers ≥ 1; got $r_support"))
        K_age ≥ 1 || throw(ArgumentError("K_age must be ≥ 1; got $K_age"))
        parameter in (:p, :mean_ip) || throw(ArgumentError("NegBinBoxcar parameter must be :p or :mean_ip; got :$parameter"))
        truncation in (:absorb, :condition) || throw(ArgumentError("truncation must be :absorb or :condition; got :$truncation"))
        return new(UnitRange{Int}(r_support), Int(K_age), parameter, truncation)
    end
end
NegBinBoxcar(r_support::AbstractUnitRange{<:Integer}, K_age::Integer; parameter::Symbol = :p,
             truncation::Symbol = :absorb) = NegBinBoxcar(r_support, K_age, parameter, truncation)

r_support(::GeometricIP) = 1:1
r_support(ip::Union{NegBinGLCT,NegBinBoxcar}) = ip.r_support

"""
    DDSAModel(; infectious_period, priors, population = nothing, terms = :all,
              transmission = :β, guards = NumericalGuards())

Specification of a single-population DDSA model, fitted with [`fit_ddsa`](@ref) or turned into a
Turing model with [`turing_model`](@ref).

# Keywords
- `infectious_period::InfectiousPeriodModel`: [`GeometricIP`](@ref), [`NegBinGLCT`](@ref) or
  [`NegBinBoxcar`](@ref).
- `priors::NamedTuple`: one univariate `Distributions` prior per sampled parameter:
  - transmission: `β` (rate per unit time) or, with `transmission = :R0`, `R0` (then
    `β = R0 / mean_ip`, which requires the `:mean_ip` IP parameter);
  - `ρ`: initially infected fraction (needed whenever the forward map is used);
  - the IP parameter named by `infectious_period.parameter`: `p`, `γ` or `mean_ip`.
- `population`: number at risk in the final-size term: a known integer `N`, a range
  `N_min:N_max` (unknown `N` with a discrete uniform prior, summed out exactly), or an
  [`NGrid`](@ref) (unknown `N` with any prior). Required when `:final_size` is in `terms`.
- `terms`: `:all` or a tuple of likelihood terms from `(:final_size, :infection_times,
  :recovery, :removal_only)` (terms without data contribute zero). Parameters that no term
  needs are not sampled: e.g. `terms = (:recovery,)` samples only the IP parameter.
- `transmission`: `:β` or `:R0`.
- `guards`: [`NumericalGuards`](@ref) applied to the map output.

Throws `ArgumentError` for missing or unused priors, priors that are not univariate
distributions, inconsistent parameterisations, or a missing population.

# Example
```julia
spec = DDSAModel(infectious_period = NegBinGLCT(1:10),
                 priors = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1)),
                 population = 99)
```
"""
struct DDSAModel{IP<:InfectiousPeriodModel, P<:NamedTuple, Pop}
    infectious_period::IP
    priors::P
    population::Pop
    terms::Tuple{Vararg{Symbol}}
    transmission::Symbol
    guards::NumericalGuards
end

function DDSAModel(; infectious_period::InfectiousPeriodModel, priors::NamedTuple, population = nothing,
                   terms = :all, transmission::Symbol = :β, guards::NumericalGuards = NumericalGuards())
    tset = terms === :all ? DDSA_TERMS : Tuple(terms)
    isempty(tset) && throw(ArgumentError("at least one likelihood term is needed"))
    _check_terms(tset)
    transmission in (:β, :R0) || throw(ArgumentError("transmission must be :β or :R0; got :$transmission"))
    needs_map = any(in(tset), (:final_size, :infection_times, :removal_only))
    ippar = infectious_period.parameter
    transmission === :R0 && ippar !== :mean_ip &&
        throw(ArgumentError("transmission = :R0 needs the :mean_ip infectious-period parameter (β = R0 / mean_ip)"))
    required = needs_map ? (transmission, :ρ, ippar) : (ippar,)
    for k in required
        haskey(priors, k) || throw(ArgumentError("a prior for $k is required (priors given: $(keys(priors)))"))
    end
    for k in keys(priors)
        k in required || throw(ArgumentError("prior for $k is not used by this model (sampled parameters: $required)"))
        priors[k] isa UnivariateDistribution ||
            throw(ArgumentError("the prior for $k must be a univariate distribution; got $(typeof(priors[k]))"))
    end
    if :final_size in tset
        population === nothing && throw(ArgumentError("the final-size term needs `population` (N, N_min:N_max or an NGrid)"))
        population isa Union{Integer, AbstractUnitRange{<:Integer}, NGrid} ||
            throw(ArgumentError("population must be an integer, a range or an NGrid; got $(typeof(population))"))
    end
    return DDSAModel(infectious_period, priors, population, tset, transmission, guards)
end

function Base.show(io::IO, spec::DDSAModel)
    print(io, "DDSAModel(", spec.infectious_period, ", parameters: ", join(keys(spec.priors), ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::DDSAModel)
    g = spec.guards
    println(io, "DDSAModel")
    println(io, "  infectious period: ", spec.infectious_period)
    println(io, "  priors:            ", join(("$k ~ $(repr(d))" for (k, d) in pairs(spec.priors)), ", "))
    _needs_map(spec) && println(io, "  transmission:      ", spec.transmission)
    spec.population === nothing || println(io, "  population:        ", spec.population)
    println(io, "  terms:             ", join(spec.terms, ", "))
    print(io, "  guards:            τ_bounds = ", g.τ_bounds, ", f_floor = ", g.f_floor)
end

_needs_map(spec::DDSAModel) = any(in(spec.terms), (:final_size, :infection_times, :removal_only))

"""Population of the final-size term for this data set (a range becomes a uniform `NGrid`)."""
_resolve_population(pop::Integer, data) = Int(pop)
_resolve_population(pop::NGrid, data) = pop
_resolve_population(pop::AbstractUnitRange{<:Integer}, data) = NGrid(data.final_size, first(pop), last(pop))
_resolve_population(::Nothing, data) = nothing

function _check_data(spec::DDSAModel, data::LineList, pop)
    if pop isa Integer
        pop ≥ data.final_size || throw(ArgumentError("population N = $pop is smaller than the final size K = $(data.final_size)"))
    elseif pop isa NGrid
        pop.K == data.final_size || throw(ArgumentError("the NGrid was built for K = $(pop.K), but the data have K = $(data.final_size)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Per-r likelihood
# ---------------------------------------------------------------------------------------------

"""Per-step progression probability from the sampled IP parameter."""
function _progression(ip::InfectiousPeriodModel, r, value, Δt)
    ip.parameter === :p && return value
    ip.parameter === :γ && return rate_to_proportion(value, Δt)
    return r * Δt / value                   # :mean_ip
end

_dph(::GeometricIP, r, p) = dph_geometric(p)
_dph(::Union{NegBinGLCT,NegBinBoxcar}, r, p) = dph_negative_binomial(r, p)

_structure(::Union{GeometricIP,NegBinGLCT}, r, p, dph) = GLCT(dph)
function _structure(ip::NegBinBoxcar, r, p, dph)
    h = ip.truncation === :condition ? PhaseTypeDistributions._hazard_negbin(0, r, p, ip.K_age) :
                                       PhaseTypeDistributions._hazard_iterate(dph, ip.K_age)
    return Boxcar(h)
end

"""
Log-likelihood for `r` stages given (β, IP value, ρ); returns `(loglik, τ)`. Infeasible values
(`r` above an uncensored recovery interval, `p ∉ (0, 1)`, non-finite map output) return `-Inf`.
"""
function _loglik_r(spec::DDSAModel, data::LineList, pop, r, β, ipvalue, ρ)
    ip = spec.infectious_period
    p = _progression(ip, r, ipvalue, data.Δt)
    T = typeof(float(p) * (β === nothing ? 1.0 : β) * (ρ === nothing ? 1.0 : ρ))
    NaNτ = T(NaN)
    if :recovery in spec.terms
        # an observed interval below r is outside {r, r+1, …}; a censored one only says W > δ
        any(<(r), observed_intervals(data)) && return (oftype(NaNτ, -Inf), NaNτ)
    end
    _is_probability(p) || return (oftype(NaNτ, -Inf), NaNτ)
    d = _dph(ip, r, p)
    if !_needs_map(spec)
        return (recovery_loglik(d, data.recovery_intervals, data.censored) + zero(NaNτ), NaNτ)
    end
    sol = forward_map(_structure(ip, r, p, d), β, ρ, data.nsteps; Δt = data.Δt)
    (isfinite(sol.τ) && all(isfinite, sol.f)) || return (oftype(NaNτ, -Inf), NaNτ)
    res = ddsa_loglikelihood(sol, data, d; population = pop, guards = spec.guards, terms = spec.terms)
    return (res.loglik, res.τ)
end

# ---------------------------------------------------------------------------------------------
# Turing model
# ---------------------------------------------------------------------------------------------

@model function _ddsa_turing(spec, data, pop)
    pri = spec.priors
    ip = spec.infectious_period
    needs_map = _needs_map(spec)
    βv = nothing
    if needs_map && spec.transmission === :β
        β ~ pri.β
        βv = β
    elseif needs_map
        R0 ~ pri.R0
    end
    if ip.parameter === :p
        p ~ pri.p
        ipv = p
    elseif ip.parameter === :γ
        γ ~ pri.γ
        ipv = γ
    else
        mean_ip ~ pri.mean_ip
        ipv = mean_ip
    end
    ρv = nothing
    if needs_map
        ρ ~ pri.ρ
        ρv = ρ
        spec.transmission === :R0 && (βv = R0 / mean_ip)
    end
    support = r_support(ip)
    if length(support) == 1
        ll, τ = _loglik_r(spec, data, pop, first(support), βv, ipv, ρv)
        Turing.@addlogprob! ll
        return (lps = [ll], τ = [τ])
    else
        res = [_loglik_r(spec, data, pop, r, βv, ipv, ρv) for r in support]
        logmarg, lps = marginal_r_loglik(i -> res[i][1], eachindex(res))
        Turing.@addlogprob! logmarg
        return (lps = lps, τ = [x[2] for x in res])
    end
end

"""
    turing_model(spec::DDSAModel, data::LineList) -> DynamicPPL.Model

Turing model of `spec` for `data`. Its parameters are those named in `spec.priors`; the number of
stages (if any) is summed out of the likelihood and an unknown population size is summed out of
the final-size term. The model returns `(lps, τ)`: the per-`r` log-likelihoods and guarded attack
fractions, from which [`fit_ddsa`](@ref) draws `r` and `N` after sampling. Use it directly to
sample with other Turing settings, or to evaluate `logjoint`/`loglikelihood` at given parameters.
"""
function turing_model(spec::DDSAModel, data::LineList)
    pop = _resolve_population(spec.population, data)
    _check_data(spec, data, pop)
    return _ddsa_turing(spec, data, pop)
end
