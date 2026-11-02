# The DDSA log-likelihood: final size (ℓ_constr), infection times (ℓ_I), recovery intervals (ℓ_R),
# plus the right-censoring and removal-only (convolution) terms used by the manuscript examples.

"""
    NumericalGuards(; τ_bounds = (0.0, 1.0), f_floor = 0.0)

Numerical guards applied to the forward-map output before the likelihood is evaluated.

- `τ_bounds = (lo, hi)`: the attack fraction is clamped to `[lo, hi]` before it enters the
  final-size term. The default `(0, 1)` leaves the exact likelihood unchanged (the final-size term
  is `-Inf` at a boundary that is incompatible with the data).
- `f_floor`: if positive, the infection-time distribution is floored at this value and
  renormalised before the infection-time and removal-only terms.

The guards only matter when the map produces attack fractions or infection probabilities at those
extremes.
"""
struct NumericalGuards
    τ_bounds::Tuple{Float64,Float64}
    f_floor::Float64
    function NumericalGuards(τ_bounds::Tuple{Real,Real}, f_floor::Real)
        lo, hi = τ_bounds
        (0 ≤ lo < hi ≤ 1) || throw(ArgumentError("τ_bounds must satisfy 0 ≤ lo < hi ≤ 1; got $τ_bounds"))
        f_floor ≥ 0 || throw(ArgumentError("f_floor must be ≥ 0; got $f_floor"))
        return new((Float64(lo), Float64(hi)), Float64(f_floor))
    end
end
NumericalGuards(; τ_bounds::Tuple{Real,Real} = (0.0, 1.0), f_floor::Real = 0.0) = NumericalGuards(τ_bounds, f_floor)

_guard_τ(τ, g::NumericalGuards) = clamp(τ, g.τ_bounds[1], g.τ_bounds[2])

"""Infection-time distribution after the guards, or `nothing` if it is not a valid distribution."""
function _guard_f(f::AbstractVector, g::NumericalGuards)
    n = length(f)
    fs = normalise_pmf(f, n)
    if g.f_floor > 0
        fs = max.(fs, g.f_floor)
        fs = fs ./ sum(fs)
    end
    (all(isfinite, fs) && abs(sum(fs) - 1) < 0.01) || return nothing
    return fs
end

# ---------------------------------------------------------------------------------------------
# Term functions
# ---------------------------------------------------------------------------------------------

"""
    final_size_loglik(population, K, τ)

Final-size term ℓ_constr: `log P(K | N, τ)` for `K ~ Binomial(N, τ)`.

- `population::Integer`: known number at risk `N` (`N ≥ K`).
- `population::NGrid`: unknown `N`, summed out over the grid's prior support with
  [`marginal_N_loglik`](@ref) (the grid must have been built for this `K`).

`τ` may be a ForwardDiff `Dual`.
"""
function final_size_loglik(N::Integer, K::Integer, τ)
    N ≥ K || throw(ArgumentError("population N = $N is smaller than the final size K = $K"))
    return binomlogpdf(N, τ, K)
end

function final_size_loglik(grid::NGrid, K::Integer, τ)
    grid.K == K || throw(ArgumentError("the NGrid was built for K = $(grid.K), not K = $K"))
    return marginal_N_loglik(grid, τ)
end

"""
    infection_time_loglik(counts, f)

Infection-time term ℓ_I: `log Multinomial(counts; sum(counts), f)`, the probability of the
observed histogram of infection steps given the normalised infection-time distribution `f` of the
map (same length as `counts`). Zero when there are no counts.
"""
function infection_time_loglik(counts::AbstractVector{<:Integer}, f::AbstractVector)
    length(counts) == length(f) || throw(DimensionMismatch("counts and f must have the same length"))
    K = sum(counts)
    K > 0 || return zero(eltype(f))
    return logpdf(Multinomial(K, f), counts)
end

"""
    recovery_loglik(d, intervals, censored = falses(length(intervals)))

Recovery term ℓ_R: `Σ log P(W = δᵢ)` over the observed intervals plus `Σ log P(W > δᵢ)` over the
right-censored ones, for the infectious-period distribution `d` (a [`DPH`](@ref) or any
discrete distribution with `logpdf`/`logccdf`).
"""
function recovery_loglik(d, intervals::AbstractVector{<:Integer},
                         censored::AbstractVector{Bool} = falses(length(intervals)))
    length(censored) == length(intervals) || throw(DimensionMismatch("censored and intervals must have the same length"))
    ll = zero(float(eltype(_param_type(d))))
    for i in eachindex(intervals)
        ll += censored[i] ? logccdf(d, intervals[i]) : logpdf(d, intervals[i])
    end
    return ll
end

_param_type(d::DPH) = d.s
_param_type(d) = [0.0]

"""
    removal_only_loglik(d, f, removal_steps)

Likelihood of cases whose infection time is unknown but whose removal step `T` is known: for each,
`log Σ_{u=1}^{min(T-1, nsteps)} f[u] P(W = T - u)`, summing over the possible infection steps
`u` with the infection-time distribution `f` and the infectious-period distribution `d`. A removal
at step 1 has likelihood zero (`-Inf`).
"""
function removal_only_loglik(d, f::AbstractVector, removal_steps::AbstractVector{<:Integer})
    nst = length(f)
    ll = zero(eltype(f))
    for T_rem in removal_steps
        u_max = min(T_rem - 1, nst)
        terms = (log(f[u]) + logpdf(d, T_rem - u) for u in 1:u_max)
        ll += foldl(logaddexp, terms; init = oftype(ll, -Inf))
    end
    return ll
end

# ---------------------------------------------------------------------------------------------
# Combined likelihood
# ---------------------------------------------------------------------------------------------

"""Likelihood terms of DDSA, as named in `DDSAModel(terms = …)` and `ddsa_loglikelihood`."""
const DDSA_TERMS = (:final_size, :infection_times, :recovery, :removal_only)

"""
    ddsa_loglikelihood(sol::MapSolution, data::LineList, d; population, group = 1,
                       guards = NumericalGuards(), terms = DDSA_TERMS) -> (loglik, τ)

DDSA log-likelihood of the line list `data` given a forward-map solution `sol` (group `group` of
a multi-group solution) and the infectious-period distribution `d`, summing the requested `terms`:

- `:final_size`: `final_size_loglik(population, data.final_size, τ)` (needs `population`:
  a known `N` or an [`NGrid`](@ref));
- `:infection_times`: `infection_time_loglik(data.infection_counts, f)`;
- `:recovery`: `recovery_loglik(d, data.recovery_intervals, data.censored)`;
- `:removal_only`: `removal_only_loglik(d, f, data.removal_only_steps)`.

`τ` and `f` are taken from `sol` after the [`NumericalGuards`](@ref). Returns the log-likelihood
(`-Inf` if the guarded infection-time distribution is invalid) and the guarded attack fraction
`τ`, which is what is needed to draw `N` afterwards with [`sample_N`](@ref).

For a model with several groups, call it once per group with the terms that belong to that group
and add the shared terms (e.g. a recovery term pooled across groups) once.
"""
function ddsa_loglikelihood(sol::MapSolution, data::LineList, d; population = nothing, group::Integer = 1,
                            guards::NumericalGuards = NumericalGuards(), terms = DDSA_TERMS)
    _check_terms(terms)
    nsteps(sol) == data.nsteps || throw(DimensionMismatch(
        "the map has $(nsteps(sol)) steps but the line list has $(data.nsteps)"))
    τ = _guard_τ(attack_fraction(sol, group), guards)
    ll = zero(τ)
    if :final_size in terms
        population === nothing && throw(ArgumentError("the final-size term needs `population` (N or an NGrid)"))
        ll += final_size_loglik(population, data.final_size, τ)
    end
    needs_f = (:infection_times in terms && sum(data.infection_counts) > 0) ||
              (:removal_only in terms && !isempty(data.removal_only_steps))
    if needs_f
        f = _guard_f(infection_pmf(sol, group), guards)
        f === nothing && return (loglik = oftype(ll, -Inf), τ = τ)
        :infection_times in terms && (ll += infection_time_loglik(data.infection_counts, f))
        if :recovery in terms
            ll += recovery_loglik(d, data.recovery_intervals, data.censored)
        end
        :removal_only in terms && (ll += removal_only_loglik(d, f, data.removal_only_steps))
    elseif :recovery in terms
        ll += recovery_loglik(d, data.recovery_intervals, data.censored)
    end
    return (loglik = ll, τ = τ)
end

function _check_terms(terms)
    for t in terms
        t in DDSA_TERMS || throw(ArgumentError("unknown likelihood term :$t; choose from $(DDSA_TERMS)"))
    end
    return nothing
end
