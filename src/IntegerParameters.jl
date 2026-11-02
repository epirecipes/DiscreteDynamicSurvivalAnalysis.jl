"""
    IntegerParameters

Exact summation of integer-valued parameters out of a likelihood, so the real-valued parameters
can be sampled with NUTS alone (no Gibbs/MH blocks).

* **Summing over N** (`NGrid`, `marginal_N_loglik`): a population size `N` that enters the
  likelihood only through the Binomial final-size term `K ~ Binomial(N, τ)`. `log C(N, K)` is
  precomputed, so supports of thousands of values cost one vectorised pass per evaluation.
* **Summing over r** (`marginal_r_loglik`): any finite set of values (a phase count `r`, or pairs
  such as `(r, N₂)`) whose likelihood must be recomputed for each value, e.g. because it changes
  the forward map or the infectious-period distribution. This costs one likelihood evaluation per
  value, so keep the support small (tens of values).

After sampling, posterior draws of the summed-out parameters are recovered with `sample_N` and
`sample_index`, using the exact conditional posterior for each MCMC draw.

Which helper to use:

* `N` enters only `K ~ Binomial(N, τ)`: `NGrid` + `marginal_N_loglik`.
* The parameter changes the forward map or the recovery distribution (e.g. `r`, or `N` when
  `ρ = 1/N`): `marginal_r_loglik` over its support.
* Several parameters each change the map: sum over their joint grid (`Iterators.product`), at a
  cost equal to the product of the support sizes. Parameters that enter separate Binomial terms
  each get one `NGrid`, added together.

`marginal_N_loglik` and `marginal_r_loglik` accept ForwardDiff `Dual` arguments (τ and
log-likelihoods); `N_posterior_weights`, `posterior_weights`, `sample_N` and `sample_index` work in
`Float64` and are meant for use after sampling.
"""
module IntegerParameters

using Random
using SpecialFunctions: loggamma

export NGrid, marginal_N_loglik, N_posterior_weights, sample_N,
       marginal_r_loglik, posterior_weights, sample_index

"""Numerically stable log(sum(exp.(v))); returns -Inf when every entry is -Inf."""
function _logsumexp(v)
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(exp(x - m) for x in v))
end

# =============================================================================
# Summing over N (Binomial final-size term)
# =============================================================================

"""
    NGrid(K, N_min, N_max; logprior = nothing)

Precomputed support for summing a population size `N ∈ N_min:N_max` out of the final-size term
`K ~ Binomial(N, τ)`. `log C(N, K)` is computed once here, so each likelihood evaluation costs one
vectorised pass over the support.

`logprior` is an optional vector of log prior weights over `N_min:N_max` (they need not be
normalised); the default is a discrete uniform prior, matching `N ~ DiscreteUniform(N_min, N_max)`.

Throws `ArgumentError` if `N_min < K`, `N_max < N_min`, `logprior` has the wrong length, or every
prior weight is zero.

# Fields
- `K::Int`: the observed final size
- `Ns::Vector{Int}`: the support `N_min:N_max`
- `logC::Vector{Float64}`: `log C(N, K)` for each `N`
- `logprior::Vector{Float64}`: normalised log prior weights
"""
struct NGrid
    K::Int
    Ns::Vector{Int}
    logC::Vector{Float64}       # log C(N, K)
    logprior::Vector{Float64}   # normalised log prior over Ns
end

function NGrid(K::Integer, N_min::Integer, N_max::Integer; logprior = nothing)
    K >= 0 || throw(ArgumentError("K = $K must be ≥ 0"))
    N_min >= K || throw(ArgumentError("N_min = $N_min must be ≥ K = $K"))
    N_max >= N_min || throw(ArgumentError("N_max = $N_max must be ≥ N_min = $N_min"))
    Ns   = collect(Int, N_min:N_max)
    logC = [loggamma(n + 1.0) - loggamma(K + 1.0) - loggamma(n - K + 1.0) for n in Ns]
    lp   = logprior === nothing ? fill(0.0, length(Ns)) : collect(Float64, logprior)
    length(lp) == length(Ns) || throw(ArgumentError("logprior must have length $(length(Ns)); got $(length(lp))"))
    norm = _logsumexp(lp)
    isfinite(norm) || throw(ArgumentError("logprior must give positive weight to at least one N"))
    lp .-= norm
    return NGrid(Int(K), Ns, logC, lp)
end

Base.show(io::IO, g::NGrid) = print(io, "NGrid(K = $(g.K), N ∈ $(first(g.Ns)):$(last(g.Ns)))")

"""
    marginal_N_loglik(grid::NGrid, τ)

`log Σ_N π(N) Binomial(K; N, τ)` over the grid's support. Use it in place of
`N ~ DiscreteUniform(...)` plus `K ~ Binomial(N, τ)`:

    Turing.@addlogprob! marginal_N_loglik(grid, τ)

Returns `-Inf` unless `0 < τ < 1`. Only valid when `N` enters the likelihood solely through this
Binomial term. If `N` also changes the forward map (for example `ρ = 1/N`), sum over it with
`marginal_r_loglik` instead.
"""
function marginal_N_loglik(grid::NGrid, τ)
    (0 < τ < 1) || return oftype(float(τ), -Inf)
    l1τ = log1p(-τ)
    return grid.K * log(τ) +
           _logsumexp(grid.logprior[i] + grid.logC[i] + (grid.Ns[i] - grid.K) * l1τ
                      for i in eachindex(grid.Ns))
end

"""
    marginal_N_loglik(grid::NGrid, logτ, log1mτ)

Log-space form of `marginal_N_loglik(grid, τ)`, taking `log τ` and `log(1 − τ)` computed directly
by a log-scale forward map (no clamping of τ).
"""
function marginal_N_loglik(grid::NGrid, logτ, log1mτ)
    return grid.K * logτ +
           _logsumexp(grid.logprior[i] + grid.logC[i] + (grid.Ns[i] - grid.K) * log1mτ
                      for i in eachindex(grid.Ns))
end

"""
    N_posterior_weights(grid::NGrid, τ)

Normalised posterior probabilities `P(N | K, τ)` over `grid.Ns` (`Float64`).
"""
function N_posterior_weights(grid::NGrid, τ::Real)
    l1τ = log1p(-float(τ))
    w = grid.logprior .+ grid.logC .+ (grid.Ns .- grid.K) .* l1τ
    return posterior_weights(w)
end

"""
    sample_N([rng], grid::NGrid, τ)

Draw `N` from its exact conditional posterior `P(N | K, τ)`. Call once per MCMC draw of `τ` to
obtain posterior samples of `N`. Without `rng`, `Random.default_rng()` is used.
"""
sample_N(rng::AbstractRNG, grid::NGrid, τ::Real) =
    grid.Ns[_draw(rng, N_posterior_weights(grid, τ))]
sample_N(grid::NGrid, τ::Real) = sample_N(Random.default_rng(), grid, τ)

# =============================================================================
# Summing over r (any finite discrete index)
# =============================================================================

"""
    marginal_r_loglik(loglik_r, support; logprior = nothing, threaded = false)

Sum a discrete parameter out of the likelihood:

    log Σ_{r ∈ support} π(r) exp(loglik_r(r))

`loglik_r(r)` returns the log-likelihood for value `r` given the current real-valued parameters;
return `-Inf` for infeasible values (for example `r > minimum(delta)` for a NegBin DPH) so they
cost nothing. `support` can be any finite collection, including tuples for a joint grid such as
`Iterators.product(1:4, 68:77)`.

`logprior` is an optional vector of log prior weights in the same order as `support`; the default
is uniform over `support`.

`threaded = true` evaluates the values as parallel tasks (when Julia runs with more than one
thread). This helps when each `loglik_r` call is expensive, for example a forward-map run per
value over a large grid. `loglik_r` must then be safe to call concurrently (no shared mutable
state). The result is identical to the serial one.

Returns `(logmarg, lps)`, where `lps` holds the per-value log-likelihoods. Return `lps` from the
Turing model so the posterior of `r` can be recovered afterwards with `posterior_weights` or
`sample_index`:

    logmarg, lps = marginal_r_loglik(r -> ll_given_r(r, β, ρ, p), 1:r_max)
    Turing.@addlogprob! logmarg
    return lps
"""
function marginal_r_loglik(loglik_r, support; logprior = nothing, threaded = false)
    vals = collect(support)
    isempty(vals) && throw(ArgumentError("support must not be empty"))
    lps  = if threaded && Threads.nthreads() > 1
        # one task per value; results are collected in support order, so the sum is
        # identical to the serial version
        fetch.([Threads.@spawn loglik_r(v) for v in vals])
    else
        [loglik_r(v) for v in vals]
    end
    lp   = logprior === nothing ? fill(-log(length(vals)), length(vals)) :
           collect(Float64, logprior) .- _logsumexp(logprior)
    length(lp) == length(lps) || throw(ArgumentError("logprior must match the support length $(length(lps))"))
    return _logsumexp(lp[i] + lps[i] for i in eachindex(lps)), lps
end

"""
    posterior_weights(lps; logprior = nothing)

Normalised posterior probabilities for each value of a summed-out parameter, given the per-value
log-likelihoods `lps` (as returned by `marginal_r_loglik`; `Float64`). If every entry is `-Inf`
the weights are uniform. Averaging these over MCMC draws gives the (Rao–Blackwellised) posterior
`P(r | data)`.
"""
function posterior_weights(lps; logprior = nothing)
    l = Float64.(collect(lps))
    logprior === nothing || (l .+= logprior)
    m = maximum(l)
    isfinite(m) || return fill(1 / length(l), length(l))
    w = exp.(l .- m)
    return w ./ sum(w)
end

"""
    sample_index([rng], lps; logprior = nothing)

Draw the index of a summed-out value from its exact conditional posterior, given per-value
log-likelihoods `lps` for one MCMC draw. Use `support[sample_index(rng, lps)]` (or
`CartesianIndices` for a matrix of `lps`) to get the value itself. Without `rng`,
`Random.default_rng()` is used.
"""
sample_index(rng::AbstractRNG, lps; logprior = nothing) =
    _draw(rng, posterior_weights(vec(collect(lps)); logprior = logprior))
sample_index(lps; logprior = nothing) =
    sample_index(Random.default_rng(), lps; logprior = logprior)

"""Inverse-CDF categorical draw: one uniform, smallest index whose cumulative weight reaches it."""
function _draw(rng::AbstractRNG, w::AbstractVector{<:Real})
    u = rand(rng) * sum(w)
    c = zero(eltype(w))
    for i in eachindex(w)
        c += w[i]
        c >= u && return i
    end
    return lastindex(w)
end

end # module IntegerParameters
