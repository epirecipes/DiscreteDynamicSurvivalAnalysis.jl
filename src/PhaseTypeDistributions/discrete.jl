# =============================================================================
# DISCRETE PHASE-TYPE DISTRIBUTIONS
# =============================================================================

"""
    DPH{T<:Real} <: DiscreteUnivariateDistribution

A discrete phase-type distribution: the number of steps `W ≥ 1` to absorption of a
finite-state discrete-time Markov chain.

# Fields
- `α::Vector{T}`: Initial probability distribution over the k transient states (sums to 1)
- `S::Matrix{T}`: k×k sub-stochastic transition matrix between transient states (rows sum ≤ 1)
- `s::Vector{T}`: Absorption probabilities per state, `s = 1 - S*1`

`P(W = n) = α' S^(n-1) s` for `n ≥ 1`. The element type `T` may be a ForwardDiff `Dual`.

# Example
```julia
d = DPH([1.0], reshape([0.7], 1, 1))   # geometric with per-step absorption probability 0.3
pdf(d, 1), mean(d)                      # 0.3, 3.33
```
"""
struct DPH{T<:Real} <: DiscreteUnivariateDistribution
    α::Vector{T}  # Initial distribution over transient states
    S::Matrix{T}  # Sub-stochastic transition matrix
    s::Vector{T}  # Absorption probabilities per state

    function DPH{T}(α::Vector{T}, S::Matrix{T}) where {T<:Real}
        k = length(α)
        size(S, 1) == k == size(S, 2) ||
            throw(DimensionMismatch("transition matrix S must be k×k with k = length(α) = $k; got $(size(S))"))
        abs(sum(α) - one(T)) < DEFAULT_TOLERANCE ||
            throw(ArgumentError("initial distribution α must sum to 1; got sum $(sum(α))"))
        all(α .>= 0) || throw(ArgumentError("initial distribution α must be non-negative"))
        all(S .>= 0) || throw(ArgumentError("transition matrix S must be non-negative"))
        row_sums = sum(S, dims=2)
        all(row_sums .<= 1 .+ eps(T)) || throw(ArgumentError("rows of S must sum to ≤ 1"))
        s = ones(T, k) .- vec(S * ones(T, k))
        all(s .>= 0) || throw(ArgumentError("absorption probabilities 1 - S*1 must be non-negative"))
        return new{T}(α, S, s)
    end
end

"""
    DPH(α::AbstractVector, S::AbstractMatrix)

Construct a discrete phase-type distribution from the initial distribution `α` and the
sub-stochastic transition matrix `S`. The element type is the floating-point promotion of those
of `α` and `S` (so integer inputs are accepted).

Throws `DimensionMismatch` if `S` is not `length(α) × length(α)`, and `ArgumentError` if `α` does
not sum to 1 (within `DEFAULT_TOLERANCE`) or has negative entries, or if `S` has negative entries
or a row sum above 1.
"""
function DPH(α::AbstractVector, S::AbstractMatrix)
    T = float(promote_type(eltype(α), eltype(S)))
    return DPH{T}(Vector{T}(α), Matrix{T}(S))
end

# =============================================================================
# DISCRETE PHASE-TYPE: CORE COMPUTATIONS
# =============================================================================

"""
    _fundamental(dph::DPH)

Fundamental matrix `N = (I - S)^{-1}` (expected number of visits to each transient state).
"""
function _fundamental(dph::DPH)
    k = length(dph.α)
    I_matrix = Matrix{eltype(dph.α)}(LinearAlgebra.I, k, k)
    return inv(I_matrix - dph.S)
end

"""
    mean(dph::DPH)

Mean number of steps to absorption, `μ = α' N 1` with `N = (I - S)^{-1}`.
"""
function mean(dph::DPH)
    k = length(dph.α)
    N = _fundamental(dph)
    return (dph.α' * (N * ones(eltype(dph.α), k)))[1]
end

"""
    var(dph::DPH)

Variance of the number of steps to absorption: `Var(W) = 2 α' N² S 1 + μ - μ²`, where
`E[W(W-1)] = 2 α' N² S 1` is the second factorial moment and `N = (I - S)^{-1}`.
"""
function var(dph::DPH)
    k = length(dph.α)
    N = _fundamental(dph)
    onev = ones(eltype(dph.α), k)
    μ = (dph.α' * (N * onev))[1]
    term = (dph.α' * (N * (N * (dph.S * onev))))[1]  # α' N^2 S 1
    return 2 * term + μ - μ^2
end

"""
    pmf(d, n::Integer)

Probability mass `P(W = n)`. For a `DPH`, `α' S^(n-1) s` (zero for `n < 1`). Equivalent to
`pdf(d, n)`.
"""
function pmf(dph::DPH, n::Integer)
    n ≥ 1 || return zero(eltype(dph.α))
    if n == 1
        return (dph.α' * dph.s)[1]
    else
        return (dph.α' * (dph.S^(n-1) * dph.s))[1]
    end
end

pdf(dph::DPH, x::Integer) = pmf(dph, x)

"""
    cdf(dph::DPH, n::Integer)

`P(W ≤ n) = 1 - α' S^n 1` (zero for `n < 1`).
"""
function cdf(dph::DPH, n::Integer)
    n ≥ 1 || return zero(eltype(dph.α))
    k = length(dph.α)
    return 1 - (dph.α' * (dph.S^n * ones(eltype(dph.α), k)))[1]
end

"""
    ccdf(dph::DPH, n::Integer)

Survival function `P(W > n) = α' S^n 1` (one for `n < 1`).
"""
function ccdf(dph::DPH, n::Integer)
    n ≥ 1 || return one(eltype(dph.α))
    k = length(dph.α)
    return (dph.α' * (dph.S^n * ones(eltype(dph.α), k)))[1]
end

"""
    logccdf(dph::DPH, n::Integer)

`log P(W > n)`, the log-likelihood of an interval right-censored after `n` steps. For a
geometric or (shifted) negative binomial DPH it is computed in closed form with scalar
arithmetic (ForwardDiff-safe): for `c` delay steps followed by `r` stages with progression
probability `p`, `P(W > n)` is the probability of fewer than `r` progressions in `n - c` steps,
which for `r = 1` is `(n - c) * log1p(-p)`. Other DPHs use `log(ccdf(dph, n))`.
"""
function logccdf(dph::DPH, n::Integer)
    st = _shifted_negbin_structure(dph)
    st === nothing && return log(ccdf(dph, n))
    c, r, p = st
    m = n - c                                   # steps spent in the NegBin stages
    T = typeof(float(p))
    m < r && return zero(T)                     # cannot have completed r stages yet
    r == 1 && return m * log1p(-p)
    # log Σ_{j=0}^{r-1} C(m, j) p^j (1-p)^(m-j)
    acc = T(-Inf)
    for j in 0:(r - 1)
        lt = T(logabsbinomial(m, j)[1]) + (iszero(j) ? zero(T) : j * log(T(p))) + (m - j) * log1p(-T(p))
        acc = _logaddexp(acc, lt)
    end
    return acc
end

"""
Numerically stable `log(exp(a) + exp(b))`, exact when either argument is `-Inf`. Written with
`min`/`max` rather than `abs(a - b)`: at a tie they pick different arguments, so the ForwardDiff
derivative is `(a' + b') / 2` there (with `abs` it was `(3b' - a') / 2`).
"""
_logaddexp(a, b) = a == -Inf ? b : b == -Inf ? a : max(a, b) + log1p(exp(min(a, b) - max(a, b)))

"""
    _shifted_negbin_structure(dph) -> (c, r, p) or nothing

Detect a DPH that is `c ≥ 0` deterministic one-step delay states followed by a NegBin chain of
`r` stages with per-step progression probability `p`, i.e. the structure built by
`dph_negative_binomial` (c = 0) and `dph_prepend_deterministic_delay(dph_negative_binomial(r, p), c)`.
Every entry of `α` and `S` is checked, so any other DPH returns `nothing`. The structural zeros
and ones must be exact constants (with ForwardDiff they carry zero partials, as built by the
constructors); the remaining entries are compared with `isapprox(...; atol = 1e-12)`.
"""
function _shifted_negbin_structure(dph::DPH)
    α, S, s = dph.α, dph.S, dph.s
    k = length(s)
    isone(α[1]) && all(i -> iszero(α[i]), 2:k) || return nothing
    # leading deterministic delay states: S[i, i+1] == 1, nothing else in the row
    c = 0
    while c < k - 1 && iszero(S[c+1, c+1]) && isone(S[c+1, c+2]) && iszero(s[c+1])
        c += 1
    end
    r = k - c
    p = s[k]
    (zero(p) < p <= one(p)) || return nothing
    for i in 1:k, j in 1:k
        target = if i <= c
            j == i + 1 ? one(p) : zero(p)
        elseif j == i
            one(p) - p
        elseif j == i + 1
            p
        else
            zero(p)
        end
        isapprox(S[i, j], target; atol = 1e-12) || return nothing
    end
    return (c, r, p)
end

"""
    logpdf(dph::DPH, x::Integer)

Log probability mass. Geometric, negative binomial and delayed negative binomial DPHs (detected by
`_shifted_negbin_structure`) use the closed form `negbin_total_logpdf` with scalar arithmetic,
which is exact and ForwardDiff-safe. Other DPHs use `log(max(pmf(dph, x), 1e-12))`. Returns
`-Inf` for `x < 1`.
"""
function logpdf(dph::DPH, x::Integer)
    st = _shifted_negbin_structure(dph)
    if st !== nothing
        c, r, p = st
        return negbin_total_logpdf(x - c, r, p)
    end
    x ≥ 1 || return oftype(float(dph.α[1]), -Inf)
    pmf_val = pmf(dph, x)
    return log(max(pmf_val, 1e-12))
end

"""
    negbin_total_logpdf(k, r, p)

Log-PMF of the negative binomial counting TOTAL trials (support `{r, r+1, ...}`):
`P(W = k) = C(k-1, r-1) p^r (1-p)^(k-r)`. Scalar arithmetic only, so it is safe for ForwardDiff
when `p` is a Dual number. For `p = 1` it returns `0` at `k = r` and `-Inf` above.

The method with real-valued `r` uses `loggamma` instead of the binomial coefficient
(support `{⌈r⌉, ⌈r⌉+1, ...}`), so ForwardDiff can differentiate through `r`:
`log P(W=k) = loggamma(k) - loggamma(r) - loggamma(k-r+1) + r log(p) + (k-r) log(1-p)`.
"""
function negbin_total_logpdf(k::Integer, r::Integer, p::Real)
    k < r && return oftype(float(p), -Inf)
    T  = typeof(float(p))
    lc = logabsbinomial(k - 1, r - 1)[1]
    tail = k == r ? zero(T) : (k - r) * log(one(T) - T(p))
    return T(lc) + r * log(T(p)) + tail
end

function negbin_total_logpdf(k::Integer, r::Real, p::Real)
    T  = promote_type(typeof(float(r)), typeof(float(p)))
    kr = T(k) - T(r)
    kr < 0 && return T(-Inf)
    lc = loggamma(T(k)) - loggamma(T(r)) - loggamma(kr + one(T))
    tail = iszero(kr) ? zero(T) : kr * log(one(T) - T(p))
    return lc + T(r) * log(T(p)) + tail
end

# =============================================================================
# DISCRETE PHASE-TYPE: DISTRIBUTIONS.JL INTERFACE
# =============================================================================

support(::DPH) = 1:typemax(Int)
Base.minimum(::DPH) = 1
Base.maximum(::DPH) = Inf          # unbounded, the Distributions.jl convention (cf. Poisson)
insupport(::DPH, x::Real) = isa(x, Integer) && x >= 1
Base.eltype(::Type{<:DPH}) = Int
Base.eltype(::DPH) = Int

"""
    rand(rng::AbstractRNG, dph::DPH)

Draw the number of steps to absorption by inversion of the CDF: one uniform `u = rand(rng)`,
then the smallest `n` with `pmf(1) + … + pmf(n) ≥ u`.
"""
function Base.rand(rng::AbstractRNG, dph::DPH)
    u = rand(rng)
    cum_prob = 0.0
    n = 1
    while cum_prob < u
        pmf_val = pmf(dph, n)
        cum_prob += pmf_val
        cum_prob >= u && break
        n += 1
    end
    return n
end

# =============================================================================
# DISCRETE HAZARD (infection-age / Boxcar structure)
# =============================================================================

"""
    discrete_hazard(d::DPH, K::Integer; truncation = :absorb) -> Vector

Removal probability per age class, `h[1:K]`, of a Boxcar (infection-age) structure with `K`
classes whose dwell time follows `d`. The last class removes everyone (`h[K] = 1`) and every entry
is clamped to `[0, 1]`. Two truncations are available:

- `truncation = :absorb`: the exact hazard `h[k] = P(W = k) / P(W ≥ k)` for
  `k < K`, so the dwell time in the map is `min(W, K)`. Computed by iterating the state vector;
  where `P(W ≥ k)` falls below `1e-15` the hazard is set to `0.01` (no mass is left at those
  ages).
- `truncation = :condition`: the hazard of `W` conditioned on
  `W ≤ K`, `h[k] = P(W = k) / P(k ≤ W ≤ K)`. Geometric, negative binomial and delayed negative
  binomial DPHs use the backward recursion `h[k] = h[k+1] / (h[k+1] + P(W = k+1) / P(W = k))` from
  `h[K] = 1`. The PMF ratio is rational in `p`, so no logs or exponentials are needed, and values
  and ForwardDiff derivatives stay finite at every age. Other DPHs accumulate the truncated
  survival on the log scale from age `K` down, from `log(pmf)`; ages where it underflows get
  `h = 0`.

The two coincide when `P(W > K)` is negligible.
"""
function discrete_hazard(d::DPH, K::Integer; truncation::Symbol = :absorb)
    K ≥ 1 || throw(ArgumentError("number of age classes K must be ≥ 1; got $K"))
    if truncation === :absorb
        return _hazard_iterate(d, K)
    elseif truncation === :condition
        st = _shifted_negbin_structure(d)
        st === nothing || return _hazard_negbin(st..., K)
        T = eltype(d.α)
        return _hazard_from_logpmf([log(max(pmf(d, k), zero(T))) for k in 1:K])
    else
        throw(ArgumentError("truncation must be :absorb or :condition; got :$truncation"))
    end
end

# h[k] = P(W = k) / P(k ≤ W ≤ K) for the delayed NegBin (c delay steps, then r stages), by the
# backward recursion h[k] = h[k+1] / (h[k+1] + ρ_k) from h[K] = 1, where
# ρ_k = P(W = k+1) / P(W = k) = (1 - p) j / (j - r + 1) with j = k - c. Every h stays in [0, 1] and
# no logs or exponentials are needed, so values and ForwardDiff derivatives are finite at any age.
# Ages below c + r have no mass and get 0.
function _hazard_negbin(c, r, p, K)
    T = typeof(float(p))
    h = zeros(T, K)
    h[K] = one(T)
    q = one(T) - p
    for k in (K - 1):-1:max(1, c + r)
        j = k - c
        h[k] = h[k + 1] / (h[k + 1] + q * (j / (j - r + 1)))
    end
    return h
end

function _hazard_from_logpmf(lp::AbstractVector{T}) where {T}
    K   = length(lp)
    lsf = similar(lp)                    # log P(k ≤ W ≤ K), accumulated from the tail
    acc = T(-Inf)
    for k in K:-1:1
        acc = _logaddexp(acc, lp[k])
        lsf[k] = acc
    end
    h = [isfinite(lsf[k]) ? exp(lp[k] - lsf[k]) : zero(T) for k in 1:K]
    h[end] = one(T)
    return clamp.(h, zero(T), one(T))
end

function _hazard_iterate(d::DPH{T}, K) where {T}
    h = Vector{T}(undef, K)
    x = copy(d.α)          # state probability vector at infection age 1
    for k in 1:K
        sf_val  = sum(x)
        pmf_val = dot(x, d.s)
        h[k]    = (sf_val > T(1e-15) && isfinite(sf_val)) ? pmf_val / sf_val : T(0.01)
        x       = d.S' * x   # advance to the next infection age
    end
    h[end] = one(T)
    return clamp.(h, zero(T), one(T))
end
