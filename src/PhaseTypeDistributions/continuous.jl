# =============================================================================
# CONTINUOUS PHASE-TYPE DISTRIBUTIONS
# =============================================================================

"""
    PhaseType{T<:Real} <: ContinuousUnivariateDistribution

A continuous phase-type distribution representing the time to absorption in a
finite-state continuous-time Markov chain.

# Fields
- `α::Vector{T}`: Initial probability distribution over transient states (sums to 1)
- `A::Matrix{T}`: k×k generator matrix for transient states (A[i,j] ≥ 0 for i ≠ j, A[i,i] ≤ 0)
- `q::Vector{T}`: Absorption rates q = -A*1 (computed automatically, all elements ≥ 0)

# Example
```julia
# 3-phase Erlang distribution
α = [1.0, 0.0, 0.0]
A = [-2.0  2.0  0.0;
      0.0 -2.0  2.0;
      0.0  0.0 -2.0]
ph = PhaseType(α, A)
```
"""
struct PhaseType{T<:Real} <: ContinuousUnivariateDistribution
    α::Vector{T}  # Initial distribution over transient states
    A::Matrix{T}  # Generator matrix for transient states
    q::Vector{T}  # Absorption rates (computed automatically)
end

"""
    PhaseType(α::AbstractVector, A::AbstractMatrix)

Construct a continuous phase-type distribution from the initial distribution `α` (normalised
to sum to 1) and the transient generator `A`. The element type is the promotion of those of
`α` and `A`.

Throws `DimensionMismatch` if `A` is not `length(α) × length(α)`, and `ArgumentError` if `α` has
a negative entry or zero sum, or if an absorption rate `-A*1` is negative.

# Example
```julia
α = [0.5, 0.3, 0.2]
A = [-1.0  0.5  0.0;
      0.0 -2.0  1.0;
      0.0  0.0 -1.5]
ph = PhaseType(α, A)
```
"""
function PhaseType(α::AbstractVector{Ta}, A::AbstractMatrix{Tb}) where {Ta<:Real, Tb<:Real}
    Tc = promote_type(Ta, Tb)
    αp = Vector{Tc}(α)
    Ap = Matrix{Tc}(A)
    k = length(αp)
    size(Ap, 1) == k == size(Ap, 2) ||
        throw(DimensionMismatch("generator matrix A must be k×k with k = length(α) = $k; got $(size(Ap))"))
    all(αp .>= zero(Tc)) || throw(ArgumentError("initial distribution α must be elementwise ≥ 0"))
    s = sum(αp)
    s > zero(Tc) || throw(ArgumentError("initial distribution α must have positive sum"))
    αp ./= s
    qp = -Ap * ones(Tc, k)
    all(qp .>= zero(Tc)) || throw(ArgumentError("absorption rates q = -A*1 must be ≥ 0"))
    return PhaseType{Tc}(αp, Ap, qp)
end

# =============================================================================
# CONTINUOUS PHASE-TYPE: CORE COMPUTATIONS
# =============================================================================

"""
    _ph_pdf(t, α, A, q)

Density `f(t) = α' exp(A t) q` of a continuous phase-type distribution. Erlang distributions
(Erlang generator and `α = e₁`) use the closed form; otherwise the matrix exponential is used,
which needs a floating-point (BLAS) element type: Dual numbers are only supported on the
closed-form path. Errors are raised, not returned as a zero density.
"""
function _ph_pdf(t, α, A, q)
    if _is_erlang_matrix(A) && _starts_in_first_phase(α)
        return _erlang_pdf_exact(t, α, A)
    end
    At = A * t
    eltype(At) <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "the density of a non-Erlang PhaseType needs exp(A t), available for floating-point element " *
        "types only; got $(eltype(At))"))
    result = (α' * exp(At) * q)[]
    return max(result, zero(result))
end

"""
    _is_erlang_matrix(A; atol=DEFAULT_TOLERANCE)

Check whether the generator `A` is an Erlang generator: `-r` on the diagonal, `r` on the
superdiagonal and zero elsewhere, for some `r > 0`.
"""
function _is_erlang_matrix(A; atol=DEFAULT_TOLERANCE)
    k = size(A, 1)
    r = -A[1, 1]
    r > 0 || return false
    for i in 1:k
        abs(A[i, i] + r) ≤ atol || return false
        if i < k
            abs(A[i, i+1] - r) ≤ atol || return false
        end
        for j in 1:k
            if j != i && j != i+1
                abs(A[i, j]) ≤ atol || return false
            end
        end
    end
    return true
end

"""Whether `α` is the unit vector `e₁` (the Erlang initial distribution), within `atol`."""
_starts_in_first_phase(α; atol=DEFAULT_TOLERANCE) =
    abs(α[1] - 1) ≤ atol && all(x -> abs(x) ≤ atol, @view α[2:end])

"""
    _erlang_pdf_exact(t, α, A)

Closed-form Erlang density `r^k t^(k-1) exp(-r t) / (k-1)!`, evaluated on the log scale.
Requires an Erlang generator `A` and `α = e₁` (checked by the caller).
"""
function _erlang_pdf_exact(t, α, A)
    k = length(α)
    r = -A[1, 1]
    t > 0 || return zero(float(r * t))
    log_pdf = k * log(r) + (k-1) * log(t) - r * t - loggamma(k)
    return exp(log_pdf)
end

"""
    _ph_surv(t, α, A, k)

Survival function `S(t) = α' exp(A t) 1` of a continuous phase-type distribution.
"""
_ph_surv(t, α, A, k) = (α' * exp(A * t) * ones(eltype(α), k))[]

# =============================================================================
# CONTINUOUS PHASE-TYPE: DISTRIBUTIONS.JL INTERFACE
# =============================================================================

insupport(d::PhaseType, x::Real) = isfinite(x) && x > 0

# evaluation point in the promoted element type (a Dual x keeps its partials)
_point(d::PhaseType, x::Real) = convert(promote_type(eltype(d.α), typeof(x)), x)

function pdf(d::PhaseType, x::Real)
    x ≤ 0 && return zero(eltype(d.α))
    return _ph_pdf(_point(d, x), d.α, d.A, d.q)
end

function logpdf(d::PhaseType, x::Real)
    x ≤ 0 && return -Inf
    v = _ph_pdf(_point(d, x), d.α, d.A, d.q)
    return (isfinite(v) && v > 0) ? log(v) : -Inf
end

function cdf(d::PhaseType, x::Real)
    x ≤ 0 && return zero(eltype(d.α))
    k = length(d.α)
    S = _ph_surv(_point(d, x), d.α, d.A, k)
    return one(eltype(d.α)) - S
end

function logcdf(d::PhaseType, x::Real)
    x ≤ 0 && return -Inf
    k = length(d.α)
    S = _ph_surv(_point(d, x), d.α, d.A, k)
    T = one(eltype(d.α)) - S
    return log(T > 0 ? T : eps(eltype(d.α)))
end

function mean(d::PhaseType)
    k = length(d.α)
    m = -d.A \ ones(eltype(d.α), k)
    return dot(d.α, m)
end

function var(d::PhaseType)
    k = length(d.α)
    T1 = -d.A \ ones(eltype(d.α), k)
    T2 = -d.A \ (2 .* T1)  # E[T^2] = 2 α' (-A)^{-2} 1
    μ = dot(d.α, T1)
    eT2 = dot(d.α, T2)
    return eT2 - μ^2
end

Base.minimum(::PhaseType) = 0.0
Base.maximum(::PhaseType) = Inf

# =============================================================================
# CONTINUOUS PHASE-TYPE: CONSTRUCTOR FUNCTIONS
# =============================================================================

"""
    ph_exponential(λ::Real)

Exponential distribution with rate `λ > 0` as a 1-phase phase-type distribution.

# Example
```julia
exp_dist = ph_exponential(2.0)  # rate 2
mean(exp_dist)                  # 0.5
```
"""
function ph_exponential(λ::Real)
    T = float(λ)
    T > 0 || throw(ArgumentError("rate λ must be > 0; got $λ"))
    α = [one(T)]
    A = reshape([-T], 1, 1)
    return PhaseType(α, A)
end

"""
    ph_erlang(k::Integer, r::Real)

Erlang distribution with `k ≥ 1` phases and rate `r > 0` per phase, as a k-phase
phase-type distribution (mean `k/r`).

# Example
```julia
erlang_dist = ph_erlang(3, 1.5)  # 3 phases, rate 1.5
mean(erlang_dist)                # 2.0
```
"""
function ph_erlang(k::Integer, r::Real)
    k ≥ 1 || throw(ArgumentError("number of phases k must be ≥ 1; got $k"))
    T = float(r)
    T > 0 || throw(ArgumentError("rate r must be > 0; got $r"))
    α = vcat(one(T), zeros(typeof(T), k-1))
    A = zeros(typeof(T), k, k)
    @inbounds for i in 1:k
        A[i, i] = -T
        if i < k
            A[i, i+1] = T
        end
    end
    return PhaseType(α, A)
end

"""
    ph_erlang_by_mean(k::Integer, μ::Real)

Erlang distribution with `k ≥ 1` phases and mean `μ > 0` (rate `k/μ` per phase).

# Example
```julia
erlang_dist = ph_erlang_by_mean(4, 2.0)  # 4 phases, mean 2.0
mean(erlang_dist)                        # 2.0
```
"""
function ph_erlang_by_mean(k::Integer, μ::Real)
    T = float(μ)
    (k ≥ 1 && T > 0) || throw(ArgumentError("k must be ≥ 1 and the mean μ must be > 0; got k = $k, μ = $μ"))
    r = k / T
    return ph_erlang(k, r)
end
