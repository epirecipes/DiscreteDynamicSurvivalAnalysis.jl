# =============================================================================
# SMOOTH INTERPOLATION FOR DISCRETE PHASE-TYPE DISTRIBUTIONS
# =============================================================================

"""
    SmoothDPH{T, W} <: DiscreteUnivariateDistribution

Mixture `(1 - w) DPH_k + w DPH_{k+1}` of two neighbouring negative binomial DPHs, used to give
the number of stages a real-valued parameterisation (see `dph_negative_binomial_smooth`).

# Fields
- `dph_k::DPH{T}`: negative binomial DPH with `k` stages
- `dph_k1::DPH{T}`: negative binomial DPH with `k + 1` stages
- `w::W`: interpolation weight in `[0, 1)`
"""
struct SmoothDPH{T, W} <: DiscreteUnivariateDistribution
    dph_k::DPH{T}   # DPH with k phases
    dph_k1::DPH{T}  # DPH with k+1 phases
    w::W            # Interpolation weight
end

"""
    dph_negative_binomial_smooth(r::Real, p::Real)

Negative binomial DPH with a real-valued number of stages `r > 0`: for `r = k + w` (clamped to
`[1, MAX_PHASES]`, `k = ⌊r⌋`, `w ∈ [0, 1)`) the mixture `(1 - w) DPH_k + w DPH_{k+1}` of
`dph_negative_binomial(k, p)` and `dph_negative_binomial(k + 1, p)`. When `r` is a ForwardDiff
`Dual`, `w` keeps its partials, so the mixture is differentiable in `r` (∂w/∂r = 1 inside the
clamp). `p ∈ (0, 1]`.

# Example
```julia
smooth_nb = dph_negative_binomial_smooth(3.7, 0.4)
mean(smooth_nb)   # 0.3 * 3/0.4 + 0.7 * 4/0.4
```
"""
function dph_negative_binomial_smooth(r::Real, p::Real)
    r_val = ForwardDiff.value(r)
    p_val = ForwardDiff.value(p)
    r_val > 0 || throw(ArgumentError("number of successes r must be positive; got $r_val"))
    0 < p_val ≤ 1 || throw(ArgumentError("success probability p must be in (0, 1]; got $p_val"))
    # clamp the primal r; w uses clamp(r, …) so that ∂w/∂r follows the clamp at its bounds
    r_val_clamped = clamp(r_val, 1.0, MAX_PHASES)
    k = floor(Int, r_val_clamped)
    w = clamp(r, 1, MAX_PHASES) - k
    k = max(1, k)
    if k >= MAX_PHASES
        k = MAX_PHASES
        w = zero(w)
    end
    dph_k = dph_negative_binomial(k, p)
    dph_k1 = dph_negative_binomial(k+1, p)
    T_eltype = eltype(dph_k.α)
    w_typed = convert(promote_type(T_eltype, typeof(w)), w)
    return SmoothDPH(dph_k, dph_k1, w_typed)
end

# =============================================================================
# SMOOTH DPH: DISTRIBUTION INTERFACE
# =============================================================================

"""
    pmf(d::SmoothDPH, x::Integer)

`(1 - w) pmf_k(x) + w pmf_{k+1}(x)`.
"""
function pmf(dph::SmoothDPH, x::Integer)
    pmf_k = pmf(dph.dph_k, x)
    pmf_k1 = pmf(dph.dph_k1, x)
    return (1 - dph.w) * pmf_k + dph.w * pmf_k1
end

pdf(dph::SmoothDPH, x::Integer) = pmf(dph, x)

"""Log-PMF of the mixture, `log(max(pmf, 1e-12))`."""
function logpdf(dph::SmoothDPH, x::Integer)
    pmf_val = pmf(dph, x)
    return log(max(pmf_val, 1e-12))
end

cdf(dph::SmoothDPH, x::Integer) = (1 - dph.w) * cdf(dph.dph_k, x) + dph.w * cdf(dph.dph_k1, x)

function mean(dph::SmoothDPH)
    mean_k = mean(dph.dph_k)
    mean_k1 = mean(dph.dph_k1)
    return (1 - dph.w) * mean_k + dph.w * mean_k1
end

function var(dph::SmoothDPH)
    var_k = var(dph.dph_k)
    var_k1 = var(dph.dph_k1)
    mean_k = mean(dph.dph_k)
    mean_k1 = mean(dph.dph_k1)
    # mixture: Var = E[X²] - E[X]²
    mean_mix = (1 - dph.w) * mean_k + dph.w * mean_k1
    second_moment_k = var_k + mean_k^2
    second_moment_k1 = var_k1 + mean_k1^2
    second_moment_mix = (1 - dph.w) * second_moment_k + dph.w * second_moment_k1
    return second_moment_mix - mean_mix^2
end

"""
    rand(rng::AbstractRNG, d::SmoothDPH)

Draw from the `k + 1`-stage component with probability `w`, otherwise from the `k`-stage one.
"""
function Base.rand(rng::AbstractRNG, dph::SmoothDPH)
    if rand(rng) < ForwardDiff.value(dph.w)
        return rand(rng, dph.dph_k1)
    else
        return rand(rng, dph.dph_k)
    end
end

Base.eltype(::Type{<:SmoothDPH}) = Int
Base.eltype(::SmoothDPH) = Int
support(dph::SmoothDPH) = support(dph.dph_k)
Base.minimum(dph::SmoothDPH) = minimum(dph.dph_k)
Base.maximum(dph::SmoothDPH) = maximum(dph.dph_k1)
insupport(dph::SmoothDPH, x::Real) = insupport(dph.dph_k, x) || insupport(dph.dph_k1, x)
