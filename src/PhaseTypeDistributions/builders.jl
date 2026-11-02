# =============================================================================
# DISCRETE PHASE-TYPE: CONSTRUCTOR FUNCTIONS
# =============================================================================

"""
    dph_geometric(p::Real)

Geometric distribution on `{1, 2, …}` with per-step absorption probability `p ∈ (0, 1]`,
as a 1-phase DPH (`α = [1]`, `S = [1 - p]`).

# Example
```julia
geom_dist = dph_geometric(0.3)
mean(geom_dist)  # 1/0.3 ≈ 3.33
```
"""
function dph_geometric(p::Real)
    0 < p ≤ 1 || throw(ArgumentError("success probability p must be in (0, 1]; got $p"))
    α = [one(p)]
    S = reshape([one(p) - p], 1, 1)
    return DPH(α, S)
end

"""
    dph_negative_binomial(r::Integer, p::Real)

Negative binomial distribution counting the TOTAL number of trials to the `r`-th success
(support `{r, r+1, …}`, mean `r/p`), as an `r`-stage DPH: each stage is left with probability
`p` per step. `r ≥ 1`, `p ∈ (0, 1]`. A Dual `p` keeps its partials, so gradients flow through `p`.

A method for a ForwardDiff `Dual` `r` rounds `r` to the nearest integer, so the derivative with respect to `r` is zero (use
`dph_negative_binomial_smooth` for a differentiable number of stages).

# Example
```julia
nb = dph_negative_binomial(4, 0.6)
mean(nb)   # 4/0.6 ≈ 6.67
```
"""
function dph_negative_binomial(r::Integer, p::Real)
    r ≥ 1 || throw(ArgumentError("number of successes r must be ≥ 1; got $r"))
    0 < p ≤ 1 || throw(ArgumentError("success probability p must be in (0, 1]; got $p"))
    k = r
    α = vcat(one(p), zeros(typeof(p), k-1))
    S = zeros(typeof(p), k, k)
    for i in 1:k-1
        S[i, i] = 1 - p      # stay in state i (failure)
        S[i, i+1] = p        # move to state i+1 (success)
    end
    S[k, k] = 1 - p          # final state: stay with prob 1-p, absorb with prob p
    return DPH(α, S)
end

function dph_negative_binomial(r::ForwardDiff.Dual, p::Real)
    return dph_negative_binomial(round(Int, ForwardDiff.value(r)), p)
end

"""
    dph_prepend_deterministic_delay(dph::DPH, c::Integer)

Return a new `DPH` that prepends `c ≥ 0` **deterministic** transient states in front of the
chain described by `dph`. Each prepended state advances to the next with probability 1 on the
next step (no absorption from those states); after `c` steps the process enters the original
states according to `dph.α` and then evolves with `dph.S`. The time to absorption increases by
exactly `c` steps: `E[W_new] = c + E[W_core]`. `c = 0` returns `dph` unchanged.

Use this when a dwell time has a known minimum (e.g. fixed clinical offsets) while the
stochastic tail remains phase-type.

# Example
```julia
core    = dph_negative_binomial(3, 0.25)         # minimum 3
delayed = dph_prepend_deterministic_delay(core, 2) # minimum 5
```
"""
function dph_prepend_deterministic_delay(dph::DPH, c::Integer)
    c ≥ 0 || throw(ArgumentError("c (deterministic delay steps) must be ≥ 0; got $c"))
    iszero(c) && return dph
    kr = length(dph.α)
    Te = promote_type(eltype(dph.α), eltype(dph.S))
    k = c + kr
    S = zeros(Te, k, k)
    for i in 1:(c - 1)
        S[i, i + 1] = one(Te)
    end
    for j in 1:kr
        S[c, c + j] = Te(dph.α[j])
    end
    for i in 1:kr, j in 1:kr
        S[c + i, c + j] = Te(dph.S[i, j])
    end
    α = zeros(Te, k)
    α[1] = one(Te)
    return DPH(α, S)
end

"""
    dph_negative_binomial_min_support(r::Integer, p, δ_min::Integer)

Negative binomial DPH with a known minimum `δ_min ≥ r`:
`dph_prepend_deterministic_delay(dph_negative_binomial(r, p), δ_min - r)`. Its support is
`{δ_min, δ_min + 1, …}` and its mean is `(δ_min - r) + r/p` (e.g. the Neal & Roberts infectious
period of the Hagelloch example, whose recorded intervals include a fixed offset).

As for `dph_negative_binomial`, a ForwardDiff `Dual` `r` is rounded to the nearest integer.
"""
function dph_negative_binomial_min_support(r::Integer, p, δ_min::Integer)
    δ_min ≥ r || throw(ArgumentError("δ_min ($δ_min) must be ≥ r ($r)"))
    return dph_prepend_deterministic_delay(dph_negative_binomial(r, p), δ_min - r)
end

function dph_negative_binomial_min_support(r::ForwardDiff.Dual, p::Real, δ_min::Integer)
    return dph_negative_binomial_min_support(round(Int, ForwardDiff.value(r)), p, δ_min)
end

"""
    dph_erlang(k::Integer, μ::Real, Δt::Real=1.0)

Discretised Erlang distribution: the continuous Erlang with `k` phases and continuous mean `μ`
(rate `k/μ` per phase), discretised on steps of length `Δt` with `dph_from_rates`. Identical to
`dph_erlang_from_mean(k, μ, Δt)`. Note that `μ` is the mean of the continuous distribution, not of
the resulting DPH (whose mean, in steps, is `k / (1 - exp(-k Δt / μ))`).

# Example
```julia
d = dph_erlang(3, 2.0, 0.1)   # 3 phases, continuous mean 2.0, step 0.1
```
"""
function dph_erlang(k::Integer, μ::Real, Δt::Real=1.0)
    k ≥ 1 || throw(ArgumentError("number of phases k must be ≥ 1; got $k"))
    μ > 0 || throw(ArgumentError("mean μ must be > 0; got $μ"))
    Δt > 0 || throw(ArgumentError("time step Δt must be > 0; got $Δt"))
    # scalar per-entry discretisation (avoids exp(A*Δt) on a matrix of Dual numbers)
    r = k / μ
    return dph_erlang_from_rates(k, r, Δt)
end

# =============================================================================
# DISCRETE PHASE-TYPE: CONVERSION FROM CONTINUOUS RATES
# =============================================================================

"""
    rate_to_proportion(r, t=1.0)

Probability `1 - exp(-r t)` of at least one event of a rate-`r` process in time `t`, guarded:
`0` for negative or `NaN` rates, `1` for an infinite rate, and clamped to `[0, 1]`.
"""
@inline function rate_to_proportion(r, t=1.0)
    if r < 0 || isnan(r)
        return 0.0
    elseif isinf(r)
        return 1.0
    else
        result = 1 - exp(-r * t)
        return isnan(result) ? 0.0 : max(min(result, 1.0), 0.0)
    end
end

"""
    dph_from_rates(α, A, Δt=1.0)

Discretise a continuous-time phase-type chain (initial distribution `α`, normalised; generator
`A`) on steps of length `Δt`, allowing at most one jump per step: state `i` is kept with
probability `exp(-q_i Δt)` (`q_i = -A[i,i]`) and left for state `j` with probability
`(A[i,j] / q_i) (1 - exp(-q_i Δt))`. This is an approximation of `exp(A Δt)`, exact for one-state
chains. The entries are computed one scalar at a time, so Dual numbers are supported.

# Example
```julia
d = dph_from_rates([1.0, 0.0], [-2.0 1.0; 0.0 -1.0], 0.1)
```
"""
function dph_from_rates(α::AbstractVector{Ta}, A::AbstractMatrix{Tb}, Δt::Real=1.0) where {Ta<:Real, Tb<:Real}
    Tc = promote_type(Ta, Tb, typeof(Δt))
    αp = Vector{Tc}(α)
    Ap = Matrix{Tc}(A)
    k = length(αp)
    size(Ap, 1) == k == size(Ap, 2) ||
        throw(DimensionMismatch("generator matrix A must be k×k with k = length(α) = $k; got $(size(Ap))"))
    all(αp .>= zero(Tc)) || throw(ArgumentError("initial distribution α must be elementwise ≥ 0"))
    s = sum(αp)
    s > zero(Tc) || throw(ArgumentError("initial distribution α must have positive sum"))
    αp ./= s
    S = zeros(Tc, k, k)
    for i in 1:k
        total_exit_rate = -A[i, i]
        S[i, i] = exp(-total_exit_rate * Δt)
        for j in 1:k
            if i != j && A[i, j] > 0
                S[i, j] = (A[i, j] / total_exit_rate) * (1 - exp(-total_exit_rate * Δt))
            end
        end
    end
    return DPH(αp, S)
end

"""
    dph_exponential(λ, Δt=1.0)

Geometric DPH obtained by discretising an exponential distribution with rate `λ > 0` on steps of
length `Δt` (per-step absorption probability `1 - exp(-λ Δt)`).

# Example
```julia
d = dph_exponential(2.0, 0.1)
```
"""
function dph_exponential(λ::Real, Δt::Real=1.0)
    λ > 0 || throw(ArgumentError("rate λ must be > 0; got $λ"))
    T = promote_type(typeof(λ), typeof(Δt))
    α = [one(T)]
    A = reshape([-T(λ)], 1, 1)
    return dph_from_rates(α, A, Δt)
end

"""
    dph_erlang_from_rates(k, r, Δt=1.0)

Discretised Erlang distribution with `k ≥ 1` phases and rate `r > 0` per phase, on steps of length
`Δt` (via `dph_from_rates`). The result is a negative binomial DPH with progression probability
`1 - exp(-r Δt)`.

# Example
```julia
d = dph_erlang_from_rates(3, 1.5, 0.1)
```
"""
function dph_erlang_from_rates(k::Integer, r::Real, Δt::Real=1.0)
    k ≥ 1 || throw(ArgumentError("number of phases k must be ≥ 1; got $k"))
    T = promote_type(typeof(r), typeof(Δt))
    T(r) > 0 || throw(ArgumentError("rate r must be > 0; got $r"))
    α = vcat(one(T), zeros(T, k-1))
    A = zeros(T, k, k)
    for i in 1:k
        A[i, i] = -T(r)
        if i < k
            A[i, i+1] = T(r)
        end
    end
    return dph_from_rates(α, A, Δt)
end

"""
    dph_erlang_from_mean(k, μ, Δt=1.0)

Discretised Erlang distribution with `k` phases and **continuous** mean `μ` (rate `k/μ` per
phase), on steps of length `Δt`. The mean of the resulting DPH is `k / (1 - exp(-k Δt / μ))`
steps, not `μ` (e.g. `k = 4`, `μ = 4`, `Δt = 1` gives 6.33 steps).

# Example
```julia
d = dph_erlang_from_mean(4, 4.0, 1.0)
mean(d)   # 6.328
```
"""
function dph_erlang_from_mean(k::Integer, μ::Real, Δt::Real=1.0)
    k ≥ 1 || throw(ArgumentError("number of phases k must be ≥ 1; got $k"))
    T = promote_type(typeof(μ), typeof(Δt))
    T(μ) > 0 || throw(ArgumentError("mean μ must be > 0; got $μ"))
    r = k / μ
    return dph_erlang_from_rates(k, r, Δt)
end

"""
    dph_negative_binomial_from_rates(r, p, Δt=1.0)

Discretised `r`-stage chain whose stages have the continuous rate `λ = -log(1 - p)` (so that a
unit step has progression probability `p`), on steps of length `Δt` (via `dph_from_rates`).
Requires `r ≥ 1` and `p ∈ (0, 1)`. With `Δt = 1` it equals `dph_negative_binomial(r, p)` up to
rounding.

# Example
```julia
d = dph_negative_binomial_from_rates(5, 0.4, 0.1)
```
"""
function dph_negative_binomial_from_rates(r::Integer, p::Real, Δt::Real=1.0)
    r ≥ 1 || throw(ArgumentError("number of successes r must be ≥ 1; got $r"))
    T = promote_type(typeof(p), typeof(Δt))
    (T(p) > 0 && T(p) < 1) || throw(ArgumentError("success probability p must be in (0, 1); got $p"))
    λ = -log(1 - T(p))
    A = zeros(T, r, r)
    for i in 1:r
        A[i, i] = -λ
        if i < r
            A[i, i+1] = λ
        end
    end
    α = vcat(one(T), zeros(T, r-1))
    return dph_from_rates(α, A, Δt)
end

"""
    dph_negative_binomial_from_mean(r, μ)

Negative binomial DPH (total-trials convention, as `dph_negative_binomial`) with `r ≥ 1` stages
and mean `μ ≥ r` steps, i.e. progression probability `p = r/μ`.

# Example
```julia
d = dph_negative_binomial_from_mean(5, 12.5)  # p = 0.4
```
"""
function dph_negative_binomial_from_mean(r::Integer, μ::Real)
    r ≥ 1 || throw(ArgumentError("number of successes r must be ≥ 1; got $r"))
    T = typeof(float(μ))
    T(μ) > 0 || throw(ArgumentError("mean μ must be > 0; got $μ"))
    p = T(r) / T(μ)
    0 < p ≤ 1 || throw(ArgumentError("mean μ must be ≥ r; got μ = $μ with r = $r"))
    return dph_negative_binomial(r, p)
end
