# Discrete-time SIR forward maps (the deterministic part of DDSA).

"""
    InfectionStructure

How infected individuals progress to removal in the forward map. Subtypes: [`GLCT`](@ref)
(phase-type stages) and [`Boxcar`](@ref) (infection-age classes with a discrete hazard).
"""
abstract type InfectionStructure end

"""
    GLCT(dph::DPH)

Generalised linear chain trick structure: the infected population is a vector `x` over the
transient states of the discrete phase-type distribution `dph` (initial distribution `α`,
transition matrix `S`). Each step, `x ← S' x + α λ`, where `λ` is the number of new infections.
A geometric infectious period is the one-state case (`dph_geometric(p)`), which gives the classical
discrete-time SIR map.
"""
struct GLCT{D<:DPH} <: InfectionStructure
    dph::D
end

"""
    Boxcar(hazard::AbstractVector)
    Boxcar(dph::DPH, K_age::Integer; truncation = :absorb)

Infection-age (Boxcar) structure with `K_age = length(hazard)` age classes: each step, a fraction
`hazard[k]` of the infected of age `k` is removed and the rest move to age `k + 1`; new infections
enter age 1. The second form builds the hazard from a dwell-time distribution with
[`discrete_hazard`](@ref), whose `truncation` keyword selects how the distribution is truncated at
`K_age` classes (`:absorb` or `:condition`).
"""
struct Boxcar{V<:AbstractVector{<:Real}} <: InfectionStructure
    hazard::V
    function Boxcar(hazard::V) where {V<:AbstractVector{<:Real}}
        isempty(hazard) && throw(ArgumentError("the hazard vector must have at least one age class"))
        return new{V}(hazard)
    end
end
Boxcar(dph::DPH, K_age::Integer; truncation::Symbol = :absorb) =
    Boxcar(discrete_hazard(dph, K_age; truncation = truncation))

_eltype(s::GLCT) = eltype(s.dph.S)
_eltype(s::Boxcar) = eltype(s.hazard)

"""
    MapSolution

Output of [`forward_map`](@ref).

# Fields
- `S`: susceptible fractions at steps `0, 1, …, nsteps`: a vector (one group) or an
  `(nsteps + 1) × G` matrix (`G` groups).
- `τ`: attack fraction(s) over the window: the new infections summed over the steps, which equals
  `S₀ − S_T` (scalar, or vector over groups).
- `f`: infection-time distribution(s): new infections per step normalised to sum to one (vector,
  or `nsteps × G` matrix). If a group has no infections, its `f` is uniform.

`τ` and `f` are computed from the new infections recorded at each step, not from differences of
`S`, so they keep full relative precision when the infections are tiny (e.g. `R0 < 1`).
- `Δt`: step length.

Accessors: [`attack_fraction`](@ref), [`infection_pmf`](@ref), [`susceptible_fraction`](@ref).
"""
struct MapSolution{T, AS<:AbstractArray{T}, AT, AF<:AbstractArray}
    S::AS
    τ::AT
    f::AF
    Δt::Float64
end

ngroups(sol::MapSolution) = size(sol.S, 2)
nsteps(sol::MapSolution) = size(sol.S, 1) - 1

"""
    attack_fraction(sol::MapSolution, g = 1)

Attack fraction `τ = S₀ − S_T` of group `g`: the expected fraction of the group infected during the
window, which enters the final-size term `K ~ Binomial(N, τ)`.
"""
attack_fraction(sol::MapSolution, g::Integer = 1) = sol.τ[g]

"""
    infection_pmf(sol::MapSolution, g = 1)

Infection-time distribution `f` of group `g` over the `nsteps` steps (sums to one).
"""
infection_pmf(sol::MapSolution, g::Integer = 1) = sol.f isa AbstractVector ? sol.f : view(sol.f, :, g)

"""
    susceptible_fraction(sol::MapSolution, g = 1)

Susceptible fraction of group `g` at steps `0, …, nsteps`.
"""
susceptible_fraction(sol::MapSolution, g::Integer = 1) = sol.S isa AbstractVector ? sol.S : view(sol.S, :, g)

# ---------------------------------------------------------------------------------------------
# Force of infection
# ---------------------------------------------------------------------------------------------

_rates(B::AbstractMatrix, I) = B * I
_rates(foi, I) = foi(I)

function _check_foi(foi, G)
    if foi isa AbstractMatrix
        size(foi) == (G, G) || throw(DimensionMismatch("transmission matrix must be $G×$G; got $(size(foi))"))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# In-place maps
# ---------------------------------------------------------------------------------------------

"""
    forward_map!(S_hist, structure, β, ρ; Δt = 1.0) -> S_hist
    forward_map!(S_hist, structure, foi, ρ::AbstractVector; Δt = 1.0) -> S_hist

Run the forward map and write the susceptible fractions at steps `0, …, nsteps` into `S_hist`
(a vector of length `nsteps + 1` for one group; an `(nsteps + 1) × G` matrix for `G` groups).
Only `S_hist` is mutated. See [`forward_map`](@ref) for the arguments; the element type of
`S_hist` must hold the promoted parameter types (e.g. ForwardDiff `Dual`s). [`forward_map`](@ref)
also records the new infections per step, from which it builds `τ` and `f`.
"""
function forward_map!(S_hist::AbstractVector, structure::InfectionStructure, β::Real, ρ::Real; Δt::Real = 1.0)
    _map!(reshape(S_hist, :, 1), structure, I -> β .* I, (ρ,), Float64(Δt))
    return S_hist
end

function forward_map!(S_hist::AbstractMatrix, structure::InfectionStructure, foi::F, ρ::AbstractVector;
                      Δt::Real = 1.0) where {F}
    G = length(ρ)
    size(S_hist, 2) == G || throw(DimensionMismatch("S_hist must have one column per group ($G); got $(size(S_hist, 2))"))
    _check_foi(foi, G)
    _map!(S_hist, structure, foi, ρ, Float64(Δt))
    return S_hist
end

# `_map!` writes S into `S_hist` and, unless `Λ` is `nothing`, the new infections λ of step t and
# group g into `Λ[t, g]`. `foi::F` forces specialisation on the force-of-infection callable, and the
# state vectors are built with typed comprehensions so their type is known to the compiler.

# GLCT: x_g ← S' x_g + α λ_g, with λ_g = (1 − exp(−rate_g Δt)) S_g
function _map!(S_hist, s::GLCT, foi::F, ρ, Δt, Λ = nothing) where {F}
    T = eltype(S_hist)
    Tm, α = s.dph.S, s.dph.α
    k = length(α)
    G = length(ρ)
    nst = size(S_hist, 1) - 1
    S = T[one(T) - ρ[g] for g in 1:G]
    x = Vector{T}[α .* T(ρ[g]) for g in 1:G]
    y = Vector{T}[similar(x[g]) for g in 1:G]
    I = Vector{T}(undef, G)
    @views S_hist[1, :] .= S
    for t in 1:nst
        for g in 1:G
            I[g] = sum(x[g])
        end
        r = _rates(foi, I)
        for g in 1:G
            λ = rate_to_proportion(r[g], Δt) * S[g]
            xg, yg = x[g], y[g]
            @inbounds for j in 1:k        # y = S' x + α λ, summing in the order of S' * x
                a = Tm[1, j] * xg[1]
                for i in 2:k
                    a += Tm[i, j] * xg[i]
                end
                yg[j] = a + α[j] * λ
            end
            x[g], y[g] = yg, xg
            S[g] -= λ
            S_hist[t + 1, g] = S[g]
            Λ === nothing || (Λ[t, g] = λ)
        end
    end
    return S_hist
end

# Boxcar: ages shift by one step with survival 1 − h[k]; new infections enter age 1. At step t
# the infected occupy ages 1:t, so only the first min(K_age, nsteps) classes are ever counted;
# the older ones stay empty and are not stored (same result, less work when K_age > nsteps).
function _map!(S_hist, s::Boxcar, foi::F, ρ, Δt, Λ = nothing) where {F}
    T = eltype(S_hist)
    h = s.hazard
    G = length(ρ)
    nst = size(S_hist, 1) - 1
    K = min(length(h), max(nst, 1))
    surv = T[one(T) - h[a] for a in 1:K]
    S = T[one(T) - ρ[g] for g in 1:G]
    x = Vector{T}[zeros(T, K) for _ in 1:G]
    for g in 1:G
        x[g][1] = ρ[g]                     # initial infectives enter age class 1
    end
    I = Vector{T}(undef, G)
    @views S_hist[1, :] .= S
    for t in 1:nst
        for g in 1:G
            I[g] = sum(x[g])
        end
        r = _rates(foi, I)
        for g in 1:G
            λ = rate_to_proportion(r[g], Δt) * S[g]
            xg = x[g]
            @inbounds for a in (K - 1):-1:1   # shift ages from the oldest down; age K is removed
                xg[a + 1] = xg[a] * surv[a]
            end
            xg[1] = λ
            S[g] = max(zero(T), S[g] - λ)
            S_hist[t + 1, g] = S[g]
            Λ === nothing || (Λ[t, g] = λ)
        end
    end
    return S_hist
end

# ---------------------------------------------------------------------------------------------
# Allocating maps
# ---------------------------------------------------------------------------------------------

"""Attack fraction and normalised infection-time distribution from the new infections per step
(`f` is uniform if there are none)."""
function _attack_and_pmf(λ::AbstractVector{T}) where {T}
    n = length(λ)
    f = T[max(λ[t], zero(T)) for t in 1:n]
    τ = sum(f)
    return τ, (τ > 0 ? f ./ τ : fill(one(T) / n, n))
end

"""
    forward_map(structure, β, ρ, nsteps; Δt = 1.0) -> MapSolution
    forward_map(structure, foi, ρ::AbstractVector, nsteps; Δt = 1.0) -> MapSolution

Run the deterministic discrete-time SIR map of DDSA for `nsteps` steps of length `Δt`.

**One group.** `β ≥ 0` is the transmission rate per unit time and `ρ ∈ [0, 1)` the initially
infected fraction: `S₀ = 1 − ρ`, and the initial infectives enter the first state (`GLCT`) or age
class (`Boxcar`). Each step, the new infections are `λ = (1 − exp(−β I Δt)) S` with `I` the
infected fraction, so `S` decreases monotonically.

**`G` groups.** `ρ` holds the initially infected fraction of each group (fractions of each group's
own size) and `foi` gives the force of infection on every group from the vector `I` of infected
fractions:
- a `G × G` matrix `B`: `rates = B * I` (`B[g, h]` is the transmission rate from group `h` to `g`);
- any callable `foi(I)` returning `G` rates (e.g. a closure over parameters), which must accept
  vectors of ForwardDiff `Dual`s when used for inference.
Each group then evolves as in the one-group map with its own rate.

Returns a [`MapSolution`](@ref) with the susceptible fractions, the attack fraction(s)
`τ = S₀ − S_T` and the normalised infection-time distribution(s) `f`, both computed from the new
infections recorded at each step. The element type follows the parameters, so the map is
ForwardDiff-compatible.

# Example
```julia
sol = forward_map(GLCT(dph_negative_binomial(4, 0.6)), 0.5, 0.01, 40)
attack_fraction(sol), infection_pmf(sol)
# two groups with a transmission matrix
sol2 = forward_map(GLCT(dph_geometric(0.2)), [0.4 0.1; 0.1 0.3], [0.01, 0.0], 60)
```
"""
function forward_map(structure::InfectionStructure, β::Real, ρ::Real, nsteps::Integer; Δt::Real = 1.0)
    _check_map_args(nsteps, Δt, (ρ,))
    T = promote_type(_eltype(structure), typeof(β), typeof(ρ))
    S_hist = Vector{T}(undef, nsteps + 1)
    Λ = Matrix{T}(undef, nsteps, 1)
    _map!(reshape(S_hist, :, 1), structure, I -> β .* I, (ρ,), Float64(Δt), Λ)
    τ, f = _attack_and_pmf(view(Λ, :, 1))
    return MapSolution(S_hist, τ, f, Float64(Δt))
end

function forward_map(structure::InfectionStructure, foi::F, ρ::AbstractVector{<:Real}, nsteps::Integer;
                     Δt::Real = 1.0) where {F}
    _check_map_args(nsteps, Δt, ρ)
    G = length(ρ)
    _check_foi(foi, G)
    r0 = _rates(foi, collect(float.(ρ)))
    length(r0) == G || throw(DimensionMismatch("foi must return $G rates; got $(length(r0))"))
    T = promote_type(_eltype(structure), eltype(ρ), eltype(r0))
    S_hist = Matrix{T}(undef, nsteps + 1, G)
    Λ = Matrix{T}(undef, nsteps, G)
    _map!(S_hist, structure, foi, ρ, Float64(Δt), Λ)
    τf = [_attack_and_pmf(view(Λ, :, g)) for g in 1:G]
    return MapSolution(S_hist, first.(τf), reduce(hcat, last.(τf)), Float64(Δt))
end

function _check_map_args(nsteps, Δt, ρ)
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    Δt > 0 || throw(ArgumentError("step length Δt must be > 0; got $Δt"))
    all(r -> 0 ≤ r < 1, ρ) || throw(ArgumentError("initially infected fractions ρ must be in [0, 1); got $(collect(ρ))"))
    return nothing
end
