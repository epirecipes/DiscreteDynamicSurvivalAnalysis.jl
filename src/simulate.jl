# Synthetic line lists: the DDSA working model and a stochastic chain binomial.

"""
    simulate_linelist(rng, dph::DPH, β, ρ, N, nsteps; M = 1, Δt = 1.0) -> NamedTuple

Draw a synthetic line list from the DDSA working model. The GLCT forward map with infectious period `dph`, transmission rate `β`
(per unit time) and initially infected fraction `ρ` gives the attack fraction `τ` and the
infection-time distribution `f` over `nsteps` steps of length `Δt`; then

1. the final size `K ~ Binomial(N, τ)`,
2. the infection-step histogram `~ Multinomial(K, f)`,
3. `M + K` recovery intervals drawn independently from `dph` (the `M` initial infectives and the
   `K` new cases), by inversion with one uniform each.

Returns `(data = LineList, τ, f)`.
"""
function simulate_linelist(rng::AbstractRNG, dph::DPH, β::Real, ρ::Real, N::Integer, nsteps::Integer;
                           M::Integer = 1, Δt::Real = 1.0)
    N ≥ 0 || throw(ArgumentError("N must be ≥ 0; got $N"))
    M ≥ 0 || throw(ArgumentError("M must be ≥ 0; got $M"))
    sol = forward_map(GLCT(dph), β, ρ, nsteps; Δt = Δt)
    τ, f = sol.τ, sol.f
    K = rand(rng, Binomial(N, τ))
    th = if K > 0
        fn = sum(f) > 0 ? f / sum(f) : fill(1.0 / nsteps, nsteps)
        rand(rng, Multinomial(K, fn))
    else
        zeros(Int, nsteps)
    end
    delta = [rand(rng, dph) for _ in 1:(M + K)]
    data = LineList(th; final_size = K, recovery_intervals = delta, Δt = Δt)
    return (data = data, τ = τ, f = f)
end

"""
    simulate_chain_binomial(rng, dph::DPH, β, N; M = 1, Δt = 1.0, max_steps = 5000) -> NamedTuple

Simulate a stochastic chain-binomial SIR outbreak, for checking DDSA on data that do not come from
its own (mean-field) working model. `N` susceptibles and `M` initial infectives (`ρ = M / (N + M)`
in the map's convention): on each step `k`, the number of new infections is
`Binomial(S, 1 − exp(−β I Δt / (N + M)))` (frequency-dependent transmission, as the map), and each
new infective's infectious period `W` is drawn from `dph` at infection: infected on step `k`, it is
infectious on steps `k + 1, …, k + W` and its recorded interval is `W`. The initial infectives are
infected at step 0. The outbreak runs until no infectives remain (or `max_steps`).

Returns `(data, infection_steps, removal_steps, seed_intervals)`, where `data` is a complete
[`LineList`](@ref) up to the last removal (no censoring; its recovery intervals include the
initial infectives and its final size counts the new infections), so the matching final-size term
is `Binomial(N, τ)` over the `N` initially susceptible.
"""
function simulate_chain_binomial(rng::AbstractRNG, dph::DPH, β::Real, N::Integer;
                                 M::Integer = 1, Δt::Real = 1.0, max_steps::Integer = 5000)
    N ≥ 0 || throw(ArgumentError("N must be ≥ 0; got $N"))
    M ≥ 1 || throw(ArgumentError("at least one initial infective is needed; got M = $M"))
    β ≥ 0 || throw(ArgumentError("β must be ≥ 0; got $β"))
    Npop = N + M
    rem = zeros(Int, max_steps + 1)               # removals scheduled at the end of each step
    function schedule!(step)
        step > length(rem) && append!(rem, zeros(Int, step - length(rem)))
        rem[step] += 1
    end
    seedW = [rand(rng, dph) for _ in 1:M]
    foreach(schedule!, seedW)
    S, I = N, M
    infstep = Int[]; remstep = Int[]
    k = 0
    while I > 0 && k < max_steps
        k += 1
        pinf = 1 - exp(-β * I * Δt / Npop)
        n = rand(rng, Binomial(S, pinf))
        S -= n
        I -= rem[k]
        for _ in 1:n
            w = rand(rng, dph)
            push!(infstep, k); push!(remstep, k + w); schedule!(k + w)
        end
        I += n
    end
    T = maximum(vcat(remstep, seedW))
    th = zeros(Int, T)
    for s in infstep
        th[s] += 1
    end
    delta = vcat(seedW, remstep .- infstep)
    data = LineList(th; final_size = length(infstep), recovery_intervals = delta, Δt = Δt)
    return (data = data, infection_steps = infstep, removal_steps = remstep, seed_intervals = seedW)
end
