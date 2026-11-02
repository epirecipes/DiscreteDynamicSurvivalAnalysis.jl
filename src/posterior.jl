# Posterior quantities, summaries and posterior-predictive checks of a DDSAFit.

_draws(ch, s) = vec(Array(ch[s]))     # pooled over chains, chain 1 first

"""
    posterior_draws(fit::DDSAFit) -> NamedTuple

Posterior draws pooled over chains (chain 1 first, then chain 2, …), as vectors:

- the sampled parameters (`β` or `R0`, `p`/`γ`/`mean_ip`, `ρ`, as in the model);
- `r`: number of stages (1 for a geometric infectious period);
- `N`: population size (constant when known), and `K_over_N = K/N` when it is unknown;
- `τ`: attack fraction of the map (after the numerical guards);
- derived: `β` (if `R0` was sampled), `p` (per-step progression probability), `mean_ip` (mean
  infectious period `r Δt / p`, time units), `mean_ip_steps` (`r / p`) and `R0 = β · mean_ip`
  (for a geometric period, `β / p` per step).

Fields that a model does not have (e.g. `β` in a recovery-only model) are omitted.
"""
function posterior_draws(fit::DDSAFit)
    spec, data, ch = fit.spec, fit.data, fit.chain
    ip = spec.infectious_period
    Δt = data.Δt
    pars = names(ch, :parameters)
    n = size(ch, 1) * size(ch, 3)
    out = Pair{Symbol,Vector}[]
    for k in keys(spec.priors)
        push!(out, k => _draws(ch, k))
    end
    r = :r in pars ? Float64.(_draws(ch, :r)) : fill(Float64(first(r_support(ip))), n)
    ipv = _draws(ch, ip.parameter)
    p = ip.parameter === :p ? ipv : ip.parameter === :γ ? rate_to_proportion.(ipv, Δt) : r .* Δt ./ ipv
    push!(out, :r => Int.(r))
    if fit.population isa NGrid
        N = Float64.(_draws(ch, :N))
        push!(out, :N => Int.(N), :K_over_N => data.final_size ./ N)
    elseif fit.population isa Integer
        push!(out, :N => fill(fit.population, n))
    end
    _needs_map(spec) && push!(out, :τ => vec(fit.τ))
    ip.parameter === :p || push!(out, :p => p)
    mean_ip_steps = r ./ p
    ip.parameter === :mean_ip || push!(out, :mean_ip => Δt .* mean_ip_steps)
    push!(out, :mean_ip_steps => mean_ip_steps)
    if _needs_map(spec)
        β = spec.transmission === :R0 ? _draws(ch, :R0) ./ _draws(ch, :mean_ip) : _draws(ch, :β)
        spec.transmission === :R0 ? push!(out, :β => β) : push!(out, :R0 => β .* Δt .* mean_ip_steps)
    end
    return (; out...)
end

"""
    posterior_summary(fit::DDSAFit; probs = (0.025, 0.5, 0.975)) -> Vector{NamedTuple}

One row per quantity of [`posterior_draws`](@ref): `parameter`, `mean`, `std`, the requested
quantiles (`q2_5`, `q50`, `q97_5` for the default `probs`: the percentage with `_` for the decimal
point), and the bulk effective sample size `ess` and `rhat`
across chains (`NaN` for a constant quantity). The result is a Tables.jl table (e.g.
`DataFrame(posterior_summary(fit))`).
"""
function posterior_summary(fit::DDSAFit; probs = (0.025, 0.5, 0.975))
    d = posterior_draws(fit)
    niter, nch = size(fit.chain, 1), size(fit.chain, 3)
    qnames = Tuple(_quantile_name(q) for q in probs)
    rows = NamedTuple[]
    for (name, x) in pairs(d)
        xs = Float64.(x)
        constant = all(==(first(xs)), xs)
        ess, rhat = constant ? (NaN, NaN) : ess_rhat(reshape(xs, niter, nch, 1))
        qs = NamedTuple{qnames}(Tuple(quantile(xs, q) for q in probs))
        push!(rows, merge((parameter = name, mean = mean(xs), std = std(xs)), qs,
                          (ess = only(ess), rhat = only(rhat))))
    end
    return [r for r in rows]
end

"""Field name of the `q` quantile: `0.025 → :q2_5`, `0.5 → :q50`."""
_quantile_name(q) = Symbol("q", replace(string(round(100 * q; digits = 1)), r"\.0$" => "", "." => "_"))

_pooled_index(n_total, ndraws) = unique(round.(Int, range(1, n_total; length = min(ndraws, n_total))))

"""
    PosteriorPredictive

Output of [`posterior_predictive`](@ref).

# Fields
- `times::Vector{Float64}`: start time of each step, `(k - 1) Δt`.
- `observed::Vector{Int}`: observed infection counts per step.
- `incidence::Matrix{Float64}`: predicted cases per step (steps × draws).
- `final_size::Vector{Int}`: simulated final sizes `Binomial(N, τ)`, one per draw.
- `observed_final_size::Int`: the observed final size `K`.
- `scale::Symbol`: `:expected_known` (`(K_known / K) N τ f`), `:expected` (`N τ f`) or
  `:observed_total` (`K_known f`), with `K_known = Σ counts`.

[`predictive_quantiles`](@ref) gives pointwise quantile bands; with Plots.jl loaded,
`plot(pp)` draws the observed counts and the bands.
"""
struct PosteriorPredictive
    times::Vector{Float64}
    observed::Vector{Int}
    incidence::Matrix{Float64}
    final_size::Vector{Int}
    observed_final_size::Int
    scale::Symbol
end

"""
    posterior_predictive([rng], fit::DDSAFit; ndraws = 500, scale = :expected_known) -> PosteriorPredictive

Posterior predictive infection curve: for `ndraws` posterior draws evenly spaced over the pooled
chains, the forward map is re-run at the draw's parameters (including its `r`) and the expected
cases per step are computed as below. `N` is the draw's population size, `K` the final size and
`K_known = Σ counts` the number of cases with a known infection time.

- `scale = :expected_known` (default): `(K_known / K) N τ f`, the expected number of cases with
  a known infection time, which is what the observed counts count. It assumes that infection
  times are missing independently of when the infection happened. When every infection time is
  known it equals `:expected`;
- `scale = :expected`: `N τ f`, the expected new infections of all cases. With missing infection
  times it lies above the observed counts by the factor `K / K_known`;
- `scale = :observed_total`: `K_known f`, the infection-time distribution scaled to the observed
  counts. It checks the shape of the curve only.

Also draws one final size `Binomial(N, τ)` per draw from `rng`, to compare with `K`. Needs a model
with a forward map.
"""
function posterior_predictive(rng::AbstractRNG, fit::DDSAFit; ndraws::Integer = 500, scale::Symbol = :expected_known)
    scale in (:expected_known, :expected, :observed_total) || throw(ArgumentError(
        "scale must be :expected_known, :expected or :observed_total; got :$scale"))
    spec, data = fit.spec, fit.data
    _needs_map(spec) || throw(ArgumentError("the model has no forward map (its terms are $(spec.terms))"))
    d = posterior_draws(fit)
    idx = _pooled_index(length(d.r), ndraws)
    ip = spec.infectious_period
    inc = Matrix{Float64}(undef, data.nsteps, length(idx))
    fs = Vector{Int}(undef, length(idx))
    Kobs = sum(data.infection_counts)
    known = iszero(data.final_size) ? 1.0 : Kobs / data.final_size    # fraction with a known infection time
    for (j, i) in enumerate(idx)
        r, p = d.r[i], d.p[i]
        dph = _dph(ip, r, p)
        sol = forward_map(_structure(ip, r, p, dph), d.β[i], d.ρ[i], data.nsteps; Δt = data.Δt)
        N = hasproperty(d, :N) ? d.N[i] : Kobs
        inc[:, j] .= scale === :observed_total ? Kobs .* sol.f :
                     scale === :expected ? N * sol.τ .* sol.f : known * N * sol.τ .* sol.f
        fs[j] = rand(rng, Binomial(round(Int, N), clamp(sol.τ, 0.0, 1.0)))
    end
    times = collect((0:(data.nsteps - 1)) .* data.Δt)
    return PosteriorPredictive(times, copy(data.infection_counts), inc, fs, data.final_size, scale)
end
posterior_predictive(fit::DDSAFit; kwargs...) = posterior_predictive(Random.default_rng(), fit; kwargs...)

"""
    predictive_quantiles(x; probs = (0.025, 0.5, 0.975)) -> Matrix

Pointwise quantiles of a [`PosteriorPredictive`](@ref) incidence (steps × `length(probs)`) or of a
[`RecoveryPredictive`](@ref) pmf (intervals × `length(probs)`).
"""
predictive_quantiles(pp::PosteriorPredictive; probs = (0.025, 0.5, 0.975)) = _rowquantiles(pp.incidence, probs)

_rowquantiles(M, probs) = [quantile(view(M, i, :), q) for i in axes(M, 1), q in probs]

"""
    RecoveryPredictive

Output of [`recovery_predictive`](@ref).

# Fields
- `intervals::UnitRange{Int}`: interval lengths `1:max_interval` (steps).
- `pmf::Matrix{Float64}`: infectious-period pmf at each posterior draw (intervals × draws).
- `observed::Vector{Int}`: counts of the observed (uncensored) intervals of each length.
"""
struct RecoveryPredictive
    intervals::UnitRange{Int}
    pmf::Matrix{Float64}
    observed::Vector{Int}
end

"""
    recovery_predictive(fit::DDSAFit; ndraws = 200, max_interval = maximum(intervals) + 5)
        -> RecoveryPredictive

Posterior infectious-period distribution: the pmf `P(W = δ)` for `δ = 1:max_interval` at `ndraws`
evenly spaced posterior draws, with the histogram of the observed (uncensored) recovery intervals.
`predictive_quantiles` gives bands; with Plots.jl loaded, `plot(rp)` draws them against the
observed relative frequencies.
"""
function recovery_predictive(fit::DDSAFit; ndraws::Integer = 200, max_interval::Union{Nothing,Integer} = nothing)
    δ = observed_intervals(fit.data)
    mx = max_interval === nothing ? (isempty(δ) ? 30 : maximum(δ) + 5) : Int(max_interval)
    d = posterior_draws(fit)
    idx = _pooled_index(length(d.r), ndraws)
    ip = fit.spec.infectious_period
    P = Matrix{Float64}(undef, mx, length(idx))
    for (j, i) in enumerate(idx)
        dph = _dph(ip, d.r[i], d.p[i])
        P[:, j] .= [pdf(dph, x) for x in 1:mx]
    end
    obs = [count(==(x), δ) for x in 1:mx]
    return RecoveryPredictive(1:mx, P, obs)
end

predictive_quantiles(rp::RecoveryPredictive; probs = (0.025, 0.5, 0.975)) = _rowquantiles(rp.pmf, probs)
