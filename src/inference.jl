# Posterior sampling for a DDSAModel and recovery of the summed-out integer parameters.

"""
    DDSAFit

Result of [`fit_ddsa`](@ref).

# Fields
- `spec::DDSAModel`: the model specification.
- `data::LineList`: the data.
- `model`: the Turing model ([`turing_model`](@ref)).
- `chain::MCMCChains.Chains`: posterior draws of the sampled parameters, with the summed-out
  integer parameters added as columns: `:r` (number of stages, when `r` has more than one value)
  and `:N` (population size, when it is unknown).
- `population`: the population of the final-size term (an `Int`, an [`NGrid`](@ref) or
  `nothing`).
- `τ::Matrix{Float64}`: the (guarded) attack fraction of each draw, for its drawn `r`
  (iterations × chains; `NaN` when the model has no forward map).

Use [`posterior_draws`](@ref), [`posterior_summary`](@ref), [`posterior_predictive`](@ref) and
[`recovery_predictive`](@ref) to inspect it; `fit.chain` works with the MCMCChains tools.
"""
struct DDSAFit{M<:DDSAModel, Mod, C<:Chains, P}
    spec::M
    data::LineList
    model::Mod
    chain::C
    population::P
    τ::Matrix{Float64}
end

function Base.show(io::IO, fit::DDSAFit)
    niter, _, nch = size(fit.chain)
    print(io, "DDSAFit($(nameof(typeof(fit.spec.infectious_period))), $nch chains × $niter draws, ",
          "parameters: $(join(names(fit.chain, :parameters), ", ")))")
end

"""
    default_sampler()

The NUTS settings of the manuscript: `NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0)`
(1,000 adaptation steps, target acceptance 0.8).
"""
default_sampler() = NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0)

"""
    fit_ddsa([rng], spec::DDSAModel, data::LineList; ndraws = 2000, nchains = 4,
             sampler = default_sampler(), parallel = MCMCThreads(), initial_params = :data,
             progress = false, verbose = false) -> DDSAFit

Sample the posterior of `spec` given `data` with Turing, then draw the summed-out integer
parameters (`r`, `N`) once per MCMC draw from their exact conditional posteriors. The defaults are
the manuscript's settings: 4 chains of 2,000 draws after 1,000 NUTS adaptation steps.

- `rng`: random number generator for the initial values, the sampler and the integer draws
  (default `Random.default_rng()`); a seeded generator gives reproducible results.
- `parallel`: `MCMCThreads()` (chains on Julia threads; start Julia with `--threads`),
  `MCMCSerial()` or `MCMCDistributed()`.
- `initial_params`: `:data` (jittered data-based starting values, [`data_based_inits`](@ref)),
  `:prior` (draws from the prior), or a vector of one Turing initialisation per chain (e.g.
  `InitFromParams((β = 0.2, p = 0.5, ρ = 0.01))`).
- `progress`: show Turing's progress bars.
- `verbose`: show Turing's informational messages (e.g. "Found initial step size"); warnings are
  always shown.

Throws `ArgumentError` for invalid settings and for data incompatible with `spec` (e.g. a known
population smaller than the final size).
"""
function fit_ddsa(rng::AbstractRNG, spec::DDSAModel, data::LineList; ndraws::Integer = 2000, nchains::Integer = 4,
                  sampler = default_sampler(), parallel = MCMCThreads(), initial_params = :data,
                  progress::Bool = false, verbose::Bool = false)
    ndraws ≥ 1 || throw(ArgumentError("ndraws must be ≥ 1; got $ndraws"))
    nchains ≥ 1 || throw(ArgumentError("nchains must be ≥ 1; got $nchains"))
    model = turing_model(spec, data)
    pop = model.args.pop
    inits = _initial_params(rng, spec, data, nchains, initial_params)
    run() = if nchains == 1
        sample(rng, model, sampler, ndraws; progress = progress, chain_type = Chains,
               initial_params = only(inits))
    else
        sample(rng, model, sampler, parallel, ndraws, nchains; progress = progress, chain_type = Chains,
               initial_params = inits)
    end
    # tasks spawned for the chains inherit the task-local logger
    chain = verbose ? run() : with_logger(run, _MinLevelLogger(current_logger(), Logging.Warn))
    chain, τ = _add_integer_draws(rng, spec, model, chain, pop)
    return DDSAFit(spec, data, model, chain, pop, τ)
end
fit_ddsa(spec::DDSAModel, data::LineList; kwargs...) = fit_ddsa(Random.default_rng(), spec, data; kwargs...)

function _initial_params(rng, spec, data, n, x)
    x === :data && return data_based_inits(rng, spec, data, n)
    x === :prior && return fill(InitFromPrior(), n)
    if x isa AbstractVector
        length(x) == n || throw(ArgumentError("initial_params must have one entry per chain ($n); got $(length(x))"))
        return x
    end
    throw(ArgumentError("initial_params must be :data, :prior or a vector of initialisations"))
end

"""
    data_based_inits(rng, spec::DDSAModel, data::LineList, n) -> Vector

Jittered data-based starting values for `n` chains (the heuristic of the manuscript):

- mean infectious period `μ₀`: the mean observed recovery interval (if the recovery term is
  used), otherwise the prior median;
- `p₀ = min δ / mean δ` (negative binomial) or `1 / mean δ` (geometric), `γ₀ = −log(1 − p₀)/Δt`;
- `R₀` from the final-size relation `R̂₀ = −log(1 − z)/z`, `z = K/N`, for a known population, or
  `R̂₀ = 1` for an unknown one; `β₀ = R̂₀ / μ₀`;
- `ρ₀`: the prior median.

Each value is multiplied by `exp(0.1 z)`, `z ~ Normal(0, 1)` (drawn from `rng`), and clamped to the
1%–99% quantiles of its prior. Returns `InitFromParams` objects for `sample`.
"""
function data_based_inits(rng::AbstractRNG, spec::DDSAModel, data::LineList, n::Integer)
    pri = spec.priors
    ip = spec.infectious_period
    Δt = data.Δt
    δ = :recovery in spec.terms ? observed_intervals(data) : Int[]
    has_δ = !isempty(δ)
    pop = _resolve_population(spec.population, data)
    R0h = if pop isa Integer
        z = min(data.final_size / pop, 0.99)
        z > 0 ? -log(1 - z) / z : 1.0
    else
        1.0
    end
    negbin = !(ip isa GeometricIP)
    p_hat = has_δ ? min(negbin ? minimum(δ) / mean(δ) : 1 / mean(δ), 0.99) : nothing
    ipval = if ip.parameter === :p
        has_δ ? p_hat : median(pri.p)
    elseif ip.parameter === :γ
        has_δ ? -log(1 - p_hat) / Δt : median(pri.γ)
    else
        has_δ ? mean(δ) * Δt : median(pri.mean_ip)
    end
    μ0 = if has_δ
        mean(δ) * Δt
    elseif ip.parameter === :mean_ip
        ipval
    elseif ip.parameter === :γ
        1 / ipval
    else
        Δt / ipval
    end
    clamp_prior(x, d) = clamp(x, quantile(d, 0.01), quantile(d, 0.99))
    jitter() = exp(0.1 * randn(rng))
    return map(1:n) do _
        vals = Pair{Symbol,Float64}[]
        if _needs_map(spec)
            spec.transmission === :R0 ? push!(vals, :R0 => clamp_prior(jitter() * R0h, pri.R0)) :
                                        push!(vals, :β => clamp_prior(jitter() * R0h / μ0, pri.β))
        end
        push!(vals, ip.parameter => clamp_prior(jitter() * ipval, pri[ip.parameter]))
        _needs_map(spec) && push!(vals, :ρ => clamp_prior(jitter() * median(pri.ρ), pri.ρ))
        InitFromParams((; vals...))
    end
end

"""Draw r and N for every MCMC draw and add them to the chain; also return the attack fractions."""
function _add_integer_draws(rng, spec, model, chain, pop)
    rets = returned(model, chain)
    niter, nch = size(rets)
    support = r_support(spec.infectious_period)
    rs = Matrix{Int}(undef, niter, nch)
    Ns = Matrix{Int}(undef, niter, nch)
    τs = Matrix{Float64}(undef, niter, nch)
    for c in 1:nch, i in 1:niter
        ret = rets[i, c]
        idx = length(support) > 1 ? sample_index(rng, ret.lps) : 1
        rs[i, c] = support[idx]
        τs[i, c] = ret.τ[idx]
        pop isa NGrid && (Ns[i, c] = sample_N(rng, pop, τs[i, c]))
    end
    add(ch, name, draws) = hcat(ch, setrange(Chains(reshape(draws, niter, 1, nch), [name]), range(ch)))
    length(support) > 1 && (chain = add(chain, :r, rs))
    pop isa NGrid && (chain = add(chain, :N, Ns))
    return chain, τs
end
