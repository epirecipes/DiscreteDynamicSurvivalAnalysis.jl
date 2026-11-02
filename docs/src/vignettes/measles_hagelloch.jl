# # Measles in Hagelloch, 1861: a custom three-group model
#
# This vignette reproduces the Hagelloch analysis of the manuscript. It is
# also the package's worked example of a **custom multi-group model**. The standard workflow
# ([`DDSAModel`](@ref), [`fit_ddsa`](@ref)) covers one population; here three groups with their
# own force of infection are written as a Turing model from the building blocks:
# [`forward_map`](@ref) with a callable force of infection, [`ddsa_loglikelihood`](@ref) per group,
# [`recovery_loglik`](@ref), and the integer-parameter helpers [`NGrid`](@ref) and
# [`marginal_r_loglik`](@ref).
#
# ## The question
#
# The 1861 measles outbreak in Hagelloch (188 cases among the village children) was analysed at the
# individual level by Neal & Roberts (2004, "NR"), whose model ``M_\theta`` has within-class,
# household and global transmission. Can a mean-field DDSA analogue with three groups recover the
# qualitative structure of that model? The groups are the school classes:
#
# - class 0: children not yet at school (pre-school);
# - class 1: the younger school class;
# - class 2: the older school class, which contains the index case.
#
# The questions are whether the fit gives ``\beta_{C1} > \beta_{C2}`` for the within-class rates
# of classes 1 and 2, a household rate ``\beta_H > 0`` from the school classes to class 0, the
# right order of the three epidemics, and the group sizes ``N_0, N_1, N_2``. NR treat the group
# sizes as known; DDSA infers them from the final-size terms. The same model is fitted with the
# GLCT and the Boxcar forward maps, and their agreement checks the implementation.

using DiscreteDynamicSurvivalAnalysis
using CSV, DataFrames, Distributions, Turing, MCMCChains
using LinearAlgebra, Random, Statistics
using Plots, StatsPlots, Measures
using Logging; disable_logging(Logging.Info) # restored at the end #hide

const FULL = get(ENV, "DDSA_FULL", "false") == "true"
const REPRODUCTION = Dict{String,Dict{String,Any}}() #src
t_start = time()

const NDRAWS = FULL ? 2000 : 100          # draws per chain after adaptation
const NCHAINS = FULL ? 4 : 2
const SAMPLER = FULL ? default_sampler() : NUTS(100, 0.8; max_depth = 12, Δ_max = 1000.0)
const NPPC = FULL ? 200 : 100              # posterior draws used for the predictive checks
(; FULL, NDRAWS, NCHAINS, SAMPLER, NPPC, threads = Threads.nthreads())

# ## Inputs
#
# The line list is distributed with the package (see `data/README.md` for its provenance). The
# column `t_infection` is the prodrome onset in days since the first prodrome, and `inf_period` is
# the infectious period with NR's offset already added: infectious from 1 day before prodrome
# until 3 days after rash onset, so `inf_period = t_rash − t_infection + 4`.

df = CSV.read(example_data_path(:hagelloch), DataFrame)
cls = parse.(Int, string.(df.class))       # the column may parse as Int or String
classes = [df[cls .== c, :] for c in 0:2]
first(df[:, [:case_ID, :infector, :class, :t_infection, :t_rash, :inf_period]], 5)

# The three groups share one calendar grid. As in the manuscript, the grid covers the last
# infection (day `Tmax`) plus about two mean infectious periods.

Tmax = maximum(df.t_infection)
nsteps = Tmax + 2 * round(Int, mean(df.inf_period))
groups = [linelist(d.t_infection, d.t_infection .+ d.inf_period; nsteps = nsteps) for d in classes]

# Each group is a [`LineList`](@ref) on the same `nsteps`-day grid (day `d` is step `d + 1`). The
# recovery intervals are pooled over the groups, because the infectious period is shared.

delta = reduce(vcat, [g.recovery_intervals for g in groups])
K0, K1, K2 = (g.final_size for g in groups)
(; K0, K1, K2, nsteps, mean_delta = mean(delta), min_delta = minimum(delta),
 max_delta = maximum(delta), cv_delta = std(delta) / mean(delta))

# The coefficient of variation of the recovery intervals (≈ 0.22) is far below the ≈ 0.94 of a
# geometric distribution with the same mean, so the infectious period is modelled as a negative
# binomial (DPH) distribution.
#
# Infection-time histograms of the three groups:

ymax = maximum(maximum(g.infection_counts) for g in groups)
plot([bar(0:(nsteps - 1), g.infection_counts; title = t, color = c, alpha = 0.7, label = "",
          xlabel = "day", ylabel = "new infections", ylims = (0, ymax))
      for (g, t, c) in zip(groups, ["class 0 (pre-school)", "class 1 (younger)", "class 2 (older)"],
                           [:forestgreen, :steelblue, :tomato])]...;
     layout = (1, 3), size = (1050, 300), left_margin = 4mm, bottom_margin = 4mm)

# The `infector` column (NR's reconstruction of who infected whom) shows that most pre-school
# cases were infected by a school child, which motivates the household term ``\beta_H``.

class_of = Dict(zip(df.case_ID, cls))
infector_class = [get(class_of, something(tryparse(Int, string(x)), -1), missing) for x in classes[1].infector]
(from_school = count(c -> !ismissing(c) && c in (1, 2), infector_class),
 from_class0 = count(c -> !ismissing(c) && c == 0, infector_class),
 unresolved = count(ismissing, infector_class), K0 = K0)

# ## Model configuration
#
# ### Force of infection: the callable `foi` interface
#
# For `G` groups, [`forward_map`](@ref) takes a vector `ρ` of initially infected fractions and a
# force of infection `foi` that maps the vector `I` of infected fractions (one per group) to one
# infection rate per group. Group ``k`` then has new infections
# ``(1 - e^{-\mathrm{rate}_k \Delta t}) S_k`` per step, where ``S_k`` is its susceptible fraction
# and ``\Delta t = 1`` day the step length. A matrix `B` gives `B * I`; any callable works, for
# example a closure over the parameters. The analogue of NR's ``M_\theta`` (household, two
# classroom and one global term, with the spatial decay switched off) is:

mtheta_foi(β_C1, β_C2, β_H, β_G) = I -> begin
    Itot = I[1] + I[2] + I[3]
    [β_H * (I[2] + I[3]) + β_G * Itot,     # class 0: household route from school children + global
     β_C1 * I[2] + β_G * Itot,             # class 1: within class 1 + global
     β_C2 * I[3] + β_G * Itot]             # class 2: within class 2 + global
end

# The closure must accept vectors of ForwardDiff `Dual`s, because NUTS differentiates through the
# map. A matrix would do here too, since this force of infection is linear in `I`. A callable is
# needed for non-linear forces of infection (e.g. saturating contact rates).
#
# The global term ``\beta_G (I_0 + I_1 + I_2)`` includes all three groups, so pre-school children
# also infect school children through ``\beta_G``. The next-generation matrix used below for the
# basic reproduction number ``R_0`` uses the same rates.
#
# ### Infectious period
#
# The calendar infectious period ``\delta`` has a minimum of ``\delta_{\min} = 4`` days (NR's
# offset). It is a negative binomial with ``r`` stages, preceded by ``\delta_{\min} - r``
# deterministic one-day states ([`dph_negative_binomial_min_support`](@ref)), so its support is
# ``\{\delta_{\min}, \delta_{\min} + 1, \ldots\}``. NUTS samples its mean ``E[\delta]``, and
# ``p = r / (E[\delta] - \delta_{\min} + r)``; ``r \in \{1, \ldots, \delta_{\min}\}`` is summed
# out.

δ_floor = minimum(delta)
r_max = δ_floor
dph_nominal = dph_negative_binomial_min_support(r_max, r_max / (mean(delta) - δ_floor + r_max), δ_floor)
(; δ_floor, r_max, mean = mean(dph_nominal))

# The Boxcar map needs a number of infection-age classes. As in the manuscript it is the 99.99th
# percentile of a nominal infectious period (`dph_nominal`: ``r = \delta_{\min}`` and ``E[\delta]``
# equal to the mean recorded interval), but at least 40. The hazard is the exact one of
# ``\min(\delta, K_{\text{age}})`` (`truncation = :absorb`).

K_age = max(40, findfirst(k -> cdf(dph_nominal, k) ≥ 0.9999, 1:200))

# One forward-map run at the manuscript's posterior medians shows the three epidemics: class 1
# first, then class 2 and the pre-school children almost together. The index case is in class 2
# (``\rho_2 = 1/N_2``, ``\rho_0 = \rho_1 = 0``).

foi_demo = mtheta_foi(0.1174, 0.021, 0.0124, 0.0638)
dph_demo = dph_negative_binomial_min_support(4, 4 / (8.3537 - δ_floor + 4), δ_floor)
sol_demo = forward_map(GLCT(dph_demo), foi_demo, [0.0, 0.0, 1 / 77], nsteps)
B_demo = [0.0638 0.0124+0.0638 0.0124+0.0638; 0.0638 0.1174+0.0638 0.0638; 0.0638 0.0638 0.021+0.0638]
sol_B = forward_map(GLCT(dph_demo), B_demo, [0.0, 0.0, 1 / 77], nsteps)
(τ = round.(sol_demo.τ; digits = 3), same_as_matrix_form = sol_demo.τ ≈ sol_B.τ)

#-

plot(0:(nsteps - 1), sol_demo.f; label = ["class 0" "class 1" "class 2"],
     color = [:forestgreen :steelblue :tomato], xlabel = "day", ylabel = "infection-time pmf f",
     linewidth = 2, size = (600, 300))

# ### Populations
#
# The group sizes are unknown. Pfeilsticker (1863, as cited by NR) recorded 197 children under 14,
# so with all 197 children in one of the three groups and ``K_k`` cases in group ``k``, group ``k``
# has at most ``N_{k,\max} = 197 - K_{k'} - K_{k''}`` members (``k'``, ``k''`` are the other two
# groups). Each ``N_k`` has a discrete uniform prior on ``\{K_k, \ldots, N_{k,\max}\}``.
#
# - ``N_0`` and ``N_1`` only enter their final-size terms. They are summed out with an
#   [`NGrid`](@ref) (the `population` argument of [`ddsa_loglikelihood`](@ref)).
# - ``N_2`` also sets the seed ``\rho_2``, and so changes the map. It is summed out
#   together with ``r`` over a grid of cells, one forward-map run per cell.

N_total = 197
N0_max, N1_max, N2_max = N_total - K1 - K2, N_total - K0 - K2, N_total - K0 - K1
N0_grid = NGrid(K0, K0, N0_max)
N1_grid = NGrid(K1, K1, N1_max)
cells = vec(collect(Iterators.product(1:r_max, K2:N2_max)))   # (r, N2) pairs
(; N0 = K0:N0_max, N1 = K1:N1_max, N2 = K2:N2_max, ncells = length(cells))

# ### Likelihood of one (r, N2) cell
#
# For given transmission rates, mean infectious period and cell ``(r, N_2)``, the log-likelihood
# is the sum of:
#
# - per group, the final-size and infection-time terms (`ddsa_loglikelihood` with
#   `group = g`), with ``N_0``, ``N_1`` summed out and ``N_2`` known in the cell;
# - once, the recovery term of the pooled intervals. It depends on ``r`` but not on ``N_2``, so
#   the model below computes it once per ``r`` and adds it to that ``r``'s cells;
#   `hagelloch_cell` returns the group terms only.
#
# The manuscript's numerical guard clamps the attack fraction ``\tau_k`` of each group (its expected
# fraction infected) to ``[0.001, 0.999]``. `hagelloch_cell` also returns the guarded ``\tau_0`` and
# ``\tau_1``, which are used later to draw ``N_0`` and ``N_1``.

hd = (; groups, delta, δ_floor, r_max, nsteps, N0_grid, N1_grid, cells,
      guards = NumericalGuards(τ_bounds = (0.001, 0.999)))

## infectious period with r stages and mean mean_δ: p = r / (E[δ] − δ_min + r)
hagelloch_dph(hd, r, mean_δ) = dph_negative_binomial_min_support(r, r / (mean_δ - hd.δ_floor + r), hd.δ_floor)

function hagelloch_cell(hd, make_structure, foi, r, N2, mean_δ)
    dph = hagelloch_dph(hd, r, mean_δ)
    sol = forward_map(make_structure(dph), foi, [0.0, 0.0, 1 / N2], hd.nsteps)
    ll = zero(eltype(dph.s))
    τ = zeros(eltype(dph.s), 3)
    for (g, pop) in enumerate((hd.N0_grid, hd.N1_grid, N2))
        res = ddsa_loglikelihood(sol, hd.groups[g], dph; population = pop, group = g, guards = hd.guards,
                                 terms = (:final_size, :infection_times))
        ll += res.loglik
        τ[g] = res.τ
    end
    return (ll = ll, τ0 = τ[1], τ1 = τ[2])
end

# ### Turing model
#
# NUTS samples the four transmission rates and ``E[\delta]``. The 40 cells are summed out with
# [`marginal_r_loglik`](@ref), evaluated as parallel tasks (`threaded = true`), with a uniform
# prior over the cells. The recovery term is computed once for each of the 4 values of ``r`` and
# shared by that ``r``'s 10 cells. 
# The model returns the per-cell log-likelihoods, from which ``(r, N_2)`` is drawn after sampling. 
# And `make_structure` turns the DPH into the infection structure of the map, GLCT or Boxcar.

@model function hagelloch_mtheta(hd, make_structure)
    beta_C1 ~ Exponential(0.2)
    beta_C2 ~ Exponential(0.1)
    beta_H ~ Exponential(0.1)
    beta_G ~ Exponential(0.1)
    mean_δ ~ Uniform(hd.δ_floor + 0.5, 25.0)
    foi = mtheta_foi(beta_C1, beta_C2, beta_H, beta_G)
    rec = [recovery_loglik(hagelloch_dph(hd, r, mean_δ), hd.delta) for r in 1:hd.r_max]
    logmarg, lps = marginal_r_loglik(hd.cells; threaded = true) do (r, N2)
        hagelloch_cell(hd, make_structure, foi, r, N2, mean_δ).ll + rec[r]
    end
    Turing.@addlogprob! logmarg
    return lps
end

glct_structure(dph) = GLCT(dph)
boxcar_structure(dph) = Boxcar(dph, K_age)     # truncation = :absorb
model_glct = hagelloch_mtheta(hd, glct_structure)
model_boxcar = hagelloch_mtheta(hd, boxcar_structure)
logjoint(model_glct, (beta_C1 = 0.1174, beta_C2 = 0.021, beta_H = 0.0124, beta_G = 0.0638, mean_δ = 8.3537))

# ## Fit
#
# The chains start from jittered data-based values, as in the manuscript: the transmission rates
# at their prior medians, ``E[\delta]`` at the mean recorded interval, each multiplied by
# ``e^{0.1 z}`` with ``z \sim \mathcal{N}(0, 1)``.

function data_inits(rng, hd, n)
    j() = exp(0.1 * randn(rng))
    return [InitFromParams((beta_C1 = j() * median(Exponential(0.2)), beta_C2 = j() * median(Exponential(0.1)),
                            beta_H = j() * median(Exponential(0.1)), beta_G = j() * median(Exponential(0.1)),
                            mean_δ = clamp(j() * mean(hd.delta), minimum(hd.delta) + 0.6, 24.9)))
            for _ in 1:n]
end

# After sampling, each draw gets integer values from their conditional posteriors.
#
# 1. ``(r, N_2)`` is drawn over the cells from the returned per-cell log-likelihoods
#    ([`sample_index`](@ref)).
# 2. The map of that cell is rerun to get the attack fractions of classes 0 and 1.
# 3. ``N_0`` and ``N_1`` are drawn given those attack fractions ([`sample_N`](@ref)).
#
# The draws are added to the chain as `:r`, `:N0`, `:N1` and `:N2`.

function add_integer_draws(rng, hd, model, chain, make_structure)
    lps = returned(model, chain)
    niter, nch = size(lps)
    pars = [Array(chain[s]) for s in (:beta_C1, :beta_C2, :beta_H, :beta_G, :mean_δ)]
    draws = zeros(Int, niter, 4, nch)
    for c in 1:nch, i in 1:niter
        r, N2 = hd.cells[sample_index(rng, lps[i, c])]
        foi = mtheta_foi((pars[k][i, c] for k in 1:4)...)
        cell = hagelloch_cell(hd, make_structure, foi, r, N2, pars[5][i, c])
        draws[i, :, c] = [r, sample_N(rng, hd.N0_grid, cell.τ0), sample_N(rng, hd.N1_grid, cell.τ1), N2]
    end
    return hcat(chain, MCMCChains.setrange(Chains(draws, [:r, :N0, :N1, :N2]), range(chain)))
end

function fit_mtheta(rng, hd, model, make_structure)
    chain = sample(rng, model, SAMPLER, MCMCThreads(), NDRAWS, NCHAINS; progress = false,
                   chain_type = Chains, initial_params = data_inits(rng, hd, NCHAINS))
    return add_integer_draws(rng, hd, model, chain, make_structure)
end

t_glct = @elapsed chain_glct = fit_mtheta(Xoshiro(20261006), hd, model_glct, glct_structure)
t_boxcar = @elapsed chain_boxcar = fit_mtheta(Xoshiro(20261007), hd, model_boxcar, boxcar_structure)
(seconds_glct = round(t_glct; digits = 1), seconds_boxcar = round(t_boxcar; digits = 1))

# Convergence diagnostics (bulk/tail effective sample sizes and ``\hat R``) of the GLCT fit:

summarystats(chain_glct)

# and of the Boxcar fit:

summarystats(chain_boxcar)

# ## Results
#
# ### Basic reproduction number
#
# ``R_0`` is the dominant eigenvalue of the next-generation matrix ``\mathcal{K}``. Its entry
# ``\mathcal{K}_{ij}`` is the expected number of infections in group ``i`` caused by one infective
# of group ``j`` in a fully susceptible population:
# ``\mathcal{K}_{ij} = (N_i / N_j)\, \beta_{ij}\, E[\delta]``, where ``\beta_{ij}`` is the
# coefficient of ``I_j`` in the rate of group ``i``. The factor ``N_i/N_j`` appears because the
# state variables are per-group fractions. It uses the calendar mean ``E[\delta]``, offset days included.

function compute_R0_mtheta(beta_C1, beta_C2, beta_H, beta_G, mean_ip, N0, N1, N2)
    K = mean_ip .* [(N0 / N0)*beta_G  (N0 / N1)*(beta_H + beta_G)  (N0 / N2)*(beta_H + beta_G);
                    (N1 / N0)*beta_G  (N1 / N1)*(beta_C1 + beta_G) (N1 / N2)*beta_G;
                    (N2 / N0)*beta_G  (N2 / N1)*beta_G             (N2 / N2)*(beta_C2 + beta_G)]
    return maximum(real(eigvals(K)))
end

pooled(chain, s) = vec(Array(chain[s]))     # chain 1 first, then chain 2, …

function posterior_quantities(chain, hd)
    d = Dict{Symbol,Vector{Float64}}(s => pooled(chain, s) for s in (:beta_C1, :beta_C2, :beta_H, :beta_G, :mean_δ,
                                                                     :r, :N0, :N1, :N2))
    d[:p] = d[:r] ./ (d[:mean_δ] .- hd.δ_floor .+ d[:r])
    d[:R0] = [compute_R0_mtheta(d[:beta_C1][i], d[:beta_C2][i], d[:beta_H][i], d[:beta_G][i], d[:mean_δ][i],
                                d[:N0][i], d[:N1][i], d[:N2][i]) for i in eachindex(d[:r])]
    return d
end

post_glct = posterior_quantities(chain_glct, hd)
post_boxcar = posterior_quantities(chain_boxcar, hd)
for (key, post) in (("hagelloch.glct", post_glct), ("hagelloch.boxcar", post_boxcar)) #src
    REPRODUCTION[key] = Dict{String,Any}((s === :mean_δ ? "mean_delta" : String(s)) => v for (s, v) in post #src
                                         if s !== :p) #src
end #src
nothing #hide

# ### Posterior table
#
# Posterior medians and 95% credible intervals of both fits, with the GLCT medians reported by the
# manuscript (full settings).

manuscript_glct = Dict(:beta_C1 => 0.1174, :beta_C2 => 0.0210, :beta_H => 0.0124, :beta_G => 0.0638,
                       :mean_δ => 8.3537, :r => 4, :N0 => 99, :N1 => 34, :N2 => 77, :R0 => 2.167)
fmt(v) = string(round(median(v); sigdigits = 3), " [", round(quantile(v, 0.025); sigdigits = 3), ", ",
                round(quantile(v, 0.975); sigdigits = 3), "]")
params = [:beta_C1, :beta_C2, :beta_H, :beta_G, :mean_δ, :p, :r, :N0, :N1, :N2, :R0]
DataFrame(parameter = String.(params), GLCT = [fmt(post_glct[s]) for s in params],
          Boxcar = [fmt(post_boxcar[s]) for s in params],
          manuscript_GLCT_median = [get(manuscript_glct, s, missing) for s in params])

# Comparison with the structure of NR's ``M_\theta``: the younger class has the higher
# within-class rate, and the household rate is small. The mean-field ``\beta_H`` applies to all
# (class 0) × (school) pairs, not only to siblings, and transmission not attributed to it is
# absorbed by ``\beta_G``.

(β_C1_over_β_C2 = median(post_glct[:beta_C1]) / median(post_glct[:beta_C2]),
 β_H_over_β_G = median(post_glct[:beta_H]) / median(post_glct[:beta_G]),
 P_β_C1_gt_β_C2 = mean(post_glct[:beta_C1] .> post_glct[:beta_C2]))

# Posterior distributions of the GLCT fit (integer parameters with one bin per value):

function posterior_panel(x; title, color, integer = false)
    histogram(x; bins = integer ? ((minimum(x) - 0.5):1:(maximum(x) + 0.5)) : :auto, normalize = :pdf,
              label = "", title = title, color = color, linecolor = :gray, linewidth = 0.3,
              ylabel = integer ? "probability" : "density")
end

plot(posterior_panel(post_glct[:beta_C1]; title = "β_C1", color = :steelblue),
     posterior_panel(post_glct[:beta_C2]; title = "β_C2", color = :tomato),
     posterior_panel(post_glct[:beta_H]; title = "β_H", color = :forestgreen),
     posterior_panel(post_glct[:beta_G]; title = "β_G", color = :purple),
     posterior_panel(post_glct[:mean_δ]; title = "mean δ (days)", color = :steelblue),
     posterior_panel(post_glct[:N0]; title = "N₀ (pre-school)", color = :forestgreen, integer = true),
     posterior_panel(post_glct[:N1]; title = "N₁ (class 1)", color = :steelblue, integer = true),
     posterior_panel(post_glct[:N2]; title = "N₂ (class 2)", color = :tomato, integer = true);
     layout = (2, 4), size = (1200, 500), left_margin = 4mm, bottom_margin = 4mm)

# ### Posterior predictive checks
#
# For `NPPC` draws evenly spaced over the pooled chains, the map is rerun at the draw's parameters
# (including its ``r`` and ``N_2``). The expected new infections per day of group ``k`` are
# ``N_k \tau_k f_k``, with ``N_k`` the draw's group size and ``f_k`` the group's infection-time
# distribution, so the curves show the model's own
# epidemic size. Each group's result is a [`PosteriorPredictive`](@ref), so the package's plot
# recipe and [`predictive_quantiles`](@ref) apply.

function group_predictive(rng, hd, post, make_structure; ndraws)
    idx = unique(round.(Int, range(1, length(post[:r]); length = ndraws)))
    inc = [zeros(hd.nsteps, length(idx)) for _ in 1:3]
    τ = zeros(length(idx), 3)
    final = zeros(Int, length(idx), 3)
    for (j, i) in enumerate(idx)
        r = Int(post[:r][i])
        N = Int.((post[:N0][i], post[:N1][i], post[:N2][i]))
        dph = hagelloch_dph(hd, r, post[:mean_δ][i])
        foi = mtheta_foi(post[:beta_C1][i], post[:beta_C2][i], post[:beta_H][i], post[:beta_G][i])
        sol = forward_map(make_structure(dph), foi, [0.0, 0.0, 1 / N[3]], hd.nsteps)
        for g in 1:3
            τ[j, g] = attack_fraction(sol, g)
            inc[g][:, j] .= N[g] * τ[j, g] .* infection_pmf(sol, g)
            final[j, g] = rand(rng, Binomial(N[g], clamp(τ[j, g], 0.0, 1.0)))
        end
    end
    times = collect(0.0:(hd.nsteps - 1))
    pps = [PosteriorPredictive(times, hd.groups[g].infection_counts, inc[g], final[:, g], hd.groups[g].final_size,
                               :expected) for g in 1:3]
    return (pps = pps, τ = τ, idx = idx)
end

ppc_glct = group_predictive(Xoshiro(1), hd, post_glct, glct_structure; ndraws = NPPC)
ppc_boxcar = group_predictive(Xoshiro(2), hd, post_boxcar, boxcar_structure; ndraws = NPPC)
nothing #hide

# Predicted ``\tau_k`` against the observed attack fractions ``K_k / N_k``, both per posterior
# draw (GLCT). The mean-field model reproduces class 1 but
# under-predicts the attack fractions of classes 0 and 2, whose internal contact structure is
# heterogeneous (households, the older class).

qfmt(v) = string(round(median(v); digits = 3), " [", round(quantile(v, 0.025); digits = 3), ", ",
                 round(quantile(v, 0.975); digits = 3), "]")
DataFrame(group = ["class 0", "class 1", "class 2"],
          predicted_τ = [qfmt(ppc_glct.τ[:, g]) for g in 1:3],
          observed_K_over_N = [qfmt(K ./ post_glct[N][ppc_glct.idx]) for (K, N) in zip((K0, K1, K2), (:N0, :N1, :N2))])

# The order of the epidemics: class 1 should peak before the main wave of class 2, and the
# pre-school wave, driven by infections from school children, after class 1.

peak(pp) = [argmax(view(pp.incidence, :, j)) - 1 for j in axes(pp.incidence, 2)]   # day of the peak
peaks = peak.(ppc_glct.pps)
(mean_peak_day = round.(mean.(peaks); digits = 1),
 observed_peak_day = [argmax(g.infection_counts) - 1 for g in groups],
 P_class1_after_class2 = mean(peaks[2] .>= peaks[3]),
 P_class0_after_class1 = mean(peaks[1] .>= peaks[2]))

# GLCT predictive checks (observed counts, posterior median and 95% band of the expected new
# infections):

ppc_plot(ppc, model) = plot([plot(pp; title = "$t: $model", color = c, xlabel = "day", ylabel = "new infections",
                                 legend = :topleft)
                            for (pp, t, c) in zip(ppc.pps, ["class 0", "class 1", "class 2"],
                                                  [:forestgreen, :steelblue, :tomato])]...;
                           layout = (1, 3), size = (1050, 330), left_margin = 4mm, bottom_margin = 4mm,
                           link = :y)
ppc_plot(ppc_glct, "GLCT")

# The Boxcar fit gives the same bands:

ppc_plot(ppc_boxcar, "Boxcar")

# ### Recovery intervals
#
# The posterior infectious-period distribution against the pooled recorded intervals, as a
# [`RecoveryPredictive`](@ref):

function recovery_check(post, hd; ndraws, max_interval = maximum(hd.delta) + 3)
    idx = unique(round.(Int, range(1, length(post[:r]); length = ndraws)))
    P = zeros(max_interval, length(idx))
    for (j, i) in enumerate(idx)
        r = Int(post[:r][i])
        dph = hagelloch_dph(hd, r, post[:mean_δ][i])
        P[:, j] .= [pdf(dph, x) for x in 1:max_interval]
    end
    return RecoveryPredictive(1:max_interval, P, [count(==(x), hd.delta) for x in 1:max_interval])
end

rp = recovery_check(post_glct, hd; ndraws = NPPC)
plot(rp; xlabel = "infectious period (days)", size = (600, 320), color = :steelblue)

# ## Substituting your own data or model
#
# - **Data.** Build one [`LineList`](@ref) per group on a common grid (the same `nsteps`), e.g.
#   with [`linelist`](@ref) from infection and removal times. Groups may share the recovery term
#   (as here) or each add their own.
# - **Mixing structure.** Change `mtheta_foi`: any function of the infected-fraction vector that
#   returns one rate per group, written so that it accepts ForwardDiff `Dual`s. For a linear
#   force of infection a matrix `B` can be passed instead. Keep the next-generation matrix used
#   for ``R_0`` consistent with it.
# - **Seeding.** Set the vector `ρ` of [`forward_map`](@ref) for your groups. Here it depends on
#   the summed-out ``N_2``, which is why ``N_2`` is a cell dimension.
# - **Populations.** A group whose size only enters its final-size term can be summed out with an
#   [`NGrid`](@ref) (any prior through its `logprior` keyword). A known size is passed as an
#   integer.
# - **Infectious period.** Any [`DPH`](@ref) works with `GLCT`, and `Boxcar(dph, K_age)` gives the
#   infection-age map. Integer parameters of the DPH go into the cells.
# - **Guards and terms.** `ddsa_loglikelihood(...; guards, terms)` selects the numerical guards
#   and the likelihood terms of each group.
#
# ## Runtime

disable_logging(Logging.BelowMinLevel) #hide
(mode = FULL ? "full (manuscript settings)" : "reduced (code check, not for results)",
 threads = Threads.nthreads(), seconds_total = round(time() - t_start; digits = 1),
 seconds_glct = round(t_glct; digits = 1), seconds_boxcar = round(t_boxcar; digits = 1))
