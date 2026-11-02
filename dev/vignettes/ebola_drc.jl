# # Ebola virus disease in the DRC (2018–2020)
#
# This vignette fits the three waves of the 2018–2020 Ebola virus disease (EVD) epidemic in
# the Democratic Republic of the Congo, as in the manuscript. The same data were analysed with
# continuous-time DSA by Vossler et al. (2022), whose estimates are the point of comparison.
#
# **The question.** For each wave, what are the transmission rate `β`, the removal rate `γ`, the
# initially infected fraction `ρ` and the size `N` of the population at risk, when case records 
# are incomplete?
#
# **The data and the likelihood.** Each record has a symptom-onset time, a hospitalisation time
# (treated as removal), or both. Each kind of record contributes its own likelihood terms:
#
# | Record | Terms |
# |:--|:--|
# | complete (onset and hospitalisation) | onset histogram (Multinomial) + recovery interval (Geometric) + final size |
# | onset only | onset histogram + final size |
# | hospitalisation only | convolution over the unknown onset + final size |
#
# For a hospitalisation-only case removed on day `T`, the convolution term sums over the possible
# onset days, `log Σ_{u=1}^{T−1} f_u P(W = T − u)`, where `f_u` is the map's probability of an
# onset on day `u` and `W` is the infectious period. All `K` cases count in the final size
# `K ~ Binomial(N, τ)`, where `τ` is the expected fraction infected over the observation window, 
# and `N` is summed out over a uniform prior on `K, …, K + 19,999`.

using DiscreteDynamicSurvivalAnalysis
using CSV, DataFrames, Distributions, Turing, Random, Statistics
using Plots, StatsPlots

const FULL = get(ENV, "DDSA_FULL", "false") == "true"
const REPRODUCTION = Dict{String,Dict{String,Any}}() #src
const WAVES = FULL ? (1, 2, 3) : (2,)
T_START = time() #hide
nothing #hide

# ## Inputs
#
# Each wave's CSV has the columns `onset` and `hosp` (days since the first case) and the flags
# `event1`/`event2` marking an observed onset/hospitalisation. When a flag is 0 the corresponding
# column repeats the other time, so it is set to `missing` before binning. [`linelist`](@ref) then
# bins onsets by calendar day (the grid ends on the day of the last onset), takes the rounded
# onset-to-hospitalisation interval of complete records, and sends hospitalisation-only records to the
# convolution term.

function load_wave(w)
    df = CSV.read(example_data_path(Symbol("ebola_wave$w")), DataFrame)
    onset = ifelse.(df.event1 .== 1, df.onset, missing)
    hosp = ifelse.(df.event2 .== 1, df.hosp, missing)
    return linelist(onset, hosp)
end
waves = Dict(w => load_wave(w) for w in 1:3)

DataFrame(wave = 1:3, end_of_data = ["27 Feb 2019", "27 May 2019", "12 Sep 2019"],
          records = [waves[w].final_size for w in 1:3],
          complete = [length(waves[w].recovery_intervals) for w in 1:3],
          onset_only = [sum(waves[w].infection_counts) - length(waves[w].recovery_intervals) for w in 1:3],
          hosp_only = [length(waves[w].removal_only_steps) for w in 1:3],
          days = [waves[w].nsteps for w in 1:3])

# ## Model configuration
#
# The priors are `β ~ Uniform(0.01, 1)`, `γ ~ Uniform(0.01, 1)` (per day; the
# per-day removal probability is `p = 1 − exp(−γ)`) and `ρ ~ Beta(1, 500)`. The infectious
# period is geometric (1-indexed), the discrete analogue of the exponential period of continuous-time
# DSA. The numerical guards clamp `τ` to `[1e-6, 1 − 1e-6]` and floor `f` at `1e-10`.

ebola_spec(data) = DDSAModel(infectious_period = GeometricIP(parameter = :γ),
                             priors = (β = Uniform(0.01, 1.0), γ = Uniform(0.01, 1.0), ρ = Beta(1.0, 500.0)),
                             population = data.final_size:(data.final_size + 19_999),
                             guards = NumericalGuards(τ_bounds = (1e-6, 1 - 1e-6), f_floor = 1e-10))
nothing #hide

# ## Fit
#
# The manuscript ran 4 chains of 2,000 draws after 1,000 NUTS adaptation steps
# ([`default_sampler`](@ref)) for each wave. The chains start from jittered data-based values
# ([`data_based_inits`](@ref): `p` from the mean recovery interval, `β` at `R0 = 1`, `ρ` at its prior
# median).

ndraws, sampler = FULL ? (2000, default_sampler()) : (1000, NUTS(500, 0.8; max_depth = 12, Δ_max = 1000.0))
fits = Dict{Int,Any}()
seconds = Dict{Int,Float64}()
for w in WAVES
    seconds[w] = @elapsed fits[w] = fit_ddsa(Xoshiro(20261006), ebola_spec(waves[w]), waves[w];
                                             ndraws = ndraws, nchains = 4, sampler = sampler)
    d = posterior_draws(fits[w]) #src
    REPRODUCTION["ebola_w$w"] = Dict("beta" => d.β, "gamma" => d.γ, "rho" => d.ρ, "N" => d.N, "R0" => d.R0, #src
                                     "mean_ip_steps" => d.mean_ip_steps, "I0" => d.ρ .* d.N,             #src
                                     "K_over_N" => d.K_over_N)                                         #src
end
fits[last(WAVES)]

# ## Results
#
# Posterior summaries of Wave 3. The derived quantities are `p`, `R0 = β / p` (the reproduction
# number of the discrete-time map), the mean infectious period `mean_ip_steps = 1/p` days and the
# attack rate `K_over_N = K / N`.

function summary_table(f; sigdigits = 4)
    df = DataFrame(posterior_summary(f))
    for c in names(df)[2:end]
        df[!, c] = round.(df[!, c]; sigdigits = sigdigits)
    end
    return df
end
summary_table(fits[last(WAVES)])

# The implied number of initial infectives `I0 = ρ N` (posterior median and 95% interval) for each
# fitted wave:

i0(w) = (d = posterior_draws(fits[w]); quantile(d.ρ .* d.N, [0.5, 0.025, 0.975]))
DataFrame(wave = collect(WAVES), I0_median = [round(i0(w)[1]; digits = 2) for w in WAVES],
          I0_lower = [round(i0(w)[2]; digits = 2) for w in WAVES], I0_upper = [round(i0(w)[3]; digits = 2) for w in WAVES])

# **Comparison with Vossler et al. (2022).** Their continuous-time removal rate `γ` is compared with
# DDSA's `p`, their `β/γ` with `R0` and their `1/γ` with the mean infectious period.

const VOSSLER = Dict(1 => (β = 0.190, γ = 0.169, ρ = 0.00021, R0 = 1.124, ip = 5.9),
                     2 => (β = 0.217, γ = 0.179, ρ = 0.0054, R0 = 1.214, ip = 5.6),
                     3 => (β = 0.235, γ = 0.214, ρ = 0.0067, R0 = 1.098, ip = 4.7))
fmt(x) = (q = quantile(x, [0.5, 0.025, 0.975]); "$(round(q[1]; sigdigits = 3)) [$(round(q[2]; sigdigits = 3)), $(round(q[3]; sigdigits = 3))]")
rows = map(Iterators.product((:β, :p, :ρ, :R0, :mean_ip_steps), WAVES)) do (q, w)
    d = posterior_draws(fits[w])
    v = VOSSLER[w]
    ref = q === :β ? v.β : q === :p ? v.γ : q === :ρ ? v.ρ : q === :R0 ? v.R0 : v.ip
    (wave = w, DDSA = string(q), median_95 = fmt(getproperty(d, q)),
     Vossler_quantity = q === :p ? "γ" : q === :R0 ? "β/γ" : q === :mean_ip_steps ? "1/γ" : string(q),
     Vossler_2022 = ref)
end
DataFrame(vec(rows))

# The manuscript's estimates were close to Vossler et al. for every wave, e.g. Wave 2:
# `β = 0.222` (0.208–0.236), `γ = 0.202` (0.188–0.217), i.e. `p = 0.183` (0.171–0.195), and
# `R0 = 1.215` (1.169–1.261).
#
# Posterior densities, with Vossler et al.'s estimates (dashed):

panels = Any[]
for w in WAVES
    d = posterior_draws(fits[w])
    v = VOSSLER[w]
    for (x, ref, lab) in ((d.β, v.β, "β"), (d.p, v.γ, "p = 1 − exp(−γ)"), (d.ρ, v.ρ, "ρ"), (d.N, nothing, "N"))
        p = histogram(x; normalize = :pdf, bins = 40, label = "", xlabel = lab, title = "Wave $w", titlefontsize = 9)
        ref === nothing || vline!(p, [ref]; label = "", color = :black, linestyle = :dash, linewidth = 2)
        push!(panels, p)
    end
end
plot(panels...; layout = (length(WAVES), 4), size = (1000, 260 * length(WAVES)))

# ## Posterior predictive checks
#
# `f` at the posterior draws, scaled to the number of cases with a known onset
# (scale `:observed_total`), against the onset histogram: the manuscript's epidemic-curve figure
# (one panel per wave).

ppc_plots = map(WAVES) do w
    pp = posterior_predictive(Xoshiro(1234 + w), fits[w]; ndraws = 500, scale = :observed_total)
    plot(pp; title = "Wave $w", xlabel = "days since first case", ylabel = "cases", legend = :topleft)
end
plot(ppc_plots...; layout = (1, length(WAVES)), size = (350 * length(WAVES) + 100, 320))

# The fitted geometric infectious period against the recorded onset-to-hospitalisation intervals of
# the complete records:

rp_plots = [plot(recovery_predictive(fits[w]; ndraws = 500); title = "Wave $w", xlims = (0, 30)) for w in WAVES]
plot(rp_plots...; layout = (1, length(WAVES)), size = (350 * length(WAVES) + 100, 300))

# ## Substituting your own data or model
#
# - **Data.** Pass onset and removal times to [`linelist`](@ref), with `missing` for the unobserved
#   ones: complete records give recovery intervals, onset-only records enter the onset histogram, and
#   removal-only records enter the convolution term (`:removal_only`). Records with both times missing
#   are rejected, so drop them (or count them in the final size by building the [`LineList`](@ref)
#   directly with a larger `final_size`).
# - **Population.** `population = K:(K + 19_999)` puts a uniform prior on `N`. Use an integer for a
#   known population, or an [`NGrid`](@ref) for another prior.
# - **Infectious period.** `GeometricIP(parameter = :γ)` matches continuous-time DSA. For a more
#   peaked period use `NegBinGLCT(1:10)` (with a prior on `p` or `mean_ip`); its number of stages is
#   summed out.
# - **Likelihood terms.** `terms = (:final_size, :infection_times, :recovery)` drops the convolution
#   term, e.g. to check how much the hospitalisation-only records matter.

# ## Runtime

DataFrame(wave = collect(WAVES), sampling_seconds = [round(seconds[w]; digits = 1) for w in WAVES])

#-

(settings = FULL ? "manuscript (4 chains × 2,000 draws, 1,000 adaptation steps; waves 1–3)" :
                   "reduced (4 chains × 1,000 draws, 500 adaptation steps; wave 2; code check, not for results)",
 threads = Threads.nthreads(), total_seconds = round(time() - T_START; digits = 1))
