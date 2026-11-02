# # Analysing your own line list
#
# This tutorial takes a case line list from a CSV file with calendar dates to a fitted DDSA
# model. The steps are: read the file, bin the dated cases with [`linelist`](@ref), choose a model with [`DDSAModel`](@ref), fit it with [`fit_ddsa`](@ref), and
# check the fit.
#
# ## The question
#
# Suppose you have a line list from an outbreak in a closed population: one row per case, with
# the date the case became infected (or showed symptoms, used as a proxy) and the date it stopped
# being infectious (recovery, isolation or hospitalisation). Some dates are missing, and some
# cases were still infectious when the data were exported. What are the transmission rate, the
# basic reproduction number `R0` and the infectious-period distribution, and how many people
# were at risk?
#
# To be able to check the answers, we simulate the outbreak ourselves with a stochastic
# chain-binomial model, which is not DDSA's own (deterministic, mean-field) working model, and
# write it out as a messy CSV file.

using DiscreteDynamicSurvivalAnalysis
using CSV, DataFrames, Dates, Distributions, Turing, Random, Statistics, Plots

const FULL = get(ENV, "DDSA_FULL", "false") == "true"
t_start = time()
nothing #hide

# ## Inputs: a simulated outbreak, written as a raw line list
#
# [`simulate_chain_binomial`](@ref) simulates `N` susceptibles and `M` initial infectives in
# daily steps. On each day every susceptible is infected with probability
# `1 − exp(−β I / (N + M))`, where `β` is the transmission rate per day and `I` the number of
# infectious cases. Each new case is infectious for `W` days, where `W` is drawn from
# the infectious-period distribution, here a negative binomial with `r = 3` stages and
# per-day progression probability `p = 0.5` (mean 6 days).

dph_true = dph_negative_binomial(3, 0.5)
β_true, N_true, M_true = 0.4, 1000, 5
sim = simulate_chain_binomial(Xoshiro(1), dph_true, β_true, N_true; M = M_true)
truth = (β = β_true, ρ = M_true / (N_true + M_true), p = 0.5, r = 3,
         mean_ip = mean(dph_true), R0 = β_true * mean(dph_true))
(final_size = sim.data.final_size, last_infection_day = maximum(sim.infection_steps) - 1, truth...)

# A case infected on step `k` was infected on day `k − 1`. We give every case a calendar date,
# then damage the data the way real data are damaged:
# - the data are exported four days after the last infection, so cases still infectious then
#   have no removal date yet, only the status `"infectious"`;
# - about 5% of the cases have no infection date;
# - about 8% have no removal date (status `"unknown"`, e.g. lost to follow-up).
#
# The `M = 5` index cases were infected before observation began. They are not in the line list;
# in the model they are the initially infected fraction `ρ`.

origin = Date(2024, 3, 4)                                   # start of surveillance: day 0
export_date = origin + Day(maximum(sim.infection_steps) + 3)
n = length(sim.infection_steps)
infection_date = Vector{Union{Date,Missing}}([origin + Day(k - 1) for k in sim.infection_steps])
removal_date = Vector{Union{Date,Missing}}([origin + Day(k - 1) for k in sim.removal_steps])
status = fill("removed", n)
rng = Xoshiro(2)
for i in 1:n
    if removal_date[i] >= export_date             # not yet removed when the data were exported
        removal_date[i] = missing
        status[i] = "infectious"
    elseif rand(rng) < 0.05
        infection_date[i] = missing               # infection date not recorded
    elseif rand(rng) < 0.08
        removal_date[i] = missing                 # removal date not recorded
        status[i] = "unknown"
    end
end
raw = DataFrame(case_id = 1:n, infection_date = infection_date, removal_date = removal_date, status = status)
path = joinpath(mktempdir(), "outbreak_linelist.csv")
CSV.write(path, raw)
foreach(println, first(readlines(path), 5))

# ## Read the file
#
# From here on we use only the file and the two dates that come with it: the start of
# surveillance and the export date. 

df = CSV.read(path, DataFrame)
first(df, 5)

#-
combine(groupby(df, :status), nrow => :cases)

# The *origin* is the date of day 0, the first day of the epidemic window. 
# Use the documented start of surveillance when you have it. The earliest recorded 
# infection date is a fallback, but here it would be one day late: the only case 
# infected on day 0 has no infection date in the file.
#
# A case that was still infectious at the export date must be *right-censored*, not treated as
# a case with an unknown removal date. [`linelist`](@ref) censors every removal at or after
# `censor_at`, so we give these cases the export date as their removal date.

day0 = Date(2024, 3, 4)
(day0 = day0, earliest_recorded_infection = minimum(skipmissing(df.infection_date)))

#-
removal = ifelse.(df.status .== "infectious", export_date, df.removal_date)
nothing #hide

# ## Bin the line list
#
# [`linelist`](@ref) turns event times into the binned data of the likelihood. With
# `origin = day0` it takes the dates directly and measures each time `t` in days since `day0`, so
# with `Δt = 1` one step is one day (`Δt = 7` would bin by week):
# - **infection step** `floor(t / Δt) + 1`: step `k` covers `[(k − 1)Δt, kΔt)`, so the cases of
#   day 0 are in step 1;
# - **recovery interval** (both times known): `round((t_removal − t_infection) / Δt)` steps,
#   with a minimum of 1 (a removal on the day of infection counts as one step);
# - **censoring**: with `censor_at`, a removal at or after `censor_at` gives a right-censored
#   interval of `round((censor_at − t_infection) / Δt)` steps, also with a minimum of 1 (the case
#   was still infectious then);
# - **`missing` infection time**: the case enters the *removal-only* term, which sums over its
#   possible infection days;
# - **`missing` removal time**: the case contributes its infection time only;
# - **final size** `K`: the number of rows.
#
# `censor_at` can be a date or a number of days since the origin. `nsteps`, the length of the
# epidemic window, is a number of steps. Here the window ends at the export date, so the grid
# covers days `0, …, t_export − 1`, where `t_export` is the export date in days since day 0.

t_export = Dates.value(export_date - day0)                 # days from day 0 to the export date
data = linelist(df.infection_date, removal; origin = day0, Δt = 1.0, nsteps = t_export,
                censor_at = export_date)

#-
(final_size = data.final_size, known_infection_times = sum(data.infection_counts),
 recovery_intervals = length(data.recovery_intervals), censored = count(data.censored),
 removal_only = length(data.removal_only_steps))

# `linelist` checks the data and throws an informative error instead of silently dropping rows:

function show_error(f)
    try
        f()
    catch err
        println(sprint(showerror, err))
    end
end
show_error(() -> linelist([2.0, 5.0], [8.0, 4.0]))                     # removal before infection
show_error(() -> linelist([2.0, missing], [8.0, missing]))             # no dates at all
show_error(() -> linelist([2.0, 41.0], [8.0, 44.0]; censor_at = 40.0)) # infected after the cut-off
show_error(() -> linelist([2.0, 5.0], [8.0]))                          # columns of different length
show_error(() -> linelist([Date(2024, 3, 1)], [Date(2024, 3, 9)]; origin = day0))   # before day 0
show_error(() -> linelist([Date(2024, 3, 6)], [Date(2024, 3, 9)]))                  # dates, no origin

# Fix such rows in the data (or drop them deliberately) before calling `linelist`. If your data
# are already binned, build the [`LineList`](@ref) directly with its keyword constructor, e.g.
# `LineList(counts; recovery_intervals = δ, censored = c, final_size = K)`.
#
# ## Model configuration
#
# A [`DDSAModel`](@ref) has four parts.
#
# **Infectious period** (`infectious_period`), which sets both the recovery term and how cases
# progress in the forward map:
# - `GeometricIP()`: a single stage (the classical discrete-time SIR map);
# - `NegBinGLCT(r_support)`: the negative binomial, with `r` summed out over `r_support`;
# - `NegBinBoxcar(r_support, K_age; truncation)`: the same distribution, propagated through
#   `K_age` infection-age classes.
#
# The `parameter` keyword chooses what is sampled: `:p`, `:γ` (a removal rate, geometric only) or
# `:mean_ip` (the mean infectious period).
#
# **Priors** (`priors`): a named tuple with one univariate distribution per sampled parameter:
# - transmission: `β`, or `R0` with `transmission = :R0` (this needs the `:mean_ip` parameter);
# - `ρ`;
# - the infectious-period parameter.
#
# **Population** (`population`), `N` in the final-size term `K ~ Binomial(N, τ)`, where `τ` is the
# expected fraction infected over the epidemic window:
# - a known integer;
# - a range `N_min:N_max`, for an unknown `N` with a uniform prior, summed out exactly;
# - an [`NGrid`](@ref), for any other prior on `N`.
#
# **Terms and guards**: `terms` selects the likelihood terms (`:final_size`, `:infection_times`,
# `:recovery`, `:removal_only`), and [`NumericalGuards`](@ref) clamps extreme map outputs. The
# default is the exact likelihood.
#
# Our main model has a known population, a negative binomial infectious period with
# `r ∈ 1:10`, and vague priors:

priors = (β = Uniform(0.01, 2.0), ρ = Beta(1, 100), p = Beta(1, 1))
spec = DDSAModel(infectious_period = NegBinGLCT(1:10), priors = priors, population = N_true)

# For comparison, some alternative specifications of the same data:

K = data.final_size
alternatives = [
    "unknown N, uniform on K:5000" =>
        (DDSAModel(infectious_period = NegBinGLCT(1:10), priors = priors, population = K:5000), truth),
    "unknown N, prior ∝ 1/N" =>
        (DDSAModel(infectious_period = NegBinGLCT(1:10), priors = priors,
                   population = NGrid(K, K, 5000; logprior = -log.(K:5000))), truth),
    "geometric, removal rate γ" =>
        (DDSAModel(infectious_period = GeometricIP(parameter = :γ), population = N_true,
                   priors = (β = Uniform(0.01, 2.0), ρ = Beta(1, 100), γ = Uniform(0.01, 2.0))),
         (β = 0.4, ρ = truth.ρ, γ = -log(1 - 1 / 6))),
    "NegBin, R0 and mean_ip" =>
        (DDSAModel(infectious_period = NegBinGLCT(1:10; parameter = :mean_ip), transmission = :R0,
                   priors = (R0 = Uniform(0.5, 6.0), mean_ip = Uniform(1.0, 20.0), ρ = Beta(1, 100)),
                   population = N_true), truth),
    "NegBin Boxcar, 60 age classes" =>
        (DDSAModel(infectious_period = NegBinBoxcar(1:10, 60), priors = priors, population = N_true), truth),
    "recovery intervals only" =>
        (DDSAModel(infectious_period = NegBinGLCT(1:10), priors = (p = Beta(1, 1),), terms = (:recovery,)), truth),
    "no recovery term, guarded τ" =>
        (DDSAModel(infectious_period = NegBinGLCT(1:10; parameter = :mean_ip), transmission = :R0,
                   priors = (R0 = Uniform(0.5, 6.0), mean_ip = Uniform(1.0, 20.0), ρ = Beta(1, 100)),
                   population = N_true, terms = (:final_size, :infection_times, :removal_only),
                   guards = NumericalGuards(τ_bounds = (0.001, 0.999))), truth),
]
nothing #hide

# [`turing_model`](@ref) turns a specification into a Turing model. Use it to evaluate the
# log-likelihood at given parameter values: only the parameters the model samples are used.

[(model = name, loglikelihood = loglikelihood(turing_model(s, data), pt[keys(s.priors)]))
 for (name, (s, pt)) in alternatives] |> DataFrame

# The model also returns the log-likelihood of each `r`. At the true parameters it peaks at the
# true `r = 3`; `r` above the shortest observed interval has zero likelihood (`-Inf`):

model = turing_model(spec, data)
ret = returned(model, truth[keys(priors)])
DataFrame(r = 1:10, loglikelihood = ret.lps, posterior_weight = posterior_weights(ret.lps))

# ## Fit
#
# [`fit_ddsa`](@ref) samples the continuous parameters with NUTS and then draws `r` (and `N`
# when it is unknown) from their exact conditional posteriors. 
# The manuscript settings (4 chains × 2,000 draws after 1,000 adaptation steps). 

ndraws, nchains = FULL ? (2000, 4) : (400, 2)
sampler = FULL ? default_sampler() : NUTS(300, 0.8)
fit = fit_ddsa(Xoshiro(3), spec, data; ndraws = ndraws, nchains = nchains, sampler = sampler)

# We also fit the model with an unknown population (uniform on `K:5000`):

spec_N = first(alternatives)[2][1]
fit_N = fit_ddsa(Xoshiro(4), spec_N, data; ndraws = ndraws, nchains = nchains, sampler = sampler)

# ## Results
#
# [`posterior_summary`](@ref) has one row per quantity. These include the derived ones: the mean
# infectious period `mean_ip = r Δt / p`, `R0 = β · mean_ip` and `τ`. It also reports the
# effective sample size `ess` and `rhat`. Check that `rhat` is close to 1
# and that `ess` is in the hundreds before reading the estimates.

summary_table(fit) = select(DataFrame(posterior_summary(fit)), :parameter, :mean, :q2_5, :q50, :q97_5, :ess, :rhat)
summary_table(fit)

# The 95% intervals against the values used in the simulation:

function compare_with_truth(fit, truth)
    d = posterior_draws(fit)
    DataFrame([(parameter = k, truth = Float64(v), median = median(d[k]), lower = quantile(d[k], 0.025),
                upper = quantile(d[k], 0.975), covered = quantile(d[k], 0.025) ≤ v ≤ quantile(d[k], 0.975))
               for (k, v) in pairs(truth) if haskey(d, k)])
end
compare_with_truth(fit, truth)

# The posterior of the number of stages `r`:

d = posterior_draws(fit)
DataFrame(r = 1:4, posterior_probability = [mean(d.r .== r) for r in 1:4])

# With an unknown population, `N` is estimated jointly with the other parameters. Here most of the
# population was infected, so the shape of the epidemic curve determines `τ`, and with it
# `N ≈ K / τ`, quite precisely. In a small outbreak (small `τ`) `N` is much less certain, and the
# posterior of `N` depends more on its prior:

compare_with_truth(fit_N, merge(truth, (N = N_true,)))

# ## Plots
#
# [`posterior_predictive`](@ref) re-runs the forward map at posterior draws. The bars count only
# the `K_known` cases with a known infection time (96% of the `K` cases here). By default
# (`scale = :expected_known`) the curve is the expected number of such cases, `(K_known / K) N τ f`,
# where `f` is the map's infection-time distribution, so it can be compared with the bars and
# checks both the size and the timing of the epidemic. With `scale = :expected` the curve is the
# expected number of new infections of all `K` cases, `N τ f`, so it lies above the bars.
# `scale = :observed_total` scales `f` to the number of known infection times, which checks the
# shape only. Loading Plots.jl makes `plot` work for these outputs.

pp = posterior_predictive(Xoshiro(5), fit; ndraws = 200)
pp_all = posterior_predictive(Xoshiro(5), fit; ndraws = 200, scale = :expected)
plot(plot(pp; title = "scale = :expected_known (default)", xlabel = "day"),
     plot(pp_all; title = "scale = :expected", xlabel = "day"),
     layout = (1, 2), size = (900, 350), bottom_margin = 5Plots.mm)

# Simulated final sizes `Binomial(N, τ)` against the observed `K`:

histogram(pp.final_size; bins = 30, label = "posterior predictive", xlabel = "final size K",
          ylabel = "draws", size = (500, 320))
vline!([pp.observed_final_size]; label = "observed", linewidth = 2)

# [`recovery_predictive`](@ref) gives the posterior infectious-period pmf with the observed
# (uncensored) recovery intervals:

rp = recovery_predictive(fit; ndraws = 200)
plot(rp; size = (600, 350), title = "infectious period")

# ## Substituting your own data or model
#
# A checklist for your own line list:
#
# 1. **One row per case**, from a closed population and one epidemic wave. Use the date of
#    infection or the best proxy (symptom onset), and the end of infectiousness (recovery,
#    isolation, hospitalisation or death).
# 2. **Day 0** is the start of the epidemic window. Pass dates with `origin` set to day 0, or
#    convert them to numbers of days (or weeks, with `Δt` in the same unit) since day 0. Negative
#    times are errors.
# 3. **Missing values**: use `missing` for unknown dates. A case still infectious at the end of
#    observation gets a removal time `≥ censor_at`, not `missing`.
# 4. **Window**: set `nsteps` to cover the observation period and `censor_at` to its end if
#    removals after it are unobserved.
# 5. **Population**: give `N` if it is known. Otherwise give a plausible range, or an
#    [`NGrid`](@ref) with your prior. `N` must be at least the number of cases.
# 6. **Infectious period**: start with `NegBinGLCT(1:10)`. Compare it with `GeometricIP()` if the
#    observed intervals look exponential-like. Use `NegBinBoxcar` when you want infection-age
#    classes.
# 7. **Priors** on the scale of your time unit: `β` is per unit time, and `mean_ip` is in time
#    units.
# 8. **Diagnostics**: check `rhat ≈ 1` and a large `ess` in [`posterior_summary`](@ref). Then
#    compare the posterior predictive curve and the infectious-period pmf with the data.
#
# Some analyses go beyond the single-population model, for example several groups with
# group-specific transmission, a different force of infection, or a different set of integer
# parameters to sum out. These use the building blocks:
# - [`forward_map`](@ref) runs the map for one group or for `G` groups, with a transmission
#   matrix or any force-of-infection function;
# - [`ddsa_loglikelihood`](@ref) and the term functions ([`final_size_loglik`](@ref),
#   [`infection_time_loglik`](@ref), [`recovery_loglik`](@ref), [`removal_only_loglik`](@ref))
#   evaluate the likelihood of a map solution;
# - [`marginal_r_loglik`](@ref) and [`marginal_N_loglik`](@ref) sum out integer parameters inside
#   your own Turing model.
#
# The [Hagelloch measles vignette](measles_hagelloch.md) builds a three-group model this way.
#
# ## Runtime

println("total runtime: ", round(time() - t_start; digits = 1), " s (",
        FULL ? "FULL" : "reduced", " settings: $nchains chains × $ndraws draws, $(Threads.nthreads()) threads)")
