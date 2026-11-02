# # H1N1 influenza at Washington State University (2009)
#
# This vignette fits the 2009 H1N1 influenza outbreak at Washington State University (WSU).
# The same data were analysed with continuous-time DSA by KhudaBukhsh et al. (2020), who introduced
# the concept of Survival Dynamical Systems (SDS), and whose estimates are the point of comparison.
#
# **The question.** Given the infection times and recovery intervals of several cases in a closed 
# population, what are the transmission rate `β`, the removal rate `γ`, the initially infected fraction 
# `ρ`, and the size `N` of the population at risk? `N` is not known, but the campus about 18,000 students.
#
# **The model.** A discrete-time SIR map with a geometric infectious period on a daily grid, with
# - the final size `K ~ Binomial(N, τ)`, where `τ` is the expected fraction infected over the
#   observation window;
# - the infection-day histogram `~ Multinomial(K, f)`, where `f` is the distribution of infection
#   days given by the map;
# - geometric recovery intervals, right-censored for the 13 cases still infectious at day 104.

using DiscreteDynamicSurvivalAnalysis
using CSV, DataFrames, Distributions, Turing, Random, Statistics
using Plots, StatsPlots

const FULL = get(ENV, "DDSA_FULL", "false") == "true"
const REPRODUCTION = Dict{String,Dict{String,Any}}() #src
T_START = time() #hide
nothing #hide

# ## Inputs
#
# `WSU.csv` comes from KhudaBukhsh et al. (SDS.Epidemic repository): column `V1` holds the infection
# times and `V2` the recovery times, in days. The raw data were daily case counts, but KhudaBukhsh et al.
# spread the infection times within each day and *simulated* the recovery times from an exponential
# distribution with rate `1/5.5` per day, so the recovery intervals are synthetic (see `data/README.md`).
#
# The observation window is days 0–103: with `nsteps = 104` the infections of day `d` fall in step
# `d + 1`, and recoveries at or after day 104 are right-censored there.

df = CSV.read(example_data_path(:wsu), DataFrame)
data = linelist(df.V1, df.V2; nsteps = 104, censor_at = 104.0)

#-

bar(0:(data.nsteps - 1), data.infection_counts; label = "", xlabel = "day", ylabel = "cases",
    title = "WSU H1N1: infections per day", linewidth = 0)

# ## Model configuration
#
# The priors are `β ~ Uniform(0.01, 5)`, `γ ~ LogNormal(log(1/5.5), 0.3)` truncated to
# `[0.01, 1]` (centred on the rate used to simulate the recovery times), and `ρ ~ Beta(1, 10)`. `γ` is
# a rate per day: the per-day removal probability is `p = 1 − exp(−γ)`. `N` has a discrete uniform
# prior on `(K + 1):18000`, which is summed out of the likelihood.
# The infection-time distribution is floored at `1e-10` before the multinomial term.

K = data.final_size
spec = DDSAModel(infectious_period = GeometricIP(parameter = :γ),
                 priors = (β = Uniform(0.01, 5.0), γ = truncated(LogNormal(log(1 / 5.5), 0.3), 0.01, 1.0),
                           ρ = Beta(1.0, 10.0)),
                 population = (K + 1):18_000,
                 guards = NumericalGuards(f_floor = 1e-10))
nothing #hide

# ## Fit
#
# The manuscript ran 4 chains of 2,000 draws after 1,000 NUTS adaptation steps
# ([`default_sampler`](@ref)). The chains start from jittered data-based values ([`data_based_inits`](@ref)).
# After sampling, `N` is drawn once per draw from its exact conditional posterior and added to the
# chain.

ndraws, sampler = FULL ? (2000, default_sampler()) : (1000, NUTS(500, 0.8; max_depth = 12, Δ_max = 1000.0))
t_fit = @elapsed wsu_fit = fit_ddsa(Xoshiro(20261006), spec, data; ndraws = ndraws, nchains = 4, sampler = sampler)
wsu_fit

# ## Results
#
# Posterior summaries, with the derived quantities: `R0 = β / p` (the reproduction number of the
# discrete-time map), the mean infectious period `mean_ip_steps = 1/p` days, and the attack rate
# `K_over_N = K / N`.

function summary_table(f; sigdigits = 4)
    df = DataFrame(posterior_summary(f))
    for c in names(df)[2:end]
        df[!, c] = round.(df[!, c]; sigdigits = sigdigits)
    end
    return df
end
summary_table(wsu_fit)

# The convergence diagnostics are in the `ess` and `rhat` columns; trace plots of the sampled
# parameters and `N`:

plot(wsu_fit.chain[[:β, :γ, :ρ, :N]])

# **Comparison with KhudaBukhsh et al. (2020).** Their SDS analysis fixed `1/γ = 5.5` days and imputed
# `N` from the final-size relation. They report `N = 7,051` (90% interval 6,602–7,581), `β = 0.189`
# and `R0 = 1.06`. The 90% intervals here:

d = posterior_draws(wsu_fit)
DataFrame(quantity = ["N", "β", "γ", "ρ", "R0", "mean_ip_steps"],
          median = [median(x) for x in (d.N, d.β, d.γ, d.ρ, d.R0, d.mean_ip_steps)],
          q5 = [quantile(x, 0.05) for x in (d.N, d.β, d.γ, d.ρ, d.R0, d.mean_ip_steps)],
          q95 = [quantile(x, 0.95) for x in (d.N, d.β, d.γ, d.ρ, d.R0, d.mean_ip_steps)],
          KhudaBukhsh_2020 = ["7051 [6602, 7581]", "0.189", "1/5.5 (fixed)", "", "1.06", "5.5 (fixed)"])

# The manuscript's run gave `N = 14,144` (95% interval 10,359–17,727), `β = 0.190`, `R0 = 0.97`
# (0.94–1.03) and a mean infectious period of 5.12 days. `N` is larger than the SDS estimate because
# the two analyses treat it differently: here it enters the Binomial final-size term and is summed out
# over a wide uniform prior, so its posterior is broad.

# ## Posterior predictive checks
#
# The expected number of new infections per day, `N τ f`, re-running the map at every posterior draw
# (scale `:expected`), against the observed counts. Each draw also gives a replicate final size
# `K_rep ~ Binomial(N, τ)`.

pp = posterior_predictive(Xoshiro(2276), wsu_fit; ndraws = length(d.β), scale = :expected)
REPRODUCTION["h1n1"] = Dict("beta" => d.β, "gamma" => d.γ, "rho" => d.ρ, "N" => d.N, "R0" => d.R0, #src
                            "mean_ip_steps" => d.mean_ip_steps, "K_over_N" => d.K_over_N,         #src
                            "final_size_ppc" => pp.final_size)                                    #src
plot(pp; xlabel = "day", ylabel = "new infections", title = "WSU H1N1: posterior predictive")

# The map follows the overall decline of incidence but not the two early peaks (days 10 and 17).
#
# The final-size replicates are centred on the observed `K` (the draws of `N` come from
# `P(N | K, τ)`, so `N τ` matches `K` by construction; the replicates show the Binomial spread
# plus the posterior uncertainty in `N`):

histogram(pp.final_size; normalize = :pdf, bins = 60, label = "", xlabel = "final size",
          ylabel = "density", title = "Replicated final size")
vline!([K]; label = "observed K = $K", linewidth = 3, linestyle = :dot)

#-

fs_q = quantile(pp.final_size, [0.025, 0.5, 0.975])
(median = fs_q[2], lower = fs_q[1], upper = fs_q[3], observed = K)

# The fitted geometric infectious period against the recorded (uncensored) recovery intervals:

plot(recovery_predictive(wsu_fit; ndraws = 500); xlims = (0, 30), title = "Recovery intervals")

# The susceptible curve at the posterior medians (marked with hats), scaled to `N̂`, against the
# observed depletion `(1 − ρ̂) N̂ − Σ cases` (the comparison figure of KhudaBukhsh et al.). This
# uses the building blocks of the package: [`forward_map`](@ref) with a one-state [`GLCT`](@ref)
# structure.

β̂, γ̂, ρ̂, N̂ = median(d.β), median(d.γ), median(d.ρ), round(Int, median(d.N))
sol = forward_map(GLCT(dph_geometric(1 - exp(-γ̂))), β̂, ρ̂, data.nsteps)
S_obs = (1 - ρ̂) * N̂ .- [0; cumsum(data.infection_counts)]
plot(0:data.nsteps, N̂ .* susceptible_fraction(sol); label = "map (posterior medians)", linewidth = 3,
     xlabel = "day", ylabel = "susceptibles", title = "Susceptible curve")
plot!(0:data.nsteps, S_obs; label = "observed depletion", linewidth = 3, linestyle = :dash)

# Posterior densities of the population at risk and the sampled parameters (the panels of the
# manuscript figure):

plot(histogram(d.N; normalize = :pdf, bins = 40, label = "", xlabel = "N"),
     histogram(d.β; normalize = :pdf, bins = 40, label = "", xlabel = "β"),
     histogram(d.γ; normalize = :pdf, bins = 40, label = "", xlabel = "γ"),
     histogram(d.ρ; normalize = :pdf, bins = 40, label = "", xlabel = "ρ");
     layout = (1, 4), size = (1000, 250))

# ## Substituting your own data or model
#
# - **Data.** Any line list of infection and recovery times works: pass them to [`linelist`](@ref),
#   with `missing` for unknown times, `nsteps` for the observation window and `censor_at` for its end.
#   Cases still infectious at the end are right-censored.
# - **Population.** A known population is an integer (`population = 18_000`); an unknown one is a
#   range (uniform prior) or an [`NGrid`](@ref) with any prior on `N`.
# - **Infectious period.** `GeometricIP(parameter = :γ)` is the manuscript's model. A negative binomial
#   period with `r` stages summed out is `NegBinGLCT(1:10)` (with a prior on `p` or `mean_ip`).
# - **Priors and guards.** Change the `priors` named tuple; `NumericalGuards()` (the default) gives the
#   exact likelihood without the `f` floor.

# ## Runtime

(settings = FULL ? "manuscript (4 chains × 2,000 draws, 1,000 adaptation steps)" :
                   "reduced (4 chains × 1,000 draws, 500 adaptation steps; code check, not for results)",
 threads = Threads.nthreads(), sampling_seconds = round(t_fit; digits = 1),
 total_seconds = round(time() - T_START; digits = 1))
