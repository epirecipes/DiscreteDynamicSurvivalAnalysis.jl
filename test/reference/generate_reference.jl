# Generates the regression fixtures in test/reference/*.toml by running the notebook implementation of the
# manuscript analyses (commit 8ea70c9 of github.com/epirecipes/DDSA, formerly DiscreteDynamicSurvivalAnalysis_DDSA), which
# DDSA_ORIGINAL_REPO must point to a checkout of:
# the notebook function definitions and data cells are extracted verbatim from the original .qmd sources
# (by function name or by a unique anchor string) and evaluated together with the original
# PhaseTypeDistributions.jl and IntegerParameters.jl, in the original repository's environment.
#
# The original repository is only read (CSV files and sources); nothing is written there.
#
# Usage (from this package's root; Julia 1.11, single thread):
#   DDSA_ORIGINAL_REPO=/path/to/DiscreteDynamicSurvivalAnalysis_DDSA \
#       julia --project=$DDSA_ORIGINAL_REPO test/reference/generate_reference.jl [outdir]
#
# Fixtures were generated from commit 8ea70c9 of github.com/epirecipes/DDSA
# (Julia 1.11.6, Turing 0.49.0, Distributions 0.25.131); a different commit only triggers a warning.

using Turing, Distributions, Random, Statistics, LinearAlgebra, CSV, DataFrames, MCMCChains, TOML
using StatsFuns: binomlogpdf

const ORIG = get(ENV, "DDSA_ORIGINAL_REPO",
                 normpath(joinpath(@__DIR__, "..", "..", "..", "DiscreteDynamicSurvivalAnalysis_DDSA")))
const OUT = length(ARGS) >= 1 ? ARGS[1] : @__DIR__
const EXPECTED_COMMIT = "8ea70c9"
isdir(ORIG) || error("original repository not found at $ORIG (set DDSA_ORIGINAL_REPO)")
const COMMIT = try readchomp(`git -C $ORIG rev-parse --short HEAD`) catch; "unknown" end
COMMIT == EXPECTED_COMMIT || @warn "original repository is at commit $COMMIT, fixtures were made at $EXPECTED_COMMIT"

# ------------------------------------------------------------------ verbatim code extraction from the .qmd sources
source(rel) = read(joinpath(ORIG, rel), String)

"Code of every ```{julia} cell of a .qmd file (Quarto `#|` option lines removed)."
function julia_cells(text)
    cells = String[]; buf = String[]; incell = false
    for line in split(text, '\n')
        if !incell && startswith(line, "```{julia}")
            incell = true; empty!(buf)
        elseif incell && startswith(line, "```")
            incell = false; push!(cells, join(buf, '\n'))
        elseif incell && !startswith(line, "#|")
            push!(buf, line)
        end
    end
    return cells
end

"The unique code cell of `rel` that contains `anchor`."
function cell_with(rel, anchor)
    cs = filter(c -> occursin(anchor, c), julia_cells(source(rel)))
    length(cs) == 1 || error("anchor $(repr(anchor)) matches $(length(cs)) cells in $rel")
    return cs[1]
end

"The top-level definition `function name` / `@model function name` (to its closing `end` at column 0)."
function definition(rel, name)
    lines = split(source(rel), '\n')
    pat = Regex("^(@inline |@model )?function \\Q$(name)\\E\\(")
    starts = findall(l -> occursin(pat, l), lines)
    length(starts) == 1 || error("definition $name found $(length(starts)) times in $rel")
    i = starts[1]
    j = findnext(l -> l == "end" || l == "end;", lines, i)
    return join(lines[i:j], '\n')
end

"`n` lines starting at the unique line that begins with `anchor`."
function lines_from(rel, anchor, n)
    lines = split(source(rel), '\n')
    i = findall(l -> startswith(l, anchor), lines)
    length(i) == 1 || error("line anchor $(repr(anchor)) found $(length(i)) times in $rel")
    return join(lines[i[1]:(i[1] + n - 1)], '\n')
end

"A fresh module with the packages the notebooks load and the original shared modules included."
function notebook_module(name)
    m = Module(Symbol(name))
    Core.eval(m, :(using Turing, Distributions, Random, Statistics, LinearAlgebra, CSV, DataFrames, MCMCChains))
    Core.eval(m, :(using StatsFuns: binomlogpdf))
    Base.include(m, joinpath(ORIG, "PhaseTypeDistributions.jl"))
    Base.include(m, joinpath(ORIG, "IntegerParameters.jl"))
    Core.eval(m, :(using .IntegerParameters))
    return m
end
run!(m, code; dir = ORIG) = cd(() -> Base.include_string(m, code), dir)
defs!(m, rel, names) = foreach(n -> run!(m, definition(rel, n)), names)

# ------------------------------------------------------------------ helpers for the fixtures
flat(M::AbstractMatrix) = Dict("size" => collect(size(M)), "data" => vec(Float64.(M)))
fl(x) = Float64.(collect(x))
nt2dict(nt) = Dict(String(k) => v for (k, v) in pairs(nt))
function model_values(m, model, points)
    lj = Float64[]; ll = Float64[]; rets = Any[]
    for pt in points
        push!(lj, Base.invokelatest(Turing.logjoint, model, pt))
        push!(ll, Base.invokelatest(Turing.loglikelihood, model, pt))
        r = Base.invokelatest(returned, model, pt)
        push!(rets, r isa Real ? Float64(r) : fl(r))
    end
    return Dict("points" => [nt2dict(asciinames(p)) for p in points], "logjoint" => lj,
                "loglikelihood" => ll, "returned" => rets)
end
const ASCII = Dict(:β => "beta", :γ => "gamma", :ρ => "rho", :nb_p => "p", :R₀ => "R0", :mean_ip => "mean_ip",
                   :mean_δ => "mean_delta", :beta_C1 => "beta_C1", :beta_C2 => "beta_C2", :beta_H => "beta_H",
                   :beta_G => "beta_G")
asciinames(nt) = (; (Symbol(ASCII[k]) => v for (k, v) in pairs(nt))...)
writetoml(name, d) = open(io -> TOML.print(io, d; sorted = true), joinpath(OUT, name), "w")
meta() = Dict("original_commit" => COMMIT, "julia" => string(VERSION),
              "generated_by" => "test/reference/generate_reference.jl")

# =================================================================== 1. PhaseTypeDistributions.jl
P = notebook_module("PTDref"); PT = P.PhaseTypeDistributions
dph_cases = [
    "geometric_0.3"               => () -> PT.dph_geometric(0.3),
    "negbin_4_0.4"                => () -> PT.dph_negative_binomial(4, 0.4),
    "negbin_1_0.7"                => () -> PT.dph_negative_binomial(1, 0.7),
    "negbin_6_0.85"               => () -> PT.dph_negative_binomial(6, 0.85),
    "erlang_from_mean_4_4.0_1.0"  => () -> PT.dph_erlang_from_mean(4, 4.0, 1.0),
    "erlang_3_2.0_0.1"            => () -> PT.dph_erlang(3, 2.0, 0.1),
    "erlang_from_rates_3_1.5_0.5" => () -> PT.dph_erlang_from_rates(3, 1.5, 0.5),
    "exponential_2.0_0.1"         => () -> PT.dph_exponential(2.0, 0.1),
    "negbin_from_rates_3_0.4_0.5" => () -> PT.dph_negative_binomial_from_rates(3, 0.4, 0.5),
    "negbin_from_mean_5_12.5"     => () -> PT.dph_negative_binomial_from_mean(5, 12.5),
    "min_support_3_0.45_5"        => () -> PT.dph_negative_binomial_min_support(3, 0.45, 5),
    "delay_2_negbin_2_0.5"        => () -> PT.dph_prepend_deterministic_delay(PT.dph_negative_binomial(2, 0.5), 2),
    "from_rates_general"          => () -> PT.dph_from_rates([0.6, 0.4], [-2.0 1.0; 0.5 -1.5], 0.3),
    "custom"                      => () -> PT.DPH([0.5, 0.5], [0.2 0.3; 0.1 0.6]),
]
ptd = Dict{String,Any}("meta" => meta(), "n" => collect(1:40))
for (name, mk) in dph_cases
    d = mk()
    ptd["dph_" * name] = Dict("alpha" => fl(d.α), "S" => flat(d.S), "s" => fl(d.s),
        "pmf" => [PT.pmf(d, n) for n in 1:40], "logpdf" => [Distributions.logpdf(d, n) for n in 1:40],
        "cdf" => [PT.cdf(d, n) for n in 1:40], "mean" => PT.mean(d), "var" => PT.var(d))
end
for (name, d) in ("negbin_4_0.4" => PT.dph_negative_binomial(4, 0.4),
                  "erlang_from_mean_4_4.0_1.0" => PT.dph_erlang_from_mean(4, 4.0, 1.0),
                  "from_rates_general" => PT.dph_from_rates([0.6, 0.4], [-2.0 1.0; 0.5 -1.5], 0.3))
    rng = Xoshiro(2024)
    ptd["rand_" * name] = Dict("seed" => 2024, "draws" => [rand(rng, d) for _ in 1:50])
end
xs = [0.1, 0.5, 1.0, 2.0, 5.0]
for (name, d) in ("general_3" => PT.PhaseType([0.5, 0.3, 0.2], [-1.0 0.5 0.0; 0.0 -2.0 1.0; 0.0 0.0 -1.5]),
                  "exponential_2.0" => PT.ph_exponential(2.0), "erlang_3_1.5" => PT.ph_erlang(3, 1.5),
                  "erlang_by_mean_4_2.0" => PT.ph_erlang_by_mean(4, 2.0))
    ptd["ph_" * name] = Dict("alpha" => fl(d.α), "A" => flat(d.A), "q" => fl(d.q), "x" => xs,
        "pdf" => [Distributions.pdf(d, x) for x in xs], "logpdf" => [Distributions.logpdf(d, x) for x in xs],
        "cdf" => [Distributions.cdf(d, x) for x in xs], "logcdf" => [Distributions.logcdf(d, x) for x in xs],
        "mean" => Distributions.mean(d), "var" => Distributions.var(d))
end
for (name, (r, p)) in ("3.7_0.4" => (3.7, 0.4), "1.0_0.5" => (1.0, 0.5), "2.25_0.8" => (2.25, 0.8))
    sd = PT.dph_negative_binomial_smooth(r, p)
    ptd["smooth_" * name] = Dict("k" => length(sd.dph_k.α), "w" => Float64(sd.w),
        "pmf" => [PT.pmf(sd, n) for n in 1:40], "logpdf" => [Distributions.logpdf(sd, n) for n in 1:40],
        "mean" => PT.mean(sd), "var" => PT.var(sd))
end
nb_args = [(k, r, p) for k in 1:12 for r in (1, 2, 4) for p in (0.2, 0.55, 0.9)]
ptd["negbin_total_logpdf_int"] = Dict("k" => first.(nb_args), "r" => getindex.(nb_args, 2),
    "p" => last.(nb_args), "value" => [PT.negbin_total_logpdf(a...) for a in nb_args])
ptd["negbin_total_logpdf_real"] = Dict("k" => collect(3:12), "r" => 2.5, "p" => 0.4,
    "value" => [PT.negbin_total_logpdf(k, 2.5, 0.4) for k in 3:12])
ptd["rate_to_proportion"] = Dict("r" => [0.0, 0.1, 1.0, 3.0], "t" => [1.0, 1.0, 0.5, 2.0],
    "value" => [PT.rate_to_proportion(r, t) for (r, t) in zip([0.0, 0.1, 1.0, 3.0], [1.0, 1.0, 0.5, 2.0])])

# Boxcar hazards: the data-availability (log-scale NegBin) and Hagelloch (matrix iteration) versions
DAh = notebook_module("DAhazard"); DAQ = "Data_avail_GLCT_Boxcar_Geo/ddsa_data_avail.qmd"
defs!(DAh, DAQ, ["rate_to_proportion", "hazard_vector_negbin"])
HAh = notebook_module("HAGhazard"); HAQ = "Measles_Hagelloch_1861/dsa_hagelloch_GLCTvsBoxcar.qmd"
defs!(HAh, HAQ, ["hazard_vector_from_dph"])
ptd["hazard_negbin"] = [Dict("r" => r, "p" => p, "K" => K,
                             "h" => fl(Base.invokelatest(DAh.hazard_vector_negbin, r, p, K)))
                        for (r, p, K) in ((4, 0.6, 15), (1, 0.3, 30), (3, 0.45, 40), (2, 0.05, 204))]
ptd["hazard_from_dph"] = [Dict("r" => r, "p" => p, "delta_min" => δ, "K" => K,
        "h" => fl(Base.invokelatest(HAh.hazard_vector_from_dph,
                                    HAh.PhaseTypeDistributions.dph_negative_binomial_min_support(r, p, δ), K)))
    for (r, p, δ, K) in ((4, 0.479, 4, 40), (2, 0.3, 4, 40), (1, 0.6, 1, 25))]
writetoml("phasetype.toml", ptd)
println("wrote phasetype.toml")

# =================================================================== 2. IntegerParameters.jl
IP = P.IntegerParameters
ipd = Dict{String,Any}("meta" => meta())
g = IP.NGrid(91, 91, 99)
ipd["ngrid_91"] = Dict("K" => 91, "N_min" => 91, "N_max" => 99, "logC" => g.logC, "logprior" => g.logprior,
    "tau" => [0.5, 0.9, 0.95], "marginal" => [IP.marginal_N_loglik(g, τ) for τ in (0.5, 0.9, 0.95)],
    "marginal_log" => [IP.marginal_N_loglik(g, log(τ), log1p(-τ)) for τ in (0.5, 0.9, 0.95)],
    "weights_0.9" => IP.N_posterior_weights(g, 0.9))
lp = log.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
gp = IP.NGrid(91, 91, 99; logprior = lp)
ipd["ngrid_91_prior"] = Dict("logprior_in" => lp, "logprior" => gp.logprior,
    "marginal_0.9" => IP.marginal_N_loglik(gp, 0.9))
gw = IP.NGrid(2276, 2277, 18000)
rng = Xoshiro(7)
ipd["sample_N_wsu"] = Dict("K" => 2276, "N_min" => 2277, "N_max" => 18000, "tau" => 0.16, "seed" => 7,
    "draws" => [IP.sample_N(rng, gw, 0.16) for _ in 1:20], "marginal" => IP.marginal_N_loglik(gw, 0.16))
f_r(r) = -0.5 * (r - 3.2)^2 + (r == 7 ? -Inf : 0.0)
logm, lps = IP.marginal_r_loglik(f_r, 1:8)
logm_p, _ = IP.marginal_r_loglik(f_r, 1:8; logprior = log.(1:8))
rng = Xoshiro(9)
ipd["marginal_r"] = Dict("lps" => lps, "logmarg" => logm, "logmarg_prior" => logm_p,
    "weights" => IP.posterior_weights(lps), "weights_prior" => IP.posterior_weights(lps; logprior = log.(1:8)),
    "seed" => 9, "draws" => [IP.sample_index(rng, lps) for _ in 1:30])
writetoml("integer_parameters.toml", ipd)
println("wrote integer_parameters.toml")

# =================================================================== 3. Notebook data, forward maps and likelihoods
dat = Dict{String,Any}("meta" => meta()); maps = Dict{String,Any}("meta" => meta())
lik = Dict{String,Any}("meta" => meta())

# ---- H1N1 (WSU)
HQ = "H1N1_WSU/dsa_WSU_Turing.qmd"; H = notebook_module("H1N1")
defs!(H, HQ, ["rate_to_proportion", "time_counts", "prep_sds", "load_real_data", "sir_map!", "solve_map",
              "simulate", "dsa_discrete_turing_multinomial_inferN"])
run!(H, cell_with(HQ, "real_data = load_real_data("); dir = joinpath(ORIG, "H1N1_WSU"))
run!(H, cell_with(HQ, "N_max = 18000"))
dat["h1n1"] = Dict("K" => H.K, "nsteps" => H.nsteps, "th" => H.th, "delta" => H.delta, "cens" => H.cens,
                   "N_min" => H.K + 1, "N_max" => H.N_max)
let (sol, τ, f) = Base.invokelatest(H.simulate, 0.19, 0.2172, 0.0257, H.nsteps)
    maps["h1n1_sir_map"] = Dict("beta" => 0.19, "gamma" => 0.2172, "rho" => 0.0257, "nsteps" => H.nsteps,
                                "tau" => τ, "f" => fl(f), "S" => fl(sol[1, :]))
end
hmodel = Base.invokelatest(H.dsa_discrete_turing_multinomial_inferN, H.N_grid, H.nsteps, H.K, H.th, H.delta, H.cens)
lik["h1n1"] = model_values(H, hmodel, [(β = 0.19, γ = 0.2172, ρ = 0.0257), (β = 0.3, γ = 0.25, ρ = 0.05),
                                       (β = 0.12, γ = 0.5, ρ = 0.003)])
println("H1N1 done")

# ---- Ebola waves 1–3
ebola_points = Dict(1 => (β = 0.1906, γ = 0.1857, ρ = 0.00022), 2 => (β = 0.2223, γ = 0.2022, ρ = 0.0053),
                    3 => (β = 0.2371, γ = 0.2420, ρ = 0.0067))
for w in 1:3
    EQ = "Ebola_DRC_2018/DSA_turing_SIRmap_W$w.qmd"; E = notebook_module("Ebola$w")
    defs!(E, EQ, ["logaddexp", "rate_to_proportion", "time_counts", "sir_map!", "solve_map", "simulate",
                  "dsa_ebola_allrecords"])
    run!(E, cell_with(EQ, "df_all = CSV.read("); dir = joinpath(ORIG, "Ebola_DRC_2018"))
    dat["ebola_w$w"] = Dict("K_total" => E.K_total, "K_onset" => E.K_onset, "nsteps" => E.nsteps,
        "th_onset" => E.th_onset, "delta_complete" => E.delta_complete, "T_hosp_only" => E.T_hosp_only,
        "N_min" => E.N_min, "N_max" => E.N_min + 19_999)
    pt = ebola_points[w]
    let (sol, τ, f) = Base.invokelatest(E.simulate, pt.β, pt.γ, pt.ρ, E.nsteps)
        maps["ebola_w$(w)_sir_map"] = Dict("beta" => pt.β, "gamma" => pt.γ, "rho" => pt.ρ, "nsteps" => E.nsteps,
                                           "tau" => τ, "f" => fl(f))
    end
    emodel = Base.invokelatest(E.dsa_ebola_allrecords, E.nsteps, E.K_total, E.K_onset, E.th_onset,
                               E.delta_complete, E.T_hosp_only, E.N_grid)
    lik["ebola_w$w"] = model_values(E, emodel, [pt, (β = 0.3, γ = 0.25, ρ = 0.01), (β = 0.15, γ = 0.4, ρ = 0.001)])
    println("Ebola W$w done")
end

# ---- Data-availability scenarios (synthetic)
D = notebook_module("DataAvail")
Core.eval(D, :(Random.seed!(1234)))
run!(D, cell_with(DAQ, "β_prior_hi = 2.5"))
defs!(D, DAQ, ["safe_normalise", "rate_to_proportion", "hazard_vector_negbin", "sir_boxcar_map!",
               "simulate_boxcar", "simulate_discrete_glct", "generate_discrete_glct_data", "dph_full_loglik",
               "dph_full", "dph_infonly_loglik", "dph_infonly", "dph_reconly", "geo_reconly", "geo_full",
               "geo_infonly", "boxcar_full_loglik", "boxcar_full", "boxcar_infonly_loglik", "boxcar_infonly"])
run!(D, cell_with(DAQ, "data_fwd = generate_discrete_glct_data("))
dat["data_availability"] = Dict("seed" => 1234, "N" => D.N_sim, "M" => D.M_sim, "rho" => D.ρ_sim,
    "beta" => D.β_true, "k" => D.k, "mu_ip" => D.μ_ip, "dt" => D.Δt_nom, "nsteps" => D.nsteps,
    "K" => D.K_obs, "th" => D.th_obs, "delta" => D.delta_obs, "tau" => D.τ_true_det, "f" => fl(D.f_true_det),
    "K_age" => D.K_age, "K_age_B" => D.K_age_B, "nb_r_max_data" => D.nb_r_max_data,
    "EW" => D.nb_mean, "R0_imp" => D.R0_imp)
PTD_D = D.PhaseTypeDistributions
let dn = D.dph_nom, r = Base.invokelatest(D.simulate_discrete_glct, dn.S, dn.α, 0.5, 0.01, 40, 1.0)
    maps["glct_erlang4"] = Dict("beta" => 0.5, "rho" => 0.01, "nsteps" => 40, "dt" => 1.0,
                                "tau" => Float64(r.τ), "f" => fl(r.f), "S" => fl(r.S))
end
for (name, d) in ("glct_negbin_4_0.6" => PTD_D.dph_negative_binomial(4, 0.6),
                  "glct_geometric_0.2" => PTD_D.dph_geometric(0.2))
    r = Base.invokelatest(D.simulate_discrete_glct, d.S, d.α, 0.7, 0.02, 40, 1.0)
    maps[name] = Dict("beta" => 0.7, "rho" => 0.02, "nsteps" => 40, "dt" => 1.0, "tau" => Float64(r.τ),
                      "f" => fl(r.f), "S" => fl(r.S))
end
let h = Base.invokelatest(D.hazard_vector_negbin, 4, 0.6, 15)
    sol, τ, f = Base.invokelatest(D.simulate_boxcar, 0.5, h, 0.01, 40, 15, 1.0)
    maps["boxcar_negbin_4_0.6_K15"] = Dict("beta" => 0.5, "rho" => 0.01, "nsteps" => 40, "K_age" => 15,
        "r" => 4, "p" => 0.6, "tau" => τ, "f" => fl(f), "S" => fl(sol[1, :]))
end
N_, M_, ns_, K_, th_, de_ = D.N_sim, D.M_sim, D.nsteps, D.K_obs, D.th_obs, D.delta_obs
ptsA = [(β = 0.489, ρ = 0.01, nb_p = 0.627), (β = 0.6, ρ = 0.02, nb_p = 0.5), (β = 0.3, ρ = 0.005, nb_p = 0.8)]
ptsB = [(R₀ = 3.12, mean_ip = 6.4, ρ = 0.01), (R₀ = 2.0, mean_ip = 4.0, ρ = 0.02), (R₀ = 4.0, mean_ip = 12.0, ρ = 0.005)]
ptsG = [(β = 0.6, ρ = 0.01, nb_p = 0.167), (β = 0.8, ρ = 0.02, nb_p = 0.3), (β = 0.4, ρ = 0.004, nb_p = 0.1)]
ptsC = [(nb_p = 0.627,), (nb_p = 0.4,), (nb_p = 0.8,)]
ptsCg = [(nb_p = 0.156,), (nb_p = 0.3,), (nb_p = 0.08,)]
mk(f, args...) = Base.invokelatest(f, args...)
lik["da_dph_full"]       = model_values(D, mk(D.dph_full, N_, M_, ns_, K_, th_, de_, 10, 1.0), ptsA)
lik["da_boxcar_full"]    = model_values(D, mk(D.boxcar_full, N_, M_, ns_, K_, th_, de_, D.K_age, 10, 1.0), ptsA)
lik["da_geo_full"]       = model_values(D, mk(D.geo_full, N_, M_, ns_, K_, th_, de_, 1.0), ptsG)
lik["da_dph_infonly"]    = model_values(D, mk(D.dph_infonly, N_, ns_, K_, th_, 10, 1.0), ptsB)
lik["da_boxcar_infonly"] = model_values(D, mk(D.boxcar_infonly, N_, ns_, K_, th_, D.K_age_B, 10, 1.0), ptsB)
lik["da_geo_infonly"]    = model_values(D, mk(D.geo_infonly, N_, M_, ns_, K_, th_, 1.0), ptsG)
lik["da_dph_reconly"]    = model_values(D, mk(D.dph_reconly, de_, D.nb_r_max_data), ptsC)
lik["da_geo_reconly"]    = model_values(D, mk(D.geo_reconly, de_), ptsCg)
println("data-availability done")

# ---- DDSA vs ODE (synthetic, ρ = 1/N)
OQ = "DDSA_vs_ODE_count/ddsa_vs_ode_count.qmd"; O = notebook_module("DDSAvsODE")
Core.eval(O, :(Random.seed!(1234)))
run!(O, cell_with(OQ, "β_prior_hi = 2.5    # shared"))
run!(O, cell_with(OQ, "data  = generate_discrete_glct_data(β_true"))
dat["ddsa_vs_ode"] = Dict("seed" => 1234, "N" => O.N_pop, "rho" => O.ρ_true, "beta" => O.β_true, "k" => O.k,
    "mu_ip" => O.μ_ip, "dt" => O.Δt, "nsteps" => O.nsteps, "K" => O.K, "th" => O.th, "delta" => O.delta)
lik["ode_dph_full"] = model_values(O, mk(O.dph_full, O.N, 1, O.nsteps, O.K, O.th, O.delta,
                                         min(10, minimum(O.delta)), O.Δt), ptsA)
println("DDSA vs ODE done")

# ---- Hagelloch 1861 measles (three groups)
Hg = notebook_module("Hagelloch")
Core.eval(Hg, :(Random.seed!(1234)))
run!(Hg, cell_with(HAQ, "df = CSV.read(\"hagelloch.csv\""); dir = joinpath(ORIG, "Measles_Hagelloch_1861"))
defs!(Hg, HAQ, ["run_sir_mtheta", "hazard_vector_from_dph", "run_sir_mtheta_boxcar"])
run!(Hg, "δ_floor_ip = minimum(delta)")
run!(Hg, cell_with(HAQ, "global K_age_boxcar"))
defs!(Hg, HAQ, ["safe_normalise"])
run!(Hg, lines_from(HAQ, "glct_propagate(beta_C1, beta_C2, beta_H, beta_G) =", 6))
defs!(Hg, HAQ, ["hagelloch_cell", "ddsa_hagelloch_mtheta", "ddsa_hagelloch_mtheta_boxcar"])
run!(Hg, cell_with(HAQ, "cells   = vec(collect(Iterators.product("))
dat["hagelloch"] = Dict("K0" => Hg.K0, "K1" => Hg.K1, "K2" => Hg.K2, "nsteps" => Hg.nsteps, "th0" => Hg.th0,
    "th1" => Hg.th1, "th2" => Hg.th2, "delta" => Hg.delta, "nb_r_max" => Hg.nb_r_max, "N0_max" => Hg.N0_max,
    "N1_max" => Hg.N1_max, "N2_max" => Hg.N2_max, "delta_floor" => Hg.δ_floor_ip, "K_age_boxcar" => Hg.K_age_boxcar,
    "Tmax" => Hg.Tmax)
let dph = Hg.PhaseTypeDistributions.dph_negative_binomial_min_support(4, 0.479, 4)
    a = (0.1174, 0.021, 0.0124, 0.0638, 0.0, 0.0, 1 / 77, Hg.nsteps)
    s = mk(Hg.run_sir_mtheta, dph.S, dph.α, a...)
    maps["hagelloch_glct"] = Dict("r" => 4, "p" => 0.479, "delta_min" => 4,
        "betas" => collect(a[1:4]), "rho" => collect(a[5:7]), "nsteps" => Hg.nsteps,
        "tau" => [s.tau0, s.tau1, s.tau2], "f0" => fl(s.f0), "f1" => fl(s.f1), "f2" => fl(s.f2))
    h = mk(Hg.hazard_vector_from_dph, dph, Hg.K_age_boxcar)
    b = mk(Hg.run_sir_mtheta_boxcar, h, Hg.K_age_boxcar, a...)
    maps["hagelloch_boxcar"] = Dict("r" => 4, "p" => 0.479, "delta_min" => 4, "K_age" => Hg.K_age_boxcar,
        "betas" => collect(a[1:4]), "rho" => collect(a[5:7]), "nsteps" => Hg.nsteps,
        "tau" => [b.tau0, b.tau1, b.tau2], "f0" => fl(b.f0), "f1" => fl(b.f1), "f2" => fl(b.f2))
end
ptsH = [(beta_C1 = 0.1174, beta_C2 = 0.021, beta_H = 0.0124, beta_G = 0.0638, mean_δ = 8.3537),
        (beta_C1 = 0.2, beta_C2 = 0.05, beta_H = 0.03, beta_G = 0.05, mean_δ = 10.0)]
hargs = (Hg.N0_grid, Hg.N1_grid, Hg.K2, Hg.cells, Hg.th0, Hg.th1, Hg.th2, Hg.delta, Hg.δ_floor_ip)
lik["hagelloch_glct"] = model_values(Hg, mk(Hg.ddsa_hagelloch_mtheta, hargs...), ptsH)
lik["hagelloch_boxcar"] = model_values(Hg, mk(Hg.ddsa_hagelloch_mtheta_boxcar, hargs...), ptsH)
lik["hagelloch_cells"] = Dict("r" => first.(Hg.cells), "N2" => last.(Hg.cells))
println("Hagelloch done")

writetoml("data.toml", dat); writetoml("forward_maps.toml", maps); writetoml("likelihoods.toml", lik)
println("wrote data.toml, forward_maps.toml, likelihoods.toml to $OUT")
