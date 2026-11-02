using DiscreteDynamicSurvivalAnalysis
using Distributions, Turing, MCMCChains, Random, Statistics, Logging, Test

# A short fit of the data-availability example (scenario A, NegBin GLCT): the synthetic data of the
# manuscript (K = 91), generated from an Erlang-4 DPH with E[W] = 6.33 steps and R0 = 3.16.
const DA = fixture("data")["data_availability"]
const DA_DATA = simulate_linelist(Xoshiro(1234), dph_erlang_from_mean(4, 4.0, 1.0), 0.5, 0.01, 99, 40).data
const PRIORS_A = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1))

@testset "end-to-end fit (data-availability scenario A)" begin
    spec = DDSAModel(infectious_period = NegBinGLCT(1:10), priors = PRIORS_A, population = 99,
                     guards = NumericalGuards(τ_bounds = (0.001, 0.999)))
    fit = fit_ddsa(Xoshiro(2024), spec, DA_DATA; ndraws = 300, nchains = 2, sampler = NUTS(300, 0.8))
    @test fit isa DDSAFit
    @test size(fit.chain, 1) == 300 && size(fit.chain, 3) == 2
    @test :r in names(fit.chain, :parameters) && !(:N in names(fit.chain, :parameters))
    @test occursin("NegBinGLCT", sprint(show, fit))

    d = posterior_draws(fit)
    @test length(d.β) == 600
    @test all(in(1:4), d.r)                                       # r ≤ shortest interval (4)
    @test all(==(99), d.N)
    @test all(0 .< d.τ .< 1)
    @test d.mean_ip ≈ d.r ./ d.p
    @test d.R0 ≈ d.β .* d.mean_ip
    @test quantile(d.R0, 0.025) < DA["R0_imp"] < quantile(d.R0, 0.975)
    @test quantile(d.mean_ip, 0.025) < DA["EW"] < quantile(d.mean_ip, 0.975)
    rcounts = [count(==(r), d.r) for r in 1:10]
    @test argmax(rcounts) == 4

    rows = posterior_summary(fit)
    @test [r.parameter for r in rows] == collect(keys(d))
    @test all(r -> keys(r) == (:parameter, :mean, :std, :q2_5, :q50, :q97_5, :ess, :rhat), rows)
    rowβ = only(filter(r -> r.parameter === :β, rows))
    @test rowβ.mean ≈ mean(d.β) && rowβ.q50 ≈ median(d.β)
    @test rowβ.ess > 50 && rowβ.rhat < 1.1
    @test isnan(only(filter(r -> r.parameter === :N, rows)).rhat)
    @test keys(posterior_summary(fit; probs = (0.1, 0.9))[1])[4:5] == (:q10, :q90)

    pp = posterior_predictive(Xoshiro(1), fit; ndraws = 50)
    @test size(pp.incidence) == (40, 50) && length(pp.final_size) == 50
    @test pp.observed == DA_DATA.infection_counts && pp.observed_final_size == 91 && pp.scale === :expected_known
    @test pp.times == 0:39
    @test all(0 .≤ pp.final_size .≤ 99)
    idx = DiscreteDynamicSurvivalAnalysis._pooled_index(600, 50)
    @test vec(sum(pp.incidence; dims = 1)) ≈ 99 .* d.τ[idx]
    ppe = posterior_predictive(Xoshiro(1), fit; ndraws = 50, scale = :expected)
    @test ppe.incidence == pp.incidence                     # every infection time is known: K_known = K
    ppo = posterior_predictive(Xoshiro(1), fit; ndraws = 50, scale = :observed_total)
    @test all(sum(ppo.incidence; dims = 1) .≈ sum(DA_DATA.infection_counts))

    ## 5 of the K = 91 infection times unknown: the default curve is thinned by K_known / K = 86 / 91
    counts = copy(DA_DATA.infection_counts)
    counts[argmax(counts)] -= 5
    fit_missing = DDSAFit(fit.spec, LineList(counts; final_size = 91, recovery_intervals = DA_DATA.recovery_intervals),
                          fit.model, fit.chain, fit.population, fit.τ)
    ppk = posterior_predictive(Xoshiro(1), fit_missing; ndraws = 50)
    @test ppk.incidence ≈ (86 / 91) .* ppe.incidence
    @test vec(sum(ppk.incidence; dims = 1)) ≈ (86 / 91) * 99 .* d.τ[idx]
    @test ppk.final_size == pp.final_size && ppk.observed_final_size == 91
    q = predictive_quantiles(pp)
    @test size(q) == (40, 3) && all(q[:, 1] .≤ q[:, 2] .≤ q[:, 3])
    @test_throws ArgumentError posterior_predictive(fit; scale = :counts)

    rp = recovery_predictive(fit; ndraws = 40, max_interval = 200)
    @test size(rp.pmf) == (200, 40) && rp.intervals == 1:200
    @test all(sum(rp.pmf; dims = 1) .≈ 1)
    @test sum(rp.observed) == length(DA_DATA.recovery_intervals)
    @test size(predictive_quantiles(recovery_predictive(fit))) == (maximum(DA_DATA.recovery_intervals) + 5, 3)
end

@testset "reproducibility and unknown N" begin
    spec = DDSAModel(infectious_period = GeometricIP(), priors = PRIORS_A, population = 91:150)
    fitrun(par) = fit_ddsa(Xoshiro(7), spec, DA_DATA; ndraws = 40, nchains = 2, sampler = NUTS(40, 0.8), parallel = par)
    for par in (MCMCSerial(), MCMCThreads())
        a, b = fitrun(par), fitrun(par)
        @test Array(a.chain) == Array(b.chain)
        @test a.τ == b.τ
    end
    fit = fitrun(MCMCSerial())
    @test :N in names(fit.chain, :parameters) && !(:r in names(fit.chain, :parameters))
    d = posterior_draws(fit)
    @test all(in(91:150), d.N) && d.K_over_N ≈ 91 ./ d.N
    @test all(==(1), d.r) && d.R0 ≈ d.β ./ d.p
    @test fit.population isa NGrid

    # a recovery-only model: no forward map, no τ, no predictive curve
    rec = DDSAModel(infectious_period = NegBinGLCT(1:4), priors = (p = Beta(1, 1),), terms = (:recovery,))
    fr = fit_ddsa(Xoshiro(3), rec, DA_DATA; ndraws = 50, nchains = 1, sampler = NUTS(50, 0.8), initial_params = :prior)
    dr = posterior_draws(fr)
    @test !hasproperty(dr, :τ) && !hasproperty(dr, :β) && hasproperty(dr, :mean_ip)
    @test all(isnan, fr.τ)
    @test_throws ArgumentError posterior_predictive(fr)

    # Turing's info messages are hidden unless verbose = true; warnings still pass through
    logger = Test.TestLogger(min_level = Logging.Info)
    with_logger(logger) do
        fit_ddsa(Xoshiro(7), spec, DA_DATA; ndraws = 10, nchains = 1, sampler = NUTS(10, 0.8))
    end
    @test !any(r -> occursin("step size", string(r.message)), logger.logs)
    logger = Test.TestLogger(min_level = Logging.Info)
    with_logger(logger) do
        fit_ddsa(Xoshiro(7), spec, DA_DATA; ndraws = 10, nchains = 1, sampler = NUTS(10, 0.8), verbose = true)
    end
    @test any(r -> occursin("step size", string(r.message)), logger.logs)

    @test_throws ArgumentError fit_ddsa(spec, DA_DATA; ndraws = 0)
    @test_throws ArgumentError fit_ddsa(spec, DA_DATA; nchains = 0)
    @test_throws ArgumentError fit_ddsa(spec, DA_DATA; nchains = 2, initial_params = [InitFromPrior()])
    @test_throws ArgumentError fit_ddsa(spec, DA_DATA; initial_params = :mle)
end

@testset "data_based_inits" begin
    specs = (DDSAModel(infectious_period = NegBinGLCT(1:10), priors = PRIORS_A, population = 99),
             DDSAModel(infectious_period = GeometricIP(parameter = :γ), population = 91:500,
                       priors = (β = Uniform(0.01, 5), γ = truncated(LogNormal(log(1 / 5.5), 0.3), 0.01, 1.0), ρ = Beta(1, 10))),
             DDSAModel(infectious_period = NegBinBoxcar(1:10, 204; parameter = :mean_ip), transmission = :R0, population = 99,
                       priors = (R0 = Uniform(0.5, 5), mean_ip = Uniform(1, 30), ρ = Beta(1, 98)),
                       terms = (:final_size, :infection_times)))
    for spec in specs
        inits = data_based_inits(Xoshiro(5), spec, DA_DATA, 4)
        @test length(inits) == 4
        for init in inits
            @test init isa InitFromParams
            vals = Dict(Symbol(string(k)) => v for (k, v) in pairs(init.params))   # keys are VarNames
            @test Set(keys(vals)) == Set(keys(spec.priors))
            for (k, v) in vals
                pr = spec.priors[k]
                @test quantile(pr, 0.01) ≤ v ≤ quantile(pr, 0.99)
            end
        end
    end
    @test data_based_inits(Xoshiro(5), specs[1], DA_DATA, 2)[1].params == data_based_inits(Xoshiro(5), specs[1], DA_DATA, 2)[1].params
end
