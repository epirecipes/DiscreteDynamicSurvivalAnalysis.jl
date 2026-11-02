using DiscreteDynamicSurvivalAnalysis
using Distributions, Turing, Test

@testset "infectious-period models" begin
    @test GeometricIP().parameter === :p
    @test GeometricIP(parameter = :γ).parameter === :γ
    @test GeometricIP(parameter = :mean_ip).parameter === :mean_ip
    @test_throws ArgumentError GeometricIP(parameter = :r)

    ip = NegBinGLCT(1:10)
    @test ip.r_support == 1:10 && ip.parameter === :p
    @test NegBinGLCT(Base.OneTo(4); parameter = :mean_ip).r_support == 1:4
    @test_throws ArgumentError NegBinGLCT(0:3)
    @test_throws ArgumentError NegBinGLCT(5:4)
    @test_throws ArgumentError NegBinGLCT(1:3; parameter = :γ)

    bx = NegBinBoxcar(1:10, 15)
    @test bx.K_age == 15 && bx.truncation === :absorb && bx.parameter === :p
    @test NegBinBoxcar(2:5, 30; parameter = :mean_ip, truncation = :condition).truncation === :condition
    @test_throws ArgumentError NegBinBoxcar(0:3, 15)
    @test_throws ArgumentError NegBinBoxcar(1:3, 0)
    @test_throws ArgumentError NegBinBoxcar(1:3, 15; parameter = :γ)
    @test_throws ArgumentError NegBinBoxcar(1:3, 15; truncation = :other)
end

@testset "DDSAModel validation" begin
    prA = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1))
    spec = DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 99)
    @test spec.terms == (:final_size, :infection_times, :recovery, :removal_only)
    @test spec.transmission === :β && spec.population == 99
    @test spec.guards.τ_bounds == (0.0, 1.0) && spec.guards.f_floor == 0.0
    @test sprint(show, spec) == "DDSAModel(NegBinGLCT(1:10, :p), parameters: β, ρ, p)"
    txt = sprint(show, MIME"text/plain"(), spec)
    @test occursin("β ~ Uniform", txt) && occursin("population:        99", txt) && occursin("f_floor = 0.0", txt)

    # missing and unused priors
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = (β = Uniform(0, 1), p = Beta(1, 1)),
                                         population = 99)
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = merge(prA, (γ = Uniform(0, 1),)),
                                         population = 99)
    @test_throws ArgumentError DDSAModel(infectious_period = GeometricIP(parameter = :γ), priors = prA, population = 99)
    # recovery-only models sample only the IP parameter
    rec = DDSAModel(infectious_period = GeometricIP(), priors = (p = Beta(1, 1),), terms = (:recovery,))
    @test rec.terms == (:recovery,) && rec.population === nothing
    @test_throws ArgumentError DDSAModel(infectious_period = GeometricIP(), priors = prA, terms = (:recovery,))
    # R0 parameterisation
    prB = (R0 = Uniform(0.5, 5), mean_ip = Uniform(1, 30), ρ = Beta(1, 98))
    @test DDSAModel(infectious_period = NegBinGLCT(1:10; parameter = :mean_ip), priors = prB, transmission = :R0,
                    population = 99).transmission === :R0
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prB, transmission = :R0,
                                         population = 99)
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, transmission = :γ,
                                         population = 99)
    # population
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA)
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 99.0)
    @test DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 91:200).population == 91:200
    @test DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = NGrid(91, 91, 200)).population isa NGrid
    @test DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, terms = (:infection_times,)).population === nothing
    # terms and priors
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 99,
                                         terms = (:final_size, :onsets))
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 99, terms = ())
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), population = 99,
                                         priors = (β = Uniform(0, 1), ρ = Beta(1, 98), p = MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])))
    @test_throws ArgumentError DDSAModel(infectious_period = NegBinGLCT(1:10), population = 99,
                                         priors = (β = 0.5, ρ = Beta(1, 98), p = Beta(1, 1)))
    # guards
    @test_throws ArgumentError NumericalGuards(τ_bounds = (0.5, 0.4))
    @test_throws ArgumentError NumericalGuards(τ_bounds = (-0.1, 1.0))
    @test_throws ArgumentError NumericalGuards(f_floor = -1)
end

@testset "turing_model" begin
    data = LineList([1, 3, 6, 4, 2, 0, 0, 0]; recovery_intervals = [3, 4, 4, 5, 2], final_size = 16)
    prA = (β = Uniform(1e-4, 2.5), ρ = Beta(1, 98), p = Beta(1, 1))
    m = turing_model(DDSAModel(infectious_period = NegBinGLCT(1:3), priors = prA, population = 20), data)
    ret = returned(m, (β = 0.8, ρ = 0.02, p = 0.5))
    @test length(ret.lps) == 3 && length(ret.τ) == 3
    @test ret.lps[3] == -Inf                         # r = 3 exceeds the shortest interval (2)
    @test isfinite(logjoint(m, (β = 0.8, ρ = 0.02, p = 0.5)))
    # a censored interval shorter than r only says W > δ, so it does not rule r out
    cdata = LineList([1, 3, 6, 4, 2, 0, 0, 0]; recovery_intervals = [3, 4, 4, 5, 1], censored = [false, false, false, false, true],
                     final_size = 16)
    mc = turing_model(DDSAModel(infectious_period = NegBinGLCT(1:4), priors = prA, population = 20), cdata)
    lps = returned(mc, (β = 0.8, ρ = 0.02, p = 0.5)).lps
    @test all(isfinite, lps[1:3]) && lps[4] == -Inf     # r = 4 exceeds the shortest uncensored interval (3)
    d3 = dph_negative_binomial(3, 0.5)
    rec = turing_model(DDSAModel(infectious_period = NegBinGLCT(3:3), priors = (p = Beta(1, 1),), terms = (:recovery,)), cdata)
    @test loglikelihood(rec, (p = 0.5,)) ≈ sum(logpdf(d3, δ) for δ in [3, 4, 4, 5]) + logccdf(d3, 1)
    # population checks against the data
    @test_throws ArgumentError turing_model(DDSAModel(infectious_period = NegBinGLCT(1:3), priors = prA, population = 15), data)
    @test_throws ArgumentError turing_model(DDSAModel(infectious_period = NegBinGLCT(1:3), priors = prA,
                                                      population = NGrid(17, 17, 40)), data)
    # an unknown population as a range is summed out on a uniform grid
    mr = turing_model(DDSAModel(infectious_period = GeometricIP(), priors = prA, population = 16:40), data)
    mg = turing_model(DDSAModel(infectious_period = GeometricIP(), priors = prA, population = NGrid(16, 16, 40)), data)
    @test logjoint(mr, (β = 0.8, ρ = 0.02, p = 0.3)) ≈ logjoint(mg, (β = 0.8, ρ = 0.02, p = 0.3))
    # the likelihood is the sum of the requested terms
    pt = (β = 0.8, ρ = 0.02, p = 0.3)
    full = loglikelihood(turing_model(DDSAModel(infectious_period = GeometricIP(), priors = prA, population = 20), data), pt)
    parts = sum(loglikelihood(turing_model(DDSAModel(infectious_period = GeometricIP(), population = 20, terms = (t,),
                                                     priors = t === :recovery ? (p = prA.p,) : prA), data),
                              t === :recovery ? (p = 0.3,) : pt)
                for t in (:final_size, :infection_times, :recovery))
    @test full ≈ parts
end
