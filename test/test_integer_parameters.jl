using DiscreteDynamicSurvivalAnalysis.IntegerParameters
using Distributions, Random, ForwardDiff, Test

@testset "IntegerParameters" begin
    ref = fixture("integer_parameters")

    @testset "agreement with the original module" begin
        r = ref["ngrid_91"]
        g = NGrid(r["K"], r["N_min"], r["N_max"])
        @test g.logC == r["logC"] && g.logprior == r["logprior"]
        @test [marginal_N_loglik(g, τ) for τ in r["tau"]] == r["marginal"]
        @test [marginal_N_loglik(g, log(τ), log1p(-τ)) for τ in r["tau"]] == r["marginal_log"]
        @test N_posterior_weights(g, 0.9) == r["weights_0.9"]
        r = ref["ngrid_91_prior"]
        gp = NGrid(91, 91, 99; logprior = r["logprior_in"])
        @test gp.logprior == r["logprior"] && marginal_N_loglik(gp, 0.9) == r["marginal_0.9"]
        r = ref["sample_N_wsu"]
        gw = NGrid(r["K"], r["N_min"], r["N_max"])
        rng = Xoshiro(r["seed"])
        @test [sample_N(rng, gw, r["tau"]) for _ in 1:20] == r["draws"]
        @test marginal_N_loglik(gw, r["tau"]) == r["marginal"]
        r = ref["marginal_r"]
        f_r(x) = -0.5 * (x - 3.2)^2 + (x == 7 ? -Inf : 0.0)
        logm, lps = marginal_r_loglik(f_r, 1:8)
        @test lps == r["lps"] && logm == r["logmarg"]
        @test marginal_r_loglik(f_r, 1:8; logprior = log.(1:8))[1] == r["logmarg_prior"]
        @test posterior_weights(lps) == r["weights"]
        @test posterior_weights(lps; logprior = log.(1:8)) == r["weights_prior"]
        rng = Xoshiro(r["seed"])
        @test [sample_index(rng, lps) for _ in 1:30] == r["draws"]
    end

    @testset "brute-force sums" begin
        g = NGrid(12, 12, 40)
        for τ in (0.2, 0.5, 0.85)
            brute = log(sum(pdf(Binomial(N, τ), 12) for N in 12:40) / 29)
            @test marginal_N_loglik(g, τ) ≈ brute rtol = 1e-12
            @test marginal_N_loglik(g, τ) ≈ marginal_N_loglik(g, log(τ), log1p(-τ)) rtol = 1e-14
            w = [pdf(Binomial(N, τ), 12) for N in 12:40]
            @test N_posterior_weights(g, τ) ≈ w ./ sum(w) rtol = 1e-10
        end
        @test marginal_N_loglik(g, 0.0) == -Inf && marginal_N_loglik(g, 1.0) == -Inf
        lps = [-1.0, -3.0, -Inf, 0.5]
        @test marginal_r_loglik(i -> lps[i], 1:4)[1] ≈ log(sum(exp.(lps)) / 4)
        @test posterior_weights(fill(-Inf, 3)) == fill(1 / 3, 3)
        # a joint grid keeps its shape (the first value varies fastest)
        @test marginal_r_loglik(t -> -abs(t[1] - t[2]), Iterators.product(1:3, 1:2))[2] == [0 -1; -1 0; -2 -1]
    end

    @testset "threaded evaluation equals serial" begin
        f(x) = -0.1 * x^2 + sin(x)
        @test marginal_r_loglik(f, 1:50; threaded = true) == marginal_r_loglik(f, 1:50)
    end

    @testset "draw frequencies match the weights" begin
        lps = log.([0.1, 0.6, 0.3])
        rng = Xoshiro(3)
        n = 30_000
        x = [sample_index(rng, lps) for _ in 1:n]
        for i in 1:3
            p = exp(lps[i])
            @test abs(count(==(i), x) / n - p) < 4 * sqrt(p * (1 - p) / n)
        end
        g = NGrid(5, 5, 9)
        w = N_posterior_weights(g, 0.4)
        y = [sample_N(rng, g, 0.4) for _ in 1:n]
        for (i, N) in enumerate(5:9)
            @test abs(count(==(N), y) / n - w[i]) < 4 * sqrt(w[i] * (1 - w[i]) / n)
        end
    end

    @testset "ForwardDiff through the marginal" begin
        g = NGrid(30, 30, 200)
        d = ForwardDiff.derivative(τ -> marginal_N_loglik(g, τ), 0.3)
        h = 1e-6
        @test d ≈ (marginal_N_loglik(g, 0.3 + h) - marginal_N_loglik(g, 0.3 - h)) / 2h rtol = 1e-6
        dr = ForwardDiff.derivative(b -> marginal_r_loglik(r -> -b * r, 1:5)[1], 0.7)
        @test isfinite(dr)
    end

    @testset "validation" begin
        @test_throws ArgumentError NGrid(10, 5, 20)
        @test_throws ArgumentError NGrid(10, 12, 11)
        @test_throws ArgumentError NGrid(10, 10, 12; logprior = [0.0, 0.0])
        @test_throws ArgumentError NGrid(10, 10, 12; logprior = fill(-Inf, 3))
        @test_throws ArgumentError marginal_r_loglik(identity, 1:3; logprior = [0.0, 0.0])
        @test_throws ArgumentError marginal_r_loglik(identity, Int[])
        @test sprint(show, NGrid(3, 3, 9)) == "NGrid(K = 3, N ∈ 3:9)"
    end
end
