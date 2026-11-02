using DiscreteDynamicSurvivalAnalysis
using Random, Statistics, Test

@testset "simulate_linelist reproduces the manuscript's synthetic data" begin
    dat = fixture("data")
    for (name, ρ) in ("data_availability" => 0.01, "ddsa_vs_ode" => 1 / 99)
        d = dat[name]
        sim = simulate_linelist(Xoshiro(d["seed"]), dph_erlang_from_mean(d["k"], d["mu_ip"], d["dt"]), d["beta"], ρ,
                                d["N"], d["nsteps"])
        @test sim.data.final_size == d["K"]
        @test sim.data.infection_counts == d["th"]
        @test sim.data.recovery_intervals == d["delta"]
        @test length(sim.data.recovery_intervals) == d["K"] + 1          # M = 1 initial infective
        if haskey(d, "tau")
            @test sim.τ ≈ d["tau"] rtol = 1e-12
            @test agree(sim.f, d["f"]; rtol = 1e-10, atol = 1e-13)
        end
    end
    sim = simulate_linelist(Xoshiro(1), dph_geometric(0.3), 0.0, 0.01, 50, 20; M = 0)
    @test sim.data.final_size == 0 && isempty(sim.data.recovery_intervals)
    @test_throws ArgumentError simulate_linelist(Xoshiro(1), dph_geometric(0.3), 0.5, 0.01, -1, 20)
    @test_throws ArgumentError simulate_linelist(Xoshiro(1), dph_geometric(0.3), 0.5, 0.01, 50, 20; M = -1)
end

@testset "chain binomial agrees with the forward map on average" begin
    # With M = 10 initial infectives early extinction is negligible, so the mean final size over
    # the simulations matches the deterministic map (τ is a fraction of the N + M individuals).
    N, M, nsim = 2000, 10, 200
    ρ = M / (N + M)
    for (dph, β) in ((dph_geometric(0.25), 0.6), (dph_negative_binomial(3, 0.5), 0.35))
        rng = Xoshiro(11)
        sims = [simulate_chain_binomial(rng, dph, β, N; M = M) for _ in 1:nsim]
        z = [s.data.final_size / (N + M) for s in sims]
        τ = attack_fraction(forward_map(GLCT(dph), β, ρ, 400))
        @test abs(mean(z) - τ) < 4 * std(z) / sqrt(nsim) + 1e-3
        w = reduce(vcat, [s.data.recovery_intervals for s in sims])
        @test abs(mean(w) - mean(dph)) < 4 * std(w) / sqrt(length(w))
        s = sims[1]
        @test s.data.final_size == length(s.infection_steps) == sum(s.data.infection_counts)
        @test length(s.seed_intervals) == M
        @test all(s.removal_steps .> s.infection_steps)
        @test s.data.recovery_intervals == vcat(s.seed_intervals, s.removal_steps .- s.infection_steps)
        @test s.data.nsteps == max(maximum(s.removal_steps), maximum(s.seed_intervals))
    end
    # no transmission: only the initial infectives
    s0 = simulate_chain_binomial(Xoshiro(2), dph_geometric(0.5), 0.0, 100; M = 3)
    @test s0.data.final_size == 0 && length(s0.data.recovery_intervals) == 3
    @test_throws ArgumentError simulate_chain_binomial(Xoshiro(1), dph_geometric(0.5), 0.5, -1)
    @test_throws ArgumentError simulate_chain_binomial(Xoshiro(1), dph_geometric(0.5), 0.5, 100; M = 0)
    @test_throws ArgumentError simulate_chain_binomial(Xoshiro(1), dph_geometric(0.5), -0.5, 100)
end
