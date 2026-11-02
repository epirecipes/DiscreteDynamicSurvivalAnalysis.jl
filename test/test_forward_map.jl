using DiscreteDynamicSurvivalAnalysis
using DiscreteDynamicSurvivalAnalysis.PhaseTypeDistributions: rate_to_proportion
using ForwardDiff, Test

# The three-group force of infection of the Hagelloch example (classes 0, 1, 2).
hagelloch_foi(β_C1, β_C2, β_H, β_G) = I -> begin
    Itot = I[1] + I[2] + I[3]
    [β_H * (I[2] + I[3]) + β_G * Itot, β_C1 * I[2] + β_G * Itot, β_C2 * I[3] + β_G * Itot]
end

# `f` may differ from the originals in relative terms where new infections are tiny (they used
# differences of S); compare it with an absolute tolerance relative to its largest entry.
f_agrees(f, ref) = agree(f, ref; rtol = 1e-10, atol = 1e-13 * maximum(ref))

@testset "forward maps agree with the original notebook maps" begin
    fm = fixture("forward_maps")

    @testset "SIR map of H1N1 / Ebola (geometric GLCT)" begin
        for name in ("h1n1_sir_map", "ebola_w1_sir_map", "ebola_w2_sir_map", "ebola_w3_sir_map")
            r = fm[name]
            sol = forward_map(GLCT(dph_geometric(rate_to_proportion(r["gamma"]))), r["beta"], r["rho"], r["nsteps"])
            @test attack_fraction(sol) ≈ r["tau"] rtol = 1e-12
            @test agree(infection_pmf(sol), r["f"]; rtol = 1e-12)
            haskey(r, "S") && @test agree(susceptible_fraction(sol), r["S"]; rtol = 1e-14)
        end
    end

    @testset "GLCT maps of the data-availability example" begin
        for (name, dph) in ("glct_erlang4" => dph_erlang_from_mean(4, 4.0, 1.0),
                            "glct_negbin_4_0.6" => dph_negative_binomial(4, 0.6),
                            "glct_geometric_0.2" => dph_geometric(0.2))
            r = fm[name]
            sol = forward_map(GLCT(dph), r["beta"], r["rho"], r["nsteps"]; Δt = r["dt"])
            @test attack_fraction(sol) ≈ r["tau"] rtol = 1e-12
            @test f_agrees(infection_pmf(sol), r["f"])
            @test agree(susceptible_fraction(sol), r["S"]; rtol = 1e-14)
        end
    end

    @testset "Boxcar map of the data-availability example (:condition)" begin
        r = fm["boxcar_negbin_4_0.6_K15"]
        h = discrete_hazard(dph_negative_binomial(r["r"], r["p"]), r["K_age"]; truncation = :condition)
        sol = forward_map(Boxcar(h), r["beta"], r["rho"], r["nsteps"])
        @test attack_fraction(sol) ≈ r["tau"] rtol = 1e-12
        @test f_agrees(infection_pmf(sol), r["f"])
        @test agree(susceptible_fraction(sol), r["S"]; rtol = 1e-14)
    end

    @testset "Hagelloch three-group maps" begin
        for name in ("hagelloch_glct", "hagelloch_boxcar")
            r = fm[name]
            dph = dph_negative_binomial_min_support(r["r"], r["p"], r["delta_min"])
            structure = name == "hagelloch_glct" ? GLCT(dph) : Boxcar(dph, r["K_age"])
            sol = forward_map(structure, hagelloch_foi(r["betas"]...), r["rho"], r["nsteps"])
            for g in 1:3
                @test attack_fraction(sol, g) ≈ r["tau"][g] rtol = 1e-12
                @test f_agrees(infection_pmf(sol, g), r["f$(g - 1)"])
            end
        end
    end
end

@testset "forward map invariants" begin
    dph = dph_negative_binomial(4, 0.6)
    for structure in (GLCT(dph), Boxcar(dph, 40), Boxcar(dph, 15; truncation = :condition), GLCT(dph_geometric(0.3)))
        for (β, ρ) in ((0.5, 0.01), (0.1, 0.05), (2.0, 0.001), (0.0, 0.02))
            sol = forward_map(structure, β, ρ, 60; Δt = 0.5)
            S = susceptible_fraction(sol)
            @test S[1] == 1 - ρ
            @test all(diff(S) .≤ 0)
            @test 0 ≤ attack_fraction(sol) ≤ 1 - ρ
            @test attack_fraction(sol) ≈ S[1] - S[end] atol = 1e-14
            @test sum(infection_pmf(sol)) ≈ 1
            @test all(≥(0), infection_pmf(sol))
            @test length(infection_pmf(sol)) == 60 && sol.Δt == 0.5
        end
    end
    # no infections: τ = 0 and a uniform f
    sol0 = forward_map(GLCT(dph), 0.0, 0.01, 10)
    @test attack_fraction(sol0) == 0 && infection_pmf(sol0) == fill(0.1, 10)

    @testset "Boxcar with a long age grid equals GLCT" begin
        # the map stores only the min(K_age, nsteps) occupied classes; with a mean IP of 50 steps
        # (geometric 0.02) many infected reach age 50, so an off-by-one there would show
        for d in (dph_negative_binomial(4, 0.6), dph_geometric(0.25), dph_negative_binomial_min_support(3, 0.45, 5),
                  dph_geometric(0.02))
            g = forward_map(GLCT(d), 0.6, 0.01, 50)
            b = forward_map(Boxcar(d, 300), 0.6, 0.01, 50)
            @test attack_fraction(b) ≈ attack_fraction(g) rtol = 1e-10
            @test infection_pmf(b) ≈ infection_pmf(g) rtol = 1e-10
            @test susceptible_fraction(b) ≈ susceptible_fraction(g) rtol = 1e-10
        end
    end

    @testset "multi-group maps" begin
        for structure in (GLCT(dph), Boxcar(dph, 30))
            single = forward_map(structure, 0.5, 0.01, 40)
            multi = forward_map(structure, fill(0.5, 1, 1), [0.01], 40)
            @test attack_fraction(multi, 1) ≈ attack_fraction(single) rtol = 1e-14
            @test infection_pmf(multi, 1) ≈ infection_pmf(single) rtol = 1e-14
            @test susceptible_fraction(multi, 1) ≈ susceptible_fraction(single) rtol = 1e-14

            B = [0.4 0.1 0.0; 0.05 0.3 0.2; 0.1 0.0 0.6]
            ρ = [0.01, 0.0, 0.02]
            byB = forward_map(structure, B, ρ, 50)
            byfoi = forward_map(structure, I -> B * I, ρ, 50)
            @test byB.S == byfoi.S && byB.τ == byfoi.τ && byB.f == byfoi.f
            @test size(byB.S) == (51, 3) && size(byB.f) == (50, 3) && length(byB.τ) == 3

            D = [0.4 0.0; 0.0 0.9]
            ind = forward_map(structure, D, [0.01, 0.03], 50)
            for (g, (β, ρg)) in enumerate(((0.4, 0.01), (0.9, 0.03)))
                single = forward_map(structure, β, ρg, 50)
                @test attack_fraction(ind, g) ≈ attack_fraction(single) rtol = 1e-14
                @test infection_pmf(ind, g) ≈ infection_pmf(single) rtol = 1e-14
            end
        end
    end

    @testset "forward_map!" begin
        structure = GLCT(dph)
        S_before = copy(dph.S)
        S_hist = zeros(41)
        out = forward_map!(S_hist, structure, 0.5, 0.01)
        @test out === S_hist
        @test S_hist == susceptible_fraction(forward_map(structure, 0.5, 0.01, 40))
        @test dph.S == S_before
        ρ = [0.01, 0.0]
        M = zeros(31, 2)
        @test forward_map!(M, Boxcar(dph, 20), [0.4 0.1; 0.1 0.3], ρ) === M
        @test M == forward_map(Boxcar(dph, 20), [0.4 0.1; 0.1 0.3], ρ, 30).S
        @test ρ == [0.01, 0.0]
        @test_throws DimensionMismatch forward_map!(zeros(31, 3), GLCT(dph), [0.4 0.1; 0.1 0.3], ρ)
    end

    @testset "ForwardDiff" begin
        for structure in (GLCT(dph), Boxcar(dph, 40))
            τ(β) = attack_fraction(forward_map(structure, β, 0.01, 40))
            h = 1e-6
            fd = (τ(0.5 + h) - τ(0.5 - h)) / 2h
            @test ForwardDiff.derivative(τ, 0.5) ≈ fd rtol = 1e-6
            # through the progression probability (the DPH) as well
            τp(p) = attack_fraction(forward_map(GLCT(dph_negative_binomial(4, p)), 0.5, 0.01, 40))
            @test ForwardDiff.derivative(τp, 0.6) ≈ (τp(0.6 + h) - τp(0.6 - h)) / 2h rtol = 1e-6
            # multi-group, through a closure over the parameters
            τB(b) = sum(forward_map(structure, I -> [b * I[1] + 0.1 * I[2], 0.2 * I[1] + b * I[2]], [0.01, 0.0], 40).τ)
            @test ForwardDiff.derivative(τB, 0.5) ≈ (τB(0.5 + h) - τB(0.5 - h)) / 2h rtol = 1e-6
        end
    end

    @testset "argument errors" begin
        @test_throws ArgumentError forward_map(GLCT(dph), 0.5, 1.0, 40)
        @test_throws ArgumentError forward_map(GLCT(dph), 0.5, -0.1, 40)
        @test_throws ArgumentError forward_map(GLCT(dph), 0.5, 0.01, 0)
        @test_throws ArgumentError forward_map(GLCT(dph), 0.5, 0.01, 40; Δt = 0)
        @test_throws ArgumentError forward_map(GLCT(dph), [0.5 0.1; 0.1 0.5], [0.01, 1.0], 40)
        @test_throws DimensionMismatch forward_map(GLCT(dph), [0.5 0.1; 0.1 0.5], [0.01, 0.0, 0.0], 40)
        @test_throws DimensionMismatch forward_map(GLCT(dph), I -> [0.5 * I[1]], [0.01, 0.0], 40)
        @test_throws ArgumentError Boxcar(Float64[])
        @test_throws ArgumentError Boxcar(dph, 10; truncation = :other)
    end
end
