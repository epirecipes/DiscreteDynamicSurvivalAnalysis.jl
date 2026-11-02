using DiscreteDynamicSurvivalAnalysis
using Distributions, Turing, Random, Test

const NAMEMAP = Dict("beta" => :β, "gamma" => :γ, "rho" => :ρ, "p" => :p, "R0" => :R0, "mean_ip" => :mean_ip,
                     "mean_delta" => :mean_δ, "beta_C1" => :beta_C1, "beta_C2" => :beta_C2,
                     "beta_H" => :beta_H, "beta_G" => :beta_G)
point(d) = (; (NAMEMAP[k] => v for (k, v) in d)...)
const MAX_RELDIFF = Dict{String,Float64}()

"""Compare logjoint/loglikelihood (and the returned per-r values) with the original notebook model.
The per-r values get a looser `rtol_returned`: in r cells where R0 < 1 the originals built `f` from
differences of S, which lose relative precision when infections are tiny (up to ~1e-10 here)."""
function compare_model(name, model, ref; rtol = 1e-12, rtol_returned = 1e-9, returned_field = :lps)
    worst = 0.0
    for (k, pt) in enumerate(ref["points"])
        nt = point(pt)
        lj = logjoint(model, nt)
        ll = loglikelihood(model, nt)
        worst = max(worst, abs(lj - ref["logjoint"][k]) / abs(ref["logjoint"][k]),
                    abs(ll - ref["loglikelihood"][k]) / abs(ref["loglikelihood"][k]))
        @test lj ≈ ref["logjoint"][k] rtol = rtol
        @test ll ≈ ref["loglikelihood"][k] rtol = rtol
        ret = returned(model, nt)
        if returned_field === :lps
            @test agree(ret.lps, ref["returned"][k]; rtol = rtol_returned)
        elseif returned_field === :τ
            @test ret.τ[1] ≈ ref["returned"][k] rtol = rtol_returned
        elseif returned_field === :raw
            @test agree(vec(ret), ref["returned"][k]; rtol = rtol_returned)
        end
    end
    MAX_RELDIFF[name] = worst
end

function hagelloch_groups(d)
    nst = d["nsteps"]
    groups = [LineList(d["th$g"]; final_size = d["K$g"]) for g in 0:2]
    @assert all(x -> x.nsteps == nst, groups)
    return groups
end

# The three-group (M_θ) model of the Hagelloch example, written with the package's building blocks.
@model function hagelloch_model(groups, N0_grid, N1_grid, K2, cells, delta, δ_floor, boxcar, K_age)
    beta_C1 ~ Exponential(0.2)
    beta_C2 ~ Exponential(0.1)
    beta_H  ~ Exponential(0.1)
    beta_G  ~ Exponential(0.1)
    mean_δ  ~ Uniform(δ_floor + 0.5, 25.0)
    foi(I) = (Itot = I[1] + I[2] + I[3];
              [beta_H * (I[2] + I[3]) + beta_G * Itot, beta_C1 * I[2] + beta_G * Itot, beta_C2 * I[3] + beta_G * Itot])
    guards = NumericalGuards(τ_bounds = (0.001, 0.999))
    nst = groups[1].nsteps
    logmarg, lps = marginal_r_loglik(cells; threaded = true) do (r, N2)
        p = r / (mean_δ - δ_floor + r)
        dph = dph_negative_binomial_min_support(r, p, δ_floor)
        structure = boxcar ? Boxcar(dph, K_age) : GLCT(dph)
        sol = forward_map(structure, foi, [0.0, 0.0, 1 / N2], nst)
        ll = zero(p)
        for (g, pop) in enumerate((N0_grid, N1_grid, N2))
            ll += ddsa_loglikelihood(sol, groups[g], dph; population = pop, group = g, guards = guards,
                                     terms = (:final_size, :infection_times)).loglik
        end
        ll + recovery_loglik(dph, delta)
    end
    Turing.@addlogprob! logmarg
    return lps
end

@testset "likelihoods agree with the original notebook models" begin
    lik = fixture("likelihoods")
    dat = fixture("data")

    @testset "H1N1 (geometric, N summed out, censoring)" begin
        d = dat["h1n1"]
        data = LineList(d["th"]; final_size = d["K"], recovery_intervals = d["delta"], censored = Bool.(d["cens"]))
        spec = DDSAModel(infectious_period = GeometricIP(parameter = :γ),
                         priors = (β = Uniform(0.01, 5.0), γ = truncated(LogNormal(log(1 / 5.5), 0.3), 0.01, 1.0),
                                   ρ = Beta(1.0, 10.0)),
                         population = d["N_min"]:d["N_max"], guards = NumericalGuards(f_floor = 1e-10))
        compare_model("h1n1", turing_model(spec, data), lik["h1n1"]; returned_field = :τ)
    end

    @testset "Ebola wave $w (geometric, N summed out, removal-only cases)" for w in 1:3
        d = dat["ebola_w$w"]
        data = LineList(d["th_onset"]; final_size = d["K_total"], recovery_intervals = d["delta_complete"],
                        removal_only_steps = d["T_hosp_only"])
        spec = DDSAModel(infectious_period = GeometricIP(parameter = :γ),
                         priors = (β = Uniform(0.01, 1.0), γ = Uniform(0.01, 1.0), ρ = Beta(1.0, 500.0)),
                         population = d["N_min"]:d["N_max"],
                         guards = NumericalGuards(τ_bounds = (1e-6, 1 - 1e-6), f_floor = 1e-10))
        compare_model("ebola_w$w", turing_model(spec, data), lik["ebola_w$w"]; returned_field = :τ)
    end

    @testset "data-availability scenarios" begin
        d = dat["data_availability"]
        data = LineList(d["th"]; final_size = d["K"], recovery_intervals = d["delta"])
        g = NumericalGuards(τ_bounds = (0.001, 0.999))
        prA = (β = Uniform(1e-4, 2.5), ρ = Beta(1.0, 98.0), p = Beta(1.0, 1.0))
        prB = (R0 = Uniform(0.5, 5.0), mean_ip = Uniform(1.0, 30.0), ρ = Beta(1.0, 98.0))
        inf_only = (:final_size, :infection_times)
        specs = Dict(
            "da_dph_full"       => DDSAModel(infectious_period = NegBinGLCT(1:10), priors = prA, population = 99, guards = g),
            "da_boxcar_full"    => DDSAModel(infectious_period = NegBinBoxcar(1:10, d["K_age"]; truncation = :condition),
                                             priors = prA, population = 99, guards = g),
            "da_geo_full"       => DDSAModel(infectious_period = GeometricIP(), priors = prA, population = 99, guards = g),
            "da_dph_infonly"    => DDSAModel(infectious_period = NegBinGLCT(1:10; parameter = :mean_ip), priors = prB,
                                             transmission = :R0, population = 99, guards = g, terms = inf_only),
            "da_boxcar_infonly" => DDSAModel(infectious_period = NegBinBoxcar(1:10, d["K_age_B"]; parameter = :mean_ip,
                                                                              truncation = :condition),
                                             priors = prB, transmission = :R0, population = 99, guards = g, terms = inf_only),
            "da_geo_infonly"    => DDSAModel(infectious_period = GeometricIP(), priors = prA, population = 99, guards = g,
                                             terms = inf_only),
            "da_dph_reconly"    => DDSAModel(infectious_period = NegBinGLCT(1:d["nb_r_max_data"]),
                                             priors = (p = Beta(1.0, 1.0),), terms = (:recovery,)),
            "da_geo_reconly"    => DDSAModel(infectious_period = GeometricIP(), priors = (p = Beta(1.0, 1.0),),
                                             terms = (:recovery,)),
        )
        for (name, spec) in specs
            @testset "$name" begin
                rf = spec.infectious_period isa GeometricIP ? :none : :lps
                compare_model(name, turing_model(spec, data), lik[name]; returned_field = rf)
            end
        end
    end

    @testset "DDSA vs ODE: DDSA model" begin
        d = dat["ddsa_vs_ode"]
        data = LineList(d["th"]; final_size = d["K"], recovery_intervals = d["delta"])
        spec = DDSAModel(infectious_period = NegBinGLCT(1:min(10, minimum(d["delta"]))),
                         priors = (β = Uniform(1e-4, 2.5), ρ = Beta(1.0, d["N"] - 1), p = Beta(1.0, 1.0)),
                         population = d["N"], guards = NumericalGuards(τ_bounds = (0.001, 0.999)))
        compare_model("ode_dph_full", turing_model(spec, data), lik["ode_dph_full"])
    end

    @testset "Hagelloch three-group model ($(boxcar ? "Boxcar" : "GLCT"))" for boxcar in (false, true)
        d = dat["hagelloch"]
        groups = hagelloch_groups(d)
        N0g = NGrid(d["K0"], d["K0"], d["N0_max"])
        N1g = NGrid(d["K1"], d["K1"], d["N1_max"])
        cells = vec(collect(Iterators.product(1:d["nb_r_max"], d["K2"]:d["N2_max"])))
        @test first.(cells) == lik["hagelloch_cells"]["r"] && last.(cells) == lik["hagelloch_cells"]["N2"]
        model = hagelloch_model(groups, N0g, N1g, d["K2"], cells, d["delta"], d["delta_floor"], boxcar, d["K_age_boxcar"])
        name = boxcar ? "hagelloch_boxcar" : "hagelloch_glct"
        compare_model(name, model, lik[name]; returned_field = :raw)
    end

    @info "largest relative difference in logjoint/loglikelihood vs the original code:\n" *
          join(("  $k: $(round(v; sigdigits = 2))" for (k, v) in sort(collect(MAX_RELDIFF))), "\n")
end
