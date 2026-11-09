using DiscreteDynamicSurvivalAnalysis.PhaseTypeDistributions
const PTD = DiscreteDynamicSurvivalAnalysis.PhaseTypeDistributions
using Distributions, Random, Statistics, LinearAlgebra, ForwardDiff, QuadGK, Test

const DPH_CASES = Dict(
    "geometric_0.3"               => () -> dph_geometric(0.3),
    "negbin_4_0.4"                => () -> dph_negative_binomial(4, 0.4),
    "negbin_1_0.7"                => () -> dph_negative_binomial(1, 0.7),
    "negbin_6_0.85"               => () -> dph_negative_binomial(6, 0.85),
    "erlang_from_mean_4_4.0_1.0"  => () -> dph_erlang_from_mean(4, 4.0, 1.0),
    "erlang_3_2.0_0.1"            => () -> dph_erlang(3, 2.0, 0.1),
    "erlang_from_rates_3_1.5_0.5" => () -> dph_erlang_from_rates(3, 1.5, 0.5),
    "exponential_2.0_0.1"         => () -> dph_exponential(2.0, 0.1),
    "negbin_from_rates_3_0.4_0.5" => () -> dph_negative_binomial_from_rates(3, 0.4, 0.5),
    "negbin_from_mean_5_12.5"     => () -> dph_negative_binomial_from_mean(5, 12.5),
    "min_support_3_0.45_5"        => () -> dph_negative_binomial_min_support(3, 0.45, 5),
    "delay_2_negbin_2_0.5"        => () -> dph_prepend_deterministic_delay(dph_negative_binomial(2, 0.5), 2),
    "from_rates_general"          => () -> dph_from_rates([0.6, 0.4], [-2.0 1.0; 0.5 -1.5], 0.3),
    "custom"                      => () -> DPH([0.5, 0.5], [0.2 0.3; 0.1 0.6]),
)

@testset "PhaseTypeDistributions" begin
    ref = fixture("phasetype")
    ns = ref["n"]

    @testset "agreement with the original module: DPH $name" for (name, mk) in DPH_CASES
        r = ref["dph_" * name]
        d = mk()
        @test agree(d.α, r["alpha"]; rtol = 1e-14)
        @test agree(d.S, unflat(r["S"]); rtol = 1e-14)
        @test agree(d.s, r["s"]; rtol = 1e-14, atol = 1e-16)
        @test agree([pmf(d, n) for n in ns], r["pmf"]; rtol = 1e-12, atol = 1e-300)
        @test agree([logpdf(d, n) for n in ns], r["logpdf"]; rtol = 1e-12)
        @test agree([cdf(d, n) for n in ns], r["cdf"]; rtol = 1e-12, atol = 1e-15)
        @test mean(d) ≈ r["mean"] rtol = 1e-12
        @test var(d) ≈ r["var"] rtol = 1e-12
    end

    @testset "seeded rand reproduces the original sampler" begin
        for (name, d) in ("negbin_4_0.4" => dph_negative_binomial(4, 0.4),
                          "erlang_from_mean_4_4.0_1.0" => dph_erlang_from_mean(4, 4.0, 1.0),
                          "from_rates_general" => dph_from_rates([0.6, 0.4], [-2.0 1.0; 0.5 -1.5], 0.3))
            r = ref["rand_" * name]
            rng = Xoshiro(r["seed"])
            @test [rand(rng, d) for _ in 1:50] == r["draws"]
        end
    end

    @testset "continuous phase-type $name" for (name, d) in (
            "general_3" => PhaseType([0.5, 0.3, 0.2], [-1.0 0.5 0.0; 0.0 -2.0 1.0; 0.0 0.0 -1.5]),
            "exponential_2.0" => ph_exponential(2.0), "erlang_3_1.5" => ph_erlang(3, 1.5),
            "erlang_by_mean_4_2.0" => ph_erlang_by_mean(4, 2.0))
        r = ref["ph_" * name]
        xs = r["x"]
        @test agree(d.α, r["alpha"]) && agree(d.A, unflat(r["A"])) && agree(d.q, r["q"])
        @test agree(pdf.(d, xs), r["pdf"])
        @test agree(logpdf.(d, xs), r["logpdf"])
        @test agree(cdf.(d, xs), r["cdf"]; atol = 1e-15)
        @test agree(logcdf.(d, xs), r["logcdf"]; atol = 1e-15)
        @test mean(d) ≈ r["mean"] && var(d) ≈ r["var"]
        # invariants by quadrature
        @test quadgk(x -> pdf(d, x), 0, Inf)[1] ≈ 1 atol = 1e-8
        @test quadgk(x -> x * pdf(d, x), 0, Inf)[1] ≈ mean(d) rtol = 1e-7
        @test quadgk(x -> (x - mean(d))^2 * pdf(d, x), 0, Inf)[1] ≈ var(d) rtol = 1e-7
        @test quadgk(x -> pdf(d, x), 0, 1.3)[1] ≈ cdf(d, 1.3) rtol = 1e-8
        @test pdf(d, -1.0) == 0 && logpdf(d, 0.0) == -Inf && insupport(d, 1.0) && !insupport(d, -1.0)
    end

    @testset "continuous phase-type: Erlang closed form and its scope" begin
        A = [-1.5 1.5 0.0; 0.0 -1.5 1.5; 0.0 0.0 -1.5]
        e = PhaseType([1.0, 0.0, 0.0], A)
        for x in (0.2, 1.0, 3.0)
            @test pdf(e, x) ≈ ([1.0, 0.0, 0.0]' * exp(A * x) * e.q)[] rtol = 1e-10
        end
        # an Erlang generator entered in a later phase is not Erlang: density from exp(A t)
        # (the original module returned 0 here)
        m = PhaseType([0.0, 1.0, 0.0], A)
        @test pdf(m, 1.0) ≈ ([0.0, 1.0, 0.0]' * exp(A * 1.0) * m.q)[] rtol = 1e-10
        @test pdf(m, 1.0) ≈ pdf(ph_erlang(2, 1.5), 1.0) rtol = 1e-10
        # non-Erlang densities need exp(A t): Dual arguments raise an informative error
        g = PhaseType([0.5, 0.5], [-1.0 0.5; 0.0 -2.0])
        @test_throws ArgumentError ForwardDiff.derivative(x -> pdf(g, x), 1.0)
        # the Erlang closed form is differentiable
        @test ForwardDiff.derivative(x -> pdf(e, x), 1.0) isa Float64
        @test_throws DimensionMismatch PhaseType([1.0, 0.0], [-1.0 0.0 0.0])
        @test_throws ArgumentError PhaseType([1.0], reshape([1.0], 1, 1))
        @test_throws ArgumentError ph_exponential(0.0)
        @test_throws ArgumentError ph_erlang(0, 1.0)
    end

    @testset "smooth negative binomial $name" for (name, (rr, p)) in
            ("3.7_0.4" => (3.7, 0.4), "1.0_0.5" => (1.0, 0.5), "2.25_0.8" => (2.25, 0.8))
        r = ref["smooth_" * name]
        sd = dph_negative_binomial_smooth(rr, p)
        @test length(sd.dph_k.α) == r["k"] && sd.w ≈ r["w"]
        @test agree([pmf(sd, n) for n in ns], r["pmf"]; atol = 1e-300)
        @test agree([logpdf(sd, n) for n in ns], r["logpdf"])
        @test mean(sd) ≈ r["mean"] && var(sd) ≈ r["var"]
        @test sum(pdf(sd, n) for n in 1:500) ≈ 1 atol = 1e-10
    end

    @testset "smooth negative binomial: rand and r-derivative" begin
        sd = dph_negative_binomial_smooth(3.7, 0.4)
        rng = Xoshiro(11)
        x = [rand(rng, sd) for _ in 1:20_000]                 # threw MethodError in the original
        @test mean(x) ≈ mean(sd) rtol = 0.02
        # ∂pmf/∂r = pmf_{k+1} - pmf_k inside (k, k+1)
        dr = ForwardDiff.derivative(r -> pmf(dph_negative_binomial_smooth(r, 0.4), 9), 3.7)
        @test dr ≈ pmf(dph_negative_binomial(4, 0.4), 9) - pmf(dph_negative_binomial(3, 0.4), 9)
    end

    @testset "negbin_total_logpdf and rate_to_proportion" begin
        r = ref["negbin_total_logpdf_int"]
        @test agree([PTD.negbin_total_logpdf(k, rr, p) for (k, rr, p) in zip(r["k"], r["r"], r["p"])], r["value"])
        r = ref["negbin_total_logpdf_real"]
        @test agree([PTD.negbin_total_logpdf(k, r["r"], r["p"]) for k in r["k"]], r["value"])
        r = ref["rate_to_proportion"]
        @test agree(PTD.rate_to_proportion.(r["r"], r["t"]), r["value"])
        # p = 1: a deterministic number of trials (the original returned NaN at k = r)
        @test PTD.negbin_total_logpdf(3, 3, 1.0) == 0.0
        @test PTD.negbin_total_logpdf(4, 3, 1.0) == -Inf
        @test PTD.negbin_total_logpdf(3, 3.0, 1.0) == 0.0
        @test logpdf(dph_geometric(1.0), 1) == 0.0
    end

    @testset "Boxcar hazards reproduce both notebook truncations" begin
        for r in ref["hazard_negbin"]          # data-availability notebook: W | W ≤ K (log scale there)
            @test agree(PTD._hazard_negbin(0, r["r"], r["p"], r["K"]), r["h"]; rtol = 1e-13)
            d = dph_negative_binomial(r["r"], r["p"])
            @test agree(discrete_hazard(d, r["K"]; truncation = :condition), r["h"]; rtol = 1e-10)
        end
        # the ratio recursion of _hazard_negbin equals the log-scale route, in value and derivative
        for (c, r, p, K) in ((0, 1, 0.5, 60), (0, 4, 0.37, 204), (0, 10, 0.02, 500), (3, 2, 0.6, 40), (0, 3, 0.9, 15))
            logscale(x) = PTD._hazard_from_logpmf([PTD.negbin_total_logpdf(k - c, r, x) for k in 1:K])
            @test PTD._hazard_negbin(c, r, p, K) ≈ logscale(p) rtol = 1e-12
            @test ForwardDiff.derivative(x -> PTD._hazard_negbin(c, r, x, K), p) ≈ ForwardDiff.derivative(logscale, p) rtol = 1e-10
        end
        @test PTD._hazard_negbin(0, 3, 1.0, 6) == [0, 0, 1, 1, 1, 1]     # p = 1 (W = 3): the limit p → 1 above r
        @test PTD._hazard_negbin(0, 5, 0.5, 3) == [0, 0, 1]              # no mass up to K: only h[K] = 1
        # far below K the :condition hazard of a geometric is ≈ p, so dh/dp ≈ 1 (the log-scale
        # version gave -1 at p = 0.5, where the log-PMF and the log tail sum tie exactly)
        @test ForwardDiff.derivative(p -> discrete_hazard(dph_geometric(p), 500; truncation = :condition)[437], 0.5) ≈ 1
        @test ForwardDiff.derivative(x -> PTD._logaddexp(x, 1.0), 1.0) ≈ 0.5
        @test ForwardDiff.derivative(x -> PTD._logaddexp(1.0, x), 1.0) ≈ 0.5
        @test PTD._logaddexp(1.0, 1.0) ≈ 1 + log(2) && PTD._logaddexp(-Inf, 2.0) == 2.0 == PTD._logaddexp(2.0, -Inf)
        for r in ref["hazard_from_dph"]        # Hagelloch notebook: min(W, K), state iteration
            d = dph_negative_binomial_min_support(r["r"], r["p"], r["delta_min"])
            @test agree(discrete_hazard(d, r["K"]), r["h"]; rtol = 1e-13)
        end
        d = DPH([0.5, 0.5], [0.2 0.3; 0.1 0.6])
        K = 12
        # :absorb is the exact hazard below K
        h = discrete_hazard(d, K)
        @test h[1:K-1] ≈ [pmf(d, k) / ccdf(d, k - 1) for k in 1:K-1] rtol = 1e-12
        # :condition is the hazard of W | W ≤ K (generic path for a non-NegBin DPH)
        hc = discrete_hazard(d, K; truncation = :condition)
        p = [pmf(d, k) for k in 1:K]
        @test hc[1:K-1] ≈ [p[k] / sum(p[k:K]) for k in 1:K-1] rtol = 1e-12
        @test h[end] == 1 && hc[end] == 1
        # :condition raises the hazard near K (a geometric's own hazard is 0.6 at every age)
        g = dph_geometric(0.6)
        @test discrete_hazard(g, 25; truncation = :condition)[24] > 0.7
        @test_throws ArgumentError discrete_hazard(d, 0)
        @test_throws ArgumentError discrete_hazard(d, 5; truncation = :other)
    end

    @testset "DPH invariants: $name" for (name, mk) in DPH_CASES
        d = mk()
        nmax = 3000
        p = [pdf(d, n) for n in 1:nmax]
        @test sum(p) ≈ 1 atol = 1e-9
        @test sum((1:nmax) .* p) ≈ mean(d) rtol = 1e-8
        @test sum(((1:nmax) .- mean(d)) .^ 2 .* p) ≈ var(d) rtol = 1e-7
        @test [cdf(d, n) for n in 1:30] ≈ cumsum(p[1:30]) rtol = 1e-10
        @test [ccdf(d, n) for n in 1:30] ≈ 1 .- cumsum(p[1:30]) rtol = 1e-8 atol = 1e-15
        @test [logccdf(d, n) for n in 1:30] ≈ log.([ccdf(d, n) for n in 1:30]) rtol = 1e-8
        @test pdf(d, 0) == 0 && logpdf(d, 0) == -Inf && cdf(d, 0) == 0 && ccdf(d, 0) == 1
        @test minimum(d) == 1 && maximum(d) == Inf && insupport(d, 3) && !insupport(d, 0)
    end

    @testset "DPH constructions" begin
        g = dph_geometric(0.25)
        @test [logccdf(g, n) for n in 0:6] == [n * log1p(-g.s[1]) for n in 0:6]
        d = dph_negative_binomial_min_support(3, 0.45, 5)
        @test all(pdf(d, n) == 0 for n in 1:4) && pdf(d, 5) > 0
        @test mean(d) ≈ (5 - 3) + 3 / 0.45
        @test dph_erlang(3, 2.0, 0.1).S == dph_erlang_from_mean(3, 2.0, 0.1).S
        @test dph_negative_binomial_from_rates(3, 0.4, 1.0).S ≈ dph_negative_binomial(3, 0.4).S rtol = 1e-12
        @test dph_negative_binomial_from_mean(5, 12.5).s[end] ≈ 0.4
        @test mean(dph_erlang_from_mean(4, 4.0, 1.0)) ≈ 4 / (1 - exp(-1))
        @test mean(dph_exponential(2.0, 0.1)) ≈ 1 / (1 - exp(-0.2))
        @test dph_prepend_deterministic_delay(g, 0) === g
        # integer inputs are promoted (the original failed for integer p)
        @test dph_geometric(1) isa DPH{Float64}
        @test DPH([1], reshape([0], 1, 1)) isa DPH{Float64}
        # validation
        @test_throws DimensionMismatch DPH([1.0, 0.0], [0.5 0.0 0.0])
        @test_throws ArgumentError DPH([0.6, 0.6], [0.5 0.0; 0.0 0.5])
        @test_throws ArgumentError DPH([1.0], reshape([1.2], 1, 1))
        @test_throws ArgumentError DPH([1.0, 0.0], [0.5 -0.1; 0.0 0.5])
        @test_throws ArgumentError dph_geometric(0.0)
        @test_throws ArgumentError dph_negative_binomial(0, 0.5)
        @test_throws ArgumentError dph_negative_binomial(2, 1.5)
        @test_throws ArgumentError dph_negative_binomial_min_support(5, 0.5, 4)
        @test_throws ArgumentError dph_prepend_deterministic_delay(g, -1)
        @test_throws ArgumentError dph_negative_binomial_from_mean(5, 4.0)
        @test_throws ArgumentError dph_negative_binomial_from_rates(2, 1.0)
    end

    @testset "rand frequencies match the pmf" begin
        d = dph_negative_binomial_min_support(2, 0.5, 3)
        rng = Xoshiro(5)
        n = 40_000
        x = rand(rng, d, n)
        for k in 3:8
            pk = pdf(d, k)
            @test abs(count(==(k), x) / n - pk) < 4 * sqrt(pk * (1 - pk) / n)
        end
    end

    @testset "ForwardDiff through DPH constructions" begin
        k, r = 9, 4
        dlp = ForwardDiff.derivative(p -> logpdf(dph_negative_binomial(r, p), k), 0.45)
        @test dlp ≈ r / 0.45 - (k - r) / (1 - 0.45) rtol = 1e-10
        dsurv = ForwardDiff.derivative(p -> logccdf(dph_negative_binomial(1, p), 6), 0.3)
        @test dsurv ≈ -6 / (1 - 0.3) rtol = 1e-10
        gh = ForwardDiff.derivative(p -> sum(discrete_hazard(dph_negative_binomial(3, p), 200; truncation = :condition)), 0.7)
        @test isfinite(gh)
        ga = ForwardDiff.derivative(p -> sum(discrete_hazard(dph_negative_binomial(3, p), 40)), 0.2)
        @test isfinite(ga)
        gm = ForwardDiff.derivative(p -> mean(dph_negative_binomial(3, p)), 0.25)
        @test gm ≈ -3 / 0.25^2 rtol = 1e-10
        # a Dual number of stages is rounded: zero derivative with respect to r
        @test ForwardDiff.derivative(rr -> mean(dph_negative_binomial(rr, 0.5)), 3.2) == 0
        @test ForwardDiff.derivative(rr -> mean(dph_negative_binomial_min_support(rr, 0.5, 5)), 3.2) == 0
    end

    @testset "Coxian DPH (dph_coxian)" begin
        w, p = [0.1, 0.2, 0.7], 0.6
        d = dph_coxian(w, p; first_exit = 2)
        @test d.α == [1.0, 0.0, 0.0, 0.0] && size(d.S) == (4, 4)
        mix(n) = sum(w[i] * pmf(dph_negative_binomial(j, p), n) for (i, j) in enumerate(2:4))
        @test all(n -> isapprox(pmf(d, n), mix(n); rtol = 1e-10), 1:60)
        @test mean(d) ≈ sum(w .* (2:4)) / p
        @test sum(pmf(d, n) for n in 1:500) ≈ 1
        # logpdf takes the closed form of the mixture: exact, with no floor in the far tail
        @test PTD._shifted_negbin_structure(d) === nothing
        cx = PTD._coxian_structure(d)
        @test cx !== nothing && cx[1] ≈ p && cx[2] ≈ [0.0; w]
        @test logpdf(d, 1) == -Inf && logpdf(d, 0) == -Inf
        @test all(n -> isapprox(logpdf(d, n), log(mix(n)); rtol = 1e-12), 2:80)
        @test logpdf(d, 150) < log(1e-12) && isapprox(logpdf(d, 150), log(mix(150)); rtol = 1e-10)
        @test DiscreteDynamicSurvivalAnalysis.recovery_loglik(d, [2, 3, 5, 9], [false, true, false, true]) ≈
              logpdf(d, 2) + log(ccdf(d, 3)) + logpdf(d, 5) + log(ccdf(d, 9))
        # special cases: all the weight on the last stage is the serial negative binomial
        @test pmf.(dph_coxian([0.0, 0.0, 1.0], 0.4), 1:30) ≈ pmf.(dph_negative_binomial(3, 0.4), 1:30)
        @test pmf.(dph_coxian([1.0], 0.3; first_exit = 3), 1:30) ≈ pmf.(dph_negative_binomial(3, 0.3), 1:30)
        @test pmf(dph_coxian([0.5, 0.5], 1.0), 1) ≈ 0.5 && pmf(dph_coxian([0.5, 0.5], 1.0), 2) ≈ 0.5
        # a negligible exit weight keeps the rows of S sub-stochastic
        dt = dph_coxian([1e-20, 1 - 1e-20], 0.5; first_exit = 2)
        @test all(dt.s .>= 0) && sum(pmf(dt, n) for n in 1:500) ≈ 1
        # derivatives of the closed form, against central differences
        f(x) = sum(logpdf(dph_coxian([x[1], x[2], 1 - x[1] - x[2]], x[3]; first_exit = 2), n) for n in (2, 3, 5, 8))
        x0, h = [0.1, 0.2, 0.6], 1e-6
        fd = [(f(x0 .+ h .* (1:3 .== i)) - f(x0 .- h .* (1:3 .== i))) / 2h for i in 1:3]
        @test ForwardDiff.gradient(f, x0) ≈ fd rtol = 1e-6
        # input checks
        @test_throws ArgumentError dph_coxian(Float64[], 0.5)
        @test_throws ArgumentError dph_coxian([0.5, 0.6], 0.5)
        @test_throws ArgumentError dph_coxian([-0.1, 1.1], 0.5)
        @test_throws ArgumentError dph_coxian([1.0], 0.0)
        @test_throws ArgumentError dph_coxian([1.0], 1.5)
        @test_throws ArgumentError dph_coxian([1.0], 0.5; first_exit = 0)
        # other DPHs are not taken for common-p Coxians
        @test PTD._coxian_structure(DPH([0.5, 0.5], [0.2 0.3; 0.1 0.6])) === nothing
        @test PTD._coxian_structure(DPH([1.0, 0.0], [0.5 0.3; 0.0 0.4])) === nothing   # stage-specific progression
    end

    @testset "structure detection keeps stage-specific derivatives" begin
        # stage-specific progression probabilities that coincide at the evaluation point are not a
        # shared p: taken for one, the closed forms would drop the derivatives of all but one stage
        cox(x) = DPH([1.0, 0.0, 0.0], [1-x[1] 0.6x[1] 0; 0 1-x[2] 0.5x[2]; 0 0 1-x[3]])
        ser(x) = DPH([1.0, 0.0, 0.0], [1-x[1] x[1] 0; 0 1-x[2] x[2]; 0 0 1-x[3]])
        x0, h = [0.4, 0.4, 0.4], 1e-6
        @test PTD._coxian_structure(cox(x0)) !== nothing && PTD._shifted_negbin_structure(ser(x0)) !== nothing
        for build in (cox, ser)
            f(x) = sum(logpdf(build(x), n) for n in (2, 3, 5, 8))
            fd = [(f(x0 .+ h .* (1:3 .== i)) - f(x0 .- h .* (1:3 .== i))) / 2h for i in 1:3]
            @test ForwardDiff.gradient(f, x0) ≈ fd rtol = 1e-6
        end
        # the builders' DPHs keep their closed forms under ForwardDiff
        seen = Bool[]
        ForwardDiff.gradient([0.3, 0.6]) do x
            d1 = dph_coxian([x[1], 1 - x[1]], x[2]; first_exit = 2)
            d2 = dph_negative_binomial_min_support(3, x[2], 5)
            push!(seen, PTD._coxian_structure(d1) !== nothing, PTD._shifted_negbin_structure(d2) !== nothing)
            logpdf(d1, 4) + logpdf(d2, 6)
        end
        @test !isempty(seen) && all(seen)
    end

    @testset "no name clashes with Distributions/Statistics" begin
        m = Module()
        Core.eval(m, :(using Distributions, Statistics, DiscreteDynamicSurvivalAnalysis.PhaseTypeDistributions))
        @test Core.eval(m, :(mean(dph_geometric(0.5)))) ≈ 2
        @test Core.eval(m, :(cdf(dph_geometric(0.5), 1))) ≈ 0.5
        @test Core.eval(m, :(pdf(ph_exponential(1.0), 0.0) + pdf(dph_geometric(0.5), 2))) ≈ 0.25
        @test PTD.mean === Statistics.mean && PTD.cdf === Distributions.cdf
    end
end
