# Rerun the manuscript analyses with the full manuscript settings and compare the posteriors with the
# manuscript's posterior tables (test/reference/manuscript_posteriors.toml).
#
# Usage (from the package root; about 15 min on 10 threads):
#   julia --project=docs --threads=10 scripts/reproduce_manuscript.jl [vignette ...]
# with vignette names from: data_availability ddsa_vs_ode h1n1_wsu ebola_drc measles_hagelloch (default: all).
# Set DDSA_REPRO_OUT=path.toml to also write the comparison table.
#
# Each vignette (docs/src/vignettes/<name>.jl) runs in its own module with DDSA_FULL=true and leaves its
# posterior draws in `REPRODUCTION` (lines marked #src, which the documentation build drops).
#
# Criterion, per quantity, with the posterior sd of the new run: |Δ median| ≤ 0.1 sd and |Δ 2.5% / 97.5% limit|
# ≤ 0.2 sd, plus half a unit of the last digit printed in the manuscript table. Integer-valued quantities (r, N)
# may also differ by one unit. The mode of r must match; P(r) frequencies must agree within 0.03.
# Runs are not bitwise reproductions of the manuscript's chains, so differences are Monte Carlo error.
#
# A median or limit outside that tolerance still passes if |Δ| ≤ 2 combined Monte Carlo standard errors (plus the
# same rounding and integer slack). The combined MCSE is √2 × the MCSE of this run's estimate, assuming the manuscript
# run had a similar one. It comes from MCMCDiagnosticTools.mcse, with the pooled draws split back into the 4 chains of
# the manuscript settings. Such quantities are reported as "pass (MCSE)".

ENV["DDSA_FULL"] = "true"
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using TOML, Statistics, Printf
using MCMCDiagnosticTools: mcse

const ROOT = dirname(@__DIR__)
const REF = TOML.parsefile(joinpath(ROOT, "test", "reference", "manuscript_posteriors.toml"))
const ALL = ["data_availability", "ddsa_vs_ode", "h1n1_wsu", "ebola_drc", "measles_hagelloch"]

"Run one vignette in a fresh module; return its REPRODUCTION dictionary and the wall time."
function run_vignette(name)
    path = joinpath(ROOT, "docs", "src", "vignettes", name * ".jl")
    isfile(path) || error("vignette not found: $path")
    m = Module(Symbol("Vignette_", name))
    println("\n=== running $name (manuscript settings) ===")
    t = @elapsed Base.include(m, path)
    @printf("=== %s finished in %.1f s ===\n", name, t)
    return Base.invokelatest(getfield, m, :REPRODUCTION), t
end

"The table of manuscript_posteriors.toml at a dotted key, e.g. \"data_availability.A_negbin\"."
reftable(key) = foldl((d, k) -> d[k], split(key, '.'); init = REF)

"Half a unit of the last printed digit of a transcribed value."
function rounding(x::Real)
    s = string(Float64(x))
    occursin('e', s) && return 0.0
    dec = length(split(s, '.')[2])
    return 0.5 * 10.0^(-dec)
end

isintegral(x) = all(v -> isinteger(v), x)

const NCHAINS = 4      # chains per fit in the manuscript settings; the vignettes pool them chain after chain

"Monte Carlo standard errors of the median and the 2.5% and 97.5% quantiles of pooled draws (0 if undefined)."
function quantile_mcse(x)
    length(x) % NCHAINS == 0 || error("$(length(x)) pooled draws cannot be split into $NCHAINS chains")
    X = reshape(x, :, NCHAINS, 1)                                   # draws × chains × parameters
    se = [mcse(X; kind = Base.Fix2(quantile, p))[1] for p in (0.5, 0.025, 0.975)]
    return [isfinite(s) ? s : 0.0 for s in se]                      # constant draws give NaN
end

function compare(key, draws)
    ref = reftable(key)
    rows = NamedTuple[]
    if key == "ddsa_vs_ode.poisson_identity"
        for (q, v) in ref
            new = Float64(draws[q])
            ok = abs(new - v) ≤ 1e-3 + rounding(v)
            push!(rows, (table = key, quantity = q, ref = [v], new = [new], score = abs(new - v), mc_score = NaN,
                         pass = ok, by_mcse = false))
        end
        return rows
    end
    for (q, v) in sort(collect(ref); by = first)
        q in ("source", "wall_seconds", "sd") && continue
        if q == "r_mode"
            r = Int.(draws["r"])
            mode = argmax(k -> count(==(k), r), unique(r))
            push!(rows, (table = key, quantity = q, ref = [v], new = [mode], score = abs(mode - v), mc_score = NaN,
                         pass = mode == v, by_mcse = false))
        elseif q == "P_r"
            r = Int.(draws["r"])
            freq = [count(==(k), r) / length(r) for k in 1:length(v)]
            d = maximum(abs.(freq .- v))
            push!(rows, (table = key, quantity = q, ref = Float64.(v), new = round.(freq; digits = 3), score = d,
                         mc_score = NaN, pass = d ≤ 0.03, by_mcse = false))
        else
            haskey(draws, q) || (push!(rows, (table = key, quantity = q, ref = Float64.(v), new = Float64[],
                                              score = NaN, mc_score = NaN, pass = false, by_mcse = false)); continue)
            x = Float64.(draws[q])
            new = [median(x), quantile(x, 0.025), quantile(x, 0.975)]
            sd = std(x)
            slack = isintegral(x) ? 1.0 : 0.0
            tol = [0.1sd, 0.2sd, 0.2sd] .+ rounding.(v) .+ slack
            Δ = abs.(new .- v)
            score = sd > 0 ? maximum(Δ ./ [0.1sd, 0.2sd, 0.2sd]) : maximum(Δ)
            mc = sqrt(2) .* quantile_mcse(x)                       # combined MCSE of this run and the manuscript's
            excess = max.(Δ .- rounding.(v) .- slack, 0.0)         # beyond the rounding of the published value
            mc_ratio = [mc[i] > 0 ? excess[i] / mc[i] : (excess[i] > 0 ? Inf : 0.0) for i in 1:3]
            within_sd = Δ .≤ tol
            pass = all(within_sd .| (mc_ratio .≤ 2))
            mc_score = all(within_sd) ? NaN : maximum(mc_ratio[.!within_sd])   # only the checks that need the rule
            push!(rows, (table = key, quantity = q, ref = Float64.(v), new = new, score = score, mc_score = mc_score,
                         pass = pass, by_mcse = pass && !all(within_sd)))
        end
    end
    return rows
end

fmt(v) = isempty(v) ? "—" : "[" * join((@sprintf("%.4g", x) for x in v), ", ") * "]"

function print_rows(rows)
    println("\n", "="^120)
    @printf("%-34s %-14s %-30s %-30s %7s %8s  %s\n", "table", "quantity", "manuscript [med, 2.5%, 97.5%]",
            "this run", "score", "Δ/MCSE₂", "result")
    println("-"^120)
    for r in rows
        @printf("%-34s %-14s %-30s %-30s %7.2f %8s  %s\n", r.table, r.quantity, fmt(r.ref), fmt(r.new), r.score,
                isnan(r.mc_score) ? "—" : @sprintf("%.2f", r.mc_score),
                r.pass ? (r.by_mcse ? "pass (MCSE)" : "pass") : "FAIL")
    end
    println("-"^120)
end

function main(names)
    rows = NamedTuple[]
    walls = Pair{String,Float64}[]
    for name in names
        repro, t = run_vignette(name)
        push!(walls, name => t)
        new_rows = NamedTuple[]
        for key in sort(collect(keys(repro)))
            # the draws may be types loaded by the vignette (e.g. AxisArrays), newer than this function
            append!(new_rows, Base.invokelatest(compare, key, repro[key]))
        end
        Base.invokelatest(print_rows, new_rows)
        append!(rows, new_rows)
    end
    println("\nAll analyses:")
    print_rows(rows)
    npass = count(r -> r.pass, rows)
    println("$npass of $(length(rows)) quantities pass, $(count(r -> r.by_mcse, rows)) of them only by the Monte Carlo ",
            "rule. score = largest |Δ| in units of the tolerance (0.1 sd for the median, 0.2 sd for the limits; ≤ 1 ",
            "passes before rounding/integer slack); Δ/MCSE₂ = for the checks outside that tolerance, the largest |Δ| ",
            "beyond the rounding of the published value, in combined Monte Carlo standard errors (≤ 2 passes).")
    println("\nWall times (this run): ", join(("$n $(round(t; digits = 1)) s" for (n, t) in walls), ", "))
    out = get(ENV, "DDSA_REPRO_OUT", "")
    if !isempty(out)
        open(out, "w") do io
            TOML.print(io, Dict("rows" => [Dict(String(k) => (v isa Vector ? collect(v) : v) for (k, v) in pairs(r))
                                           for r in rows],
                                "wall_seconds" => Dict(walls)))
        end
        println("wrote $out")
    end
    return npass == length(rows)
end

names = isempty(ARGS) ? ALL : ARGS
for n in names
    n in ALL || error("unknown vignette $n; choose from $(join(ALL, ", "))")
end
ok = main(names)
exit(ok ? 0 : 1)
