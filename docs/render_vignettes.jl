# Pre-render the manuscript vignettes with the manuscript settings. Each Literate script in
# docs/src/vignettes/ is run once with DDSA_FULL=true and written with its output to
# docs/src/vignettes/<name>.md; its figures go to docs/src/assets/vignettes/<name>-<k>.png. docs/make.jl shows
# these pages without running them. rendered.toml records when and how each page was rendered, and the hash of
# its script.
#
#   julia --project=docs --threads=10 docs/render_vignettes.jl [name ...]           # ≈15 min on 10 threads
#   julia --project=docs --threads=auto docs/render_vignettes.jl --check [name ...]  # reduced settings, writes nothing
#
# Re-render after any change that can affect the results (package code, a vignette script, package versions)
# and before a release. `--check` runs each vignette with its reduced settings, as a quick test that the code
# still works (used by CI); its results are not written anywhere.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")    # headless GR

using Literate, TOML, SHA, Dates, Printf, Base64
using DiscreteDynamicSurvivalAnalysis

const DIR = joinpath(@__DIR__, "src", "vignettes")
const FIG_DIR = joinpath(@__DIR__, "src", "assets", "vignettes")    # linked from the pages as ../assets/vignettes/
const RENDERED = ["data_availability", "ddsa_vs_ode", "h1n1_wsu", "ebola_drc", "measles_hagelloch"]
const RECORD = joinpath(DIR, "rendered.toml")

source_hash(name) = bytes2hex(sha256(read(joinpath(DIR, name * ".jl"))))

# Plots.jl figures come out as HTML with an embedded base64 PNG; store each as a file in FIG_DIR.
function extract_figures(content, name)
    k = 0
    rx = r"```@raw html\n<img src=\"data:image/png;base64,([A-Za-z0-9+/=]+)\" />\n```"
    mkpath(FIG_DIR)
    return replace(content, rx => function (block)
        k += 1
        file = "$name-$k.png"
        write(joinpath(FIG_DIR, file), base64decode(match(rx, block).captures[1]))
        return "![](../assets/vignettes/$file)"
    end)
end

# vignettes whose MCMC settings differ from the manuscript's
const SETTINGS = Dict("measles_hagelloch" => "the manuscript's MCMC settings, with a target acceptance rate of 0.95,")

function note(name)
    settings = get(SETTINGS, name, "the manuscript's MCMC settings")
    return """
    !!! note "Precomputed with the manuscript settings"
        The output on this page was precomputed using $settings and is
        displayed without being re-evaluated. To rerun it, see [Reproducibility](../reproducibility.md).
    """
end

# the note goes right under the page title
add_note(content, name) =
    replace(content, r"^# .*$"m => title -> title * "\n\n" * rstrip(note(name)); count = 1)

function render(name)
    if isdir(FIG_DIR)                              # figures of the previous rendering
        for f in readdir(FIG_DIR)
            startswith(f, name * "-") && endswith(f, ".png") && rm(joinpath(FIG_DIR, f))
        end
    end
    t0 = time()
    post(content) = add_note(extract_figures(content, name), name)
    Literate.markdown(joinpath(DIR, name * ".jl"), DIR; documenter = true, execute = true, credit = false,
                      postprocess = post)
    return time() - t0
end

function record!(name, seconds)
    rec = isfile(RECORD) ? TOML.parsefile(RECORD) : Dict{String,Any}()
    rec[name] = Dict("source_sha256" => source_hash(name), "rendered" => string(round(now(), Second)),
                     "julia" => string(VERSION),
                     "package_version" => string(pkgversion(DiscreteDynamicSurvivalAnalysis)),
                     "threads" => Threads.nthreads(), "seconds" => round(seconds; digits = 1))
    open(io -> TOML.print(io, rec; sorted = true), RECORD, "w")
end

function check(name)
    t = @elapsed Base.include(Module(Symbol("Check_", name)), joinpath(DIR, name * ".jl"))
    @printf("%s ran with its reduced settings in %.0f s\n", name, t)
end

args = copy(ARGS)
check_only = "--check" in args
filter!(!=("--check"), args)
names = isempty(args) ? RENDERED : args
for n in names
    n in RENDERED || error("unknown vignette $n; choose from $(join(RENDERED, ", "))")
end
if check_only
    ENV["DDSA_FULL"] = "false"
    foreach(check, names)
else
    ENV["DDSA_FULL"] = "true"
    for n in names
        @info "rendering $n with the manuscript settings"
        s = render(n)
        record!(n, s)
        @info @sprintf("%s rendered in %.1f min", n, s / 60)
    end
end
