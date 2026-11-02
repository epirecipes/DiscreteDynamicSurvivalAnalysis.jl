# Build the documentation:
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl
# The manuscript vignettes are shown pre-rendered with the manuscript settings: their pages are written by
# docs/render_vignettes.jl (≈15 min on 10 threads) and are not rerun here. Only the external-data vignette
# runs during the build.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")    # headless GR

using Documenter, Literate, TOML, SHA
using DiscreteDynamicSurvivalAnalysis

const SRC = joinpath(@__DIR__, "src")
const VIGNETTES = ["data_availability.jl", "ddsa_vs_ode.jl", "h1n1_wsu.jl", "ebola_drc.jl",
                   "measles_hagelloch.jl", "external_data.jl"]   # page titles are the files' H1 headings
const LIVE = ["external_data.jl"]                                # executed in every build

# Literate source → Documenter markdown (executed by Documenter's @example blocks)
const VIGNETTE_DIR = joinpath(SRC, "vignettes")
for file in LIVE
    Literate.markdown(joinpath(VIGNETTE_DIR, file), VIGNETTE_DIR; documenter = true, credit = false)
end

# The other vignettes must have been rendered, with their figures in docs/src/assets/vignettes/; warn if a script
# changed after its page was rendered.
let rec_file = joinpath(VIGNETTE_DIR, "rendered.toml")
    rec = isfile(rec_file) ? TOML.parsefile(rec_file) : Dict{String,Any}()
    for file in setdiff(VIGNETTES, LIVE)
        name = first(splitext(file))
        rerender = "`julia --project=docs --threads=10 docs/render_vignettes.jl $name`"
        page = joinpath(VIGNETTE_DIR, name * ".md")
        isfile(page) || error("vignette $name has not been rendered: run $rerender")
        for m in eachmatch(r"!\[[^\]]*\]\((\.\./assets/vignettes/[^)]+)\)", read(page, String))
            isfile(joinpath(VIGNETTE_DIR, m.captures[1])) ||
                error("figure $(m.captures[1]) of vignette $name is missing: run $rerender")
        end
        source = bytes2hex(sha256(read(joinpath(VIGNETTE_DIR, file))))
        get(get(rec, name, Dict()), "source_sha256", "") == source ||
            @warn "docs/src/vignettes/$file changed after its page was rendered; re-render it with docs/render_vignettes.jl $name"
    end
end

DocMeta.setdocmeta!(DiscreteDynamicSurvivalAnalysis, :DocTestSetup, :(using DiscreteDynamicSurvivalAnalysis);
                    recursive = true)

makedocs(;
    sitename = "DiscreteDynamicSurvivalAnalysis.jl",
    authors = "Sandra Montes-Olivas, Simon Frost",
    modules = [DiscreteDynamicSurvivalAnalysis],
    checkdocs = :exports,
    remotes = isdir(joinpath(dirname(@__DIR__), ".git")) ? Dict() : nothing,   # source links need a git checkout
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true", size_threshold = 2^21, size_threshold_warn = 2^20,
                             example_size_threshold = 2^20),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Vignettes" => ["vignettes/" * replace(file, ".jl" => ".md") for file in VIGNETTES],
        "API reference" => "api.md",
        "Reproducibility" => "reproducibility.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo = "github.com/" * get(ENV, "GITHUB_REPOSITORY", "epirecipes/DiscreteDynamicSurvivalAnalysis.jl") * ".git",
               devbranch = "main", push_preview = false)
end
