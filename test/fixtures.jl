# Reference values produced by the original manuscript code (see test/reference/generate_reference.jl).
using TOML

const REFDIR = joinpath(@__DIR__, "reference")
const FIXTURE_CACHE = Dict{String,Any}()
fixture(name) = get!(() -> TOML.parsefile(joinpath(REFDIR, name * ".toml")), FIXTURE_CACHE, name)

"Matrix stored by the generator as `{size, data}` (column-major)."
unflat(d) = reshape(Float64.(d["data"]), Tuple(d["size"]))

"Elementwise comparison that treats equal infinities as equal."
function agree(a, b; rtol = 1e-12, atol = 0.0)
    length(a) == length(b) || return false
    all(zip(a, b)) do (x, y)
        (isinf(x) || isinf(y)) ? x == y : isapprox(x, y; rtol = rtol, atol = atol)
    end
end
agree(a::Number, b::Number; kw...) = agree([a], [b]; kw...)
