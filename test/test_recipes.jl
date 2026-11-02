using DiscreteDynamicSurvivalAnalysis
using RecipesBase, Test

# Plots.jl normally defines which attributes a backend supports; without it, accept all of them.
RecipesBase.is_key_supported(::Symbol) = true

@testset "plot recipes (RecipesBase extension)" begin
    @test Base.get_extension(DiscreteDynamicSurvivalAnalysis, :DiscreteDynamicSurvivalAnalysisRecipesBaseExt) !== nothing
    pp = PosteriorPredictive(collect(0.0:4.0), [1, 3, 5, 2, 0], rand(5, 20) .* 5, rand(1:30, 20), 11, :expected)
    series = RecipesBase.apply_recipe(Dict{Symbol,Any}(), pp)
    @test length(series) == 2
    @test series[1].plotattributes[:seriestype] === :bar
    @test series[1].args == (pp.times, pp.observed)
    @test series[2].plotattributes[:seriestype] === :path
    @test series[2].args[2] == predictive_quantiles(pp)[:, 2]
    @test series[2].plotattributes[:label] == "posterior median"

    ## the :expected curve is labelled as counting all infections when some infection times are unknown
    label(scale, K) = RecipesBase.apply_recipe(Dict{Symbol,Any}(),
        PosteriorPredictive(pp.times, pp.observed, pp.incidence, pp.final_size, K, scale))[2].plotattributes[:label]
    @test label(:expected, 14) == "posterior median (all infections)"
    @test label(:expected_known, 14) == label(:observed_total, 14) == label(:expected, 11) == "posterior median"

    rp = RecoveryPredictive(1:6, rand(6, 10), [0, 2, 4, 1, 0, 1])
    series = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rp)
    @test length(series) == 2
    @test series[1].args[2] ≈ rp.observed ./ 8
    @test series[2].args == (collect(1:6), predictive_quantiles(rp)[:, 2])
end
