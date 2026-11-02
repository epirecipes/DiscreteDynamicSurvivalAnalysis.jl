using Test, DiscreteDynamicSurvivalAnalysis

include("fixtures.jl")   # reference values from the original manuscript code

@testset "DiscreteDynamicSurvivalAnalysis.jl" begin
    @testset "PhaseTypeDistributions" include("test_phasetype.jl")
    @testset "IntegerParameters" include("test_integer_parameters.jl")
    @testset "data" include("test_data.jl")
    @testset "forward maps" include("test_forward_map.jl")
    @testset "likelihood" include("test_likelihood.jl")
    @testset "models" include("test_models.jl")
    @testset "simulation" include("test_simulate.jl")
    @testset "workflow" include("test_workflow.jl")
    @testset "plot recipes" include("test_recipes.jl")
    @testset "Aqua" include("test_aqua.jl")
end
