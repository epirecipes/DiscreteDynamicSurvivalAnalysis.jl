using DiscreteDynamicSurvivalAnalysis
using Aqua, Test

@testset "Aqua" begin
    # The persistent-task check precompiles the package (and Turing) in a fresh environment, which
    # takes longer than Aqua's default 30 s limit.
    Aqua.test_all(DiscreteDynamicSurvivalAnalysis; persistent_tasks = (tmax = 300,))
end
