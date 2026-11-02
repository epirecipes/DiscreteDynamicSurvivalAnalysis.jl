module DiscreteDynamicSurvivalAnalysisRecipesBaseExt

# Plot recipes (loaded with RecipesBase, e.g. by `using Plots`): `plot(pp)` for a
# PosteriorPredictive and `plot(rp)` for a RecoveryPredictive.

using DiscreteDynamicSurvivalAnalysis: PosteriorPredictive, RecoveryPredictive, predictive_quantiles
using RecipesBase

@recipe function f(pp::PosteriorPredictive; probs = (0.025, 0.5, 0.975))
    q = predictive_quantiles(pp; probs = probs)
    xguide --> "time"
    yguide --> "cases per step"
    legend --> :topright
    @series begin
        seriestype := :bar
        label --> "observed"
        linewidth --> 0
        fillalpha --> 0.5
        pp.times, pp.observed
    end
    @series begin
        seriestype := :path
        # the :expected curve counts all cases, the bars only those with a known infection time
        all_cases = pp.scale === :expected && sum(pp.observed) < pp.observed_final_size
        label --> (all_cases ? "posterior median (all infections)" : "posterior median")
        ribbon := (q[:, 2] .- q[:, 1], q[:, 3] .- q[:, 2])
        fillalpha --> 0.3
        linewidth --> 2
        pp.times, q[:, 2]
    end
end

@recipe function f(rp::RecoveryPredictive; probs = (0.025, 0.5, 0.975))
    q = predictive_quantiles(rp; probs = probs)
    n = sum(rp.observed)
    xguide --> "recovery interval (steps)"
    yguide --> "probability"
    @series begin
        seriestype := :bar
        label --> "observed"
        linewidth --> 0
        fillalpha --> 0.5
        collect(rp.intervals), n > 0 ? rp.observed ./ n : zeros(length(rp.intervals))
    end
    @series begin
        seriestype := :path
        label --> "posterior median pmf"
        ribbon := (q[:, 2] .- q[:, 1], q[:, 3] .- q[:, 2])
        fillalpha --> 0.3
        linewidth --> 2
        markershape --> :circle
        collect(rp.intervals), q[:, 2]
    end
end

end # module
