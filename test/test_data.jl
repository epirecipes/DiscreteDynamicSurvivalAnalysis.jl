using DiscreteDynamicSurvivalAnalysis
using CSV, DataFrames, Dates, Test

@testset "linelist reproduces the notebook preprocessing" begin
    dat = fixture("data")

    @testset "H1N1 (WSU)" begin
        d = dat["h1n1"]
        df = CSV.read(example_data_path(:wsu), DataFrame)
        data = linelist(df.V1, df.V2; nsteps = 104, censor_at = 104.0)
        @test data.nsteps == d["nsteps"]
        @test data.infection_counts == d["th"]
        @test data.final_size == d["K"]
        @test data.recovery_intervals == d["delta"]
        @test data.censored == Bool.(d["cens"])
        @test isempty(data.removal_only_steps)
    end

    @testset "Ebola wave $w" for w in 1:3
        d = dat["ebola_w$w"]
        df = CSV.read(example_data_path(Symbol("ebola_wave$w")), DataFrame)
        onset = ifelse.(df.event1 .== 1, df.onset, missing)
        hosp = ifelse.(df.event2 .== 1, df.hosp, missing)
        data = linelist(onset, hosp)
        @test data.nsteps == d["nsteps"]
        @test data.infection_counts == d["th_onset"]
        @test sum(data.infection_counts) == d["K_onset"]
        @test data.final_size == d["K_total"]
        @test data.recovery_intervals == d["delta_complete"]
        @test !any(data.censored)
        @test data.removal_only_steps == d["T_hosp_only"]
    end

    @testset "Hagelloch (three classes)" begin
        d = dat["hagelloch"]
        df = CSV.read(example_data_path(:hagelloch), DataFrame)
        cls = parse.(Int, string.(df.class))          # the column may parse as Int or String
        groups = map(0:2) do c
            sub = df[cls .== c, :]
            linelist(sub.t_infection, sub.t_infection .+ sub.inf_period; nsteps = 102)
        end
        @test all(g -> g.nsteps == d["nsteps"], groups)
        for (g, data) in zip(0:2, groups)
            @test data.infection_counts == d["th$g"]
            @test data.final_size == d["K$g"]
        end
        @test reduce(vcat, [g.recovery_intervals for g in groups]) == d["delta"]
    end
end

@testset "linelist binning and validation" begin
    onset = [0.2, 1.5, 2.1, missing, 4.8]
    removal = [4.0, missing, 7.9, 6.3, 9.5]
    data = linelist(onset, removal; nsteps = 8, censor_at = 8.0)
    @test data.infection_counts == [1, 1, 1, 0, 1, 0, 0, 0]
    @test data.final_size == 5
    @test data.recovery_intervals == [4, 6, 3]       # 3.8 → 4; 5.8 → 6; censored at 8 − 4.8 = 3.2 → 3
    @test data.censored == [false, false, true]
    @test data.removal_only_steps == [7]
    @test linelist([0.0, 0.5], [0.2, 0.5]).recovery_intervals == [1, 1]   # same-step removals count as one step
    @test linelist([0.0, 3.0], [1.0, 4.0]).nsteps == 4                    # default grid: last infection step
    @test linelist([0.0, 1.0], [2.0, 4.0]; Δt = 2.0).recovery_intervals == [1, 2]
    @test linelist([missing, 1.0], [30.0, 2.0]; nsteps = 5).removal_only_steps == [5]   # clamped to the grid

    @test_throws DimensionMismatch linelist([1.0, 2.0], [3.0])
    @test_throws ArgumentError linelist(Float64[], Float64[])
    @test_throws ArgumentError linelist([-1.0], [2.0])
    @test_throws ArgumentError linelist([1.0], [Inf])
    @test_throws ArgumentError linelist([missing, 1.0], [missing, 2.0])
    @test_throws ArgumentError linelist([3.0], [2.0])
    @test_throws ArgumentError linelist([9.5], [10.0]; nsteps = 5)
    @test_throws ArgumentError linelist([1.0, 8.0], [2.0, 9.0]; censor_at = 8.0)
    @test_throws ArgumentError linelist([1.0], [2.0]; Δt = 0)
    @test_throws ArgumentError linelist(["a"], [2.0])
end

@testset "linelist with dates (origin)" begin
    same(a, b) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(LineList))
    o = Date(2024, 3, 4)
    onset = [o, o + Day(1), o + Day(2), missing, o + Day(4)]
    removal = [o + Day(4), missing, o + Day(9), o + Day(6), o + Day(10)]
    days = linelist([0.0, 1.0, 2.0, missing, 4.0], [4.0, missing, 9.0, 6.0, 10.0]; nsteps = 8, censor_at = 8.0)
    @test same(linelist(onset, removal; origin = o, nsteps = 8, censor_at = o + Day(8)), days)
    @test same(linelist(onset, removal; origin = o, nsteps = 8, censor_at = 8.0), days)   # censor_at in days
    @test linelist(onset, removal; origin = o, nsteps = 8, censor_at = o + Day(8)).censored == [false, true, true]

    ## DateTime: fractions of a day, binned with floor and rounded intervals as for real times
    dt = linelist([DateTime(2024, 3, 4, 12), DateTime(2024, 3, 5, 18)],
                  [DateTime(2024, 3, 6, 18), DateTime(2024, 3, 7)]; origin = o)
    @test same(dt, linelist([0.5, 1.75], [2.75, 3.0]))
    @test dt.infection_counts == [1, 1] && dt.recovery_intervals == [2, 1]
    @test same(linelist([o + Day(1)], [o + Day(3)]; origin = DateTime(o)), linelist([1.0], [3.0]))

    ## Δt is in days: weekly bins
    wk = linelist(o .+ Day.([0, 6, 7, 13, 14]), o .+ Day.([7, 13, 21, 20, 28]); origin = o, Δt = 7)
    @test wk.infection_counts == [2, 2, 1] && wk.recovery_intervals == [1, 1, 2, 1, 2]

    @test_throws ArgumentError linelist([o], [o + Day(3)])                        # dates need an origin
    @test_throws ArgumentError linelist([1.0], [3.0]; censor_at = o)             # date censor_at needs an origin
    @test_throws ArgumentError linelist([1.0], [3.0]; origin = o)                # numbers with an origin
    @test_throws ArgumentError linelist([o - Day(1)], [o + Day(3)]; origin = o)  # before the origin
    @test_throws ArgumentError linelist([o], [o - Day(1)]; origin = o)
    @test_throws ArgumentError linelist([o], [o + Day(3)]; origin = o, censor_at = o - Day(1))
    @test_throws ArgumentError linelist([o + Day(2)], [o]; origin = o)           # removal before infection
    @test_throws ArgumentError linelist([o + Day(9)], [o + Day(10)]; origin = o, nsteps = 5)
end

@testset "LineList constructor" begin
    data = LineList([1, 3, 6, 4, 2, 0]; recovery_intervals = [3, 4, 4, 5, 2], final_size = 16)
    @test data.nsteps == 6 && data.Δt == 1.0 && data.final_size == 16
    @test data.censored == falses(5)
    @test LineList([1, 2]; removal_only_steps = [3]).final_size == 4
    @test_throws ArgumentError LineList([1, -1])
    @test_throws ArgumentError LineList(Int[])
    @test_throws ArgumentError LineList([1, 2]; recovery_intervals = [0])
    @test_throws DimensionMismatch LineList([1, 2]; recovery_intervals = [1, 2], censored = [true])
    @test_throws ArgumentError LineList([1, 2]; removal_only_steps = [0])
    @test_throws ArgumentError LineList([1, 2]; final_size = 2)
    @test_throws ArgumentError LineList([1, 2]; Δt = -1.0)

    s = sprint(show, LineList([1, 2]; recovery_intervals = [2, 3], censored = [false, true], removal_only_steps = [2]))
    @test occursin("nsteps = 2", s) && occursin("final_size = 4", s)
    @test occursin("1 censored", s) && occursin("removal-only cases = 1", s)
end

@testset "example_data_path" begin
    # every data set is read by the preprocessing tests above; only the unknown name is left
    @test_throws ArgumentError example_data_path(:nope)
end
