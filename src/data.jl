# Line-list data on the calendar grid of the forward map.

"""
    LineList

Binned case line list of one population (or one group of a multi-group model), on a grid of
`nsteps` steps of length `Δt`. Step `k` covers the time interval `[(k-1)Δt, kΔt)` measured from the
start of observation (with `Δt = 1` day, step 1 is day 0).

# Fields
- `nsteps::Int`: number of steps of the forward map (the observation window).
- `Δt::Float64`: step length, in the time unit of the data (e.g. days).
- `infection_counts::Vector{Int}`: number of cases whose infection (onset) time falls in each
  step (length `nsteps`): the data of the infection-time term ℓ_I.
- `final_size::Int`: the number of cases `K` in the final-size term ℓ_constr, including cases
  whose infection time is unknown.
- `recovery_intervals::Vector{Int}`: observed infection-to-removal intervals, in steps (`≥ 1`):
  the data of the recovery term ℓ_R. These may include initial infectives.
- `censored::Vector{Bool}`: `true` where the interval is right-censored (the case was still
  infectious after `recovery_intervals[i]` steps, at the end of observation).
- `removal_only_steps::Vector{Int}`: removal step of each case whose infection time is unknown;
  their likelihood sums over the possible infection steps (convolution term).

Construct it directly from binned data with the keyword constructor below, or from raw event
times with [`linelist`](@ref).
"""
struct LineList
    nsteps::Int
    Δt::Float64
    infection_counts::Vector{Int}
    final_size::Int
    recovery_intervals::Vector{Int}
    censored::Vector{Bool}
    removal_only_steps::Vector{Int}

    function LineList(nsteps, Δt, infection_counts, final_size, recovery_intervals, censored, removal_only_steps)
        nsteps ≥ 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
        Δt > 0 || throw(ArgumentError("step length Δt must be > 0; got $Δt"))
        length(infection_counts) == nsteps || throw(DimensionMismatch(
            "infection_counts must have one entry per step: length $(length(infection_counts)) ≠ nsteps = $nsteps"))
        all(≥(0), infection_counts) || throw(ArgumentError("infection_counts must be non-negative"))
        all(≥(1), recovery_intervals) || throw(ArgumentError("recovery intervals must be ≥ 1 step"))
        length(censored) == length(recovery_intervals) || throw(DimensionMismatch(
            "censored must have one entry per recovery interval: $(length(censored)) ≠ $(length(recovery_intervals))"))
        all(≥(1), removal_only_steps) || throw(ArgumentError("removal_only_steps must be ≥ 1"))
        final_size ≥ sum(infection_counts) + length(removal_only_steps) || throw(ArgumentError(
            "final_size = $final_size is smaller than the number of cases with a known infection time " *
            "($(sum(infection_counts))) plus those with a removal time only ($(length(removal_only_steps)))"))
        return new(nsteps, Δt, infection_counts, final_size, recovery_intervals, censored, removal_only_steps)
    end
end

"""
    LineList(infection_counts; final_size = sum(infection_counts) + length(removal_only_steps),
             recovery_intervals = Int[], censored = falses(length(recovery_intervals)),
             removal_only_steps = Int[], Δt = 1.0)

Build a [`LineList`](@ref) from already-binned data. `nsteps` is `length(infection_counts)`.

Throws `ArgumentError` for negative counts, intervals below one step, removal steps below 1, or a
`final_size` smaller than the number of cases implied by the other fields, and
`DimensionMismatch` if `censored` and `recovery_intervals` differ in length.

# Example
```julia
data = LineList([1, 3, 6, 4, 2, 0]; recovery_intervals = [3, 4, 4, 5, 2], final_size = 16)
```
"""
function LineList(infection_counts::AbstractVector{<:Integer};
                  removal_only_steps::AbstractVector{<:Integer} = Int[],
                  final_size::Integer = sum(infection_counts) + length(removal_only_steps),
                  recovery_intervals::AbstractVector{<:Integer} = Int[],
                  censored::AbstractVector{Bool} = falses(length(recovery_intervals)),
                  Δt::Real = 1.0)
    return LineList(length(infection_counts), Float64(Δt), collect(Int, infection_counts), Int(final_size),
                    collect(Int, recovery_intervals), collect(Bool, censored), collect(Int, removal_only_steps))
end

function Base.show(io::IO, d::LineList)
    print(io, "LineList(nsteps = $(d.nsteps), Δt = $(d.Δt), final_size = $(d.final_size), ",
          "known infection times = $(sum(d.infection_counts)), recovery intervals = $(length(d.recovery_intervals))")
    any(d.censored) && print(io, " ($(count(d.censored)) censored)")
    isempty(d.removal_only_steps) || print(io, ", removal-only cases = $(length(d.removal_only_steps))")
    print(io, ")")
end

"""Recovery intervals that are not censored."""
observed_intervals(d::LineList) = d.recovery_intervals[.!d.censored]

"""
    linelist(infection_times, removal_times; origin = nothing, Δt = 1.0, nsteps = nothing,
             censor_at = nothing) -> LineList

Bin a case line list given as event times into a [`LineList`](@ref). Times are real numbers in
the time unit of `Δt`, measured from the start of observation (time 0 is the start of step 1),
or calendar dates with `origin`; `missing` marks an unknown time. Each entry is one case.

- **Infection step**: `floor(t / Δt) + 1` (calendar binning: with `Δt = 1` day, the cases of day
  `d` are in step `d + 1`).
- **Grid length**: `nsteps` if given, otherwise the step of the last infection time. An infection
  time beyond the grid is an error (filter the data to the observation window first).
- **Recovery interval** (both times known): `max(1, round(Int, (t_rem - t_inf) / Δt))` steps,
  i.e. rounded to the nearest step, with same-step removals counted as one step. If `censor_at`
  is given and `t_rem ≥ censor_at`, the interval is right-censored at
  `max(1, round(Int, (censor_at - t_inf) / Δt))`.
- **Infection time missing, removal known**: the case enters `removal_only_steps` at step
  `floor(t_rem / Δt) + 1`, clamped to `1:nsteps`.
- **Removal time missing**: the case contributes its infection time only (no recovery term). To
  treat it as still infectious at the end of observation instead, give a removal time
  `≥ censor_at`.
- **Final size**: the number of cases.
- **Dates**: with `origin` (a `Date` or `DateTime`, the start of step 1), the times must be
  `Date`s or `DateTime`s (or `missing`). They are converted to days since `origin`, so `Δt` is in
  days (`Δt = 7` bins by week). `censor_at` may then be a date or a number of days since
  `origin`; `nsteps` is still a number of steps.

Throws `DimensionMismatch` if the two vectors differ in length, and `ArgumentError` for negative
or non-finite times, a removal before infection, a case with both times missing, an infection
time beyond the grid or at/after `censor_at`, an empty line list, dates without `origin` (or
numbers with it), or a date before `origin`.

# Examples
```julia
onset   = [0.2, 1.5, 2.1, missing, 4.8]
removal = [4.0, missing, 7.9, 6.3, 9.5]
data = linelist(onset, removal; nsteps = 8, censor_at = 8.0)

using Dates
day0 = Date(2024, 3, 4)
onset   = [day0, day0 + Day(2), missing]
removal = [day0 + Day(5), day0 + Day(9), day0 + Day(6)]
data = linelist(onset, removal; origin = day0, nsteps = 7, censor_at = day0 + Day(7))
```
"""
function linelist(infection_times::AbstractVector, removal_times::AbstractVector;
                  origin::Union{Nothing,Date,DateTime} = nothing,
                  Δt::Real = 1.0, nsteps::Union{Nothing,Integer} = nothing,
                  censor_at::Union{Nothing,Real,Date,DateTime} = nothing)
    if origin === nothing
        censor_at isa Union{Date,DateTime} && throw(ArgumentError(
            "censor_at = $censor_at is a date: give `origin`, the date at which step 1 starts"))
        return _linelist(infection_times, removal_times, Δt, nsteps, censor_at)
    end
    days(t, name) = _days_since(t, origin, name)
    return _linelist([days(t, "infection") for t in infection_times],
                     [days(t, "removal") for t in removal_times], Δt, nsteps,
                     censor_at isa Union{Date,DateTime} ? days(censor_at, "censor_at") : censor_at)
end

# Days from `origin` to the date `t` (`missing` stays `missing`), for `linelist(...; origin)`.
function _days_since(t, origin, name)
    ismissing(t) && return missing
    t isa Union{Date,DateTime} || throw(ArgumentError(
        "with `origin`, $name times must be dates (Date or DateTime) or missing; got $(typeof(t))"))
    days = Dates.value(DateTime(t) - DateTime(origin)) / 86_400_000     # milliseconds per day
    days ≥ 0 || throw(ArgumentError("$name date $t is before origin = $origin"))
    return days
end

function _linelist(infection_times, removal_times, Δt, nsteps, censor_at)
    n = length(infection_times)
    length(removal_times) == n || throw(DimensionMismatch(
        "infection_times and removal_times must have the same length: $n ≠ $(length(removal_times))"))
    n > 0 || throw(ArgumentError("the line list is empty"))
    Δt > 0 || throw(ArgumentError("step length Δt must be > 0; got $Δt"))
    for (name, v) in (("infection", infection_times), ("removal", removal_times))
        for t in v
            ismissing(t) && continue
            t isa Union{Date,DateTime} && throw(ArgumentError(
                "$name times are dates: give `origin`, the date at which step 1 starts"))
            t isa Real || throw(ArgumentError("$name times must be real numbers or missing; got $(typeof(t))"))
            (isfinite(t) && t ≥ 0) || throw(ArgumentError("$name times must be finite and ≥ 0; got $t"))
        end
    end
    step_of(t) = floor(Int, t / Δt) + 1
    known_inf = [t for t in infection_times if !ismissing(t)]
    ns = nsteps === nothing ? (isempty(known_inf) ? 1 : maximum(step_of, known_inf)) : Int(nsteps)
    ns ≥ 1 || throw(ArgumentError("nsteps must be ≥ 1; got $ns"))

    counts = zeros(Int, ns)
    intervals = Int[]; cens = Bool[]; removal_only = Int[]
    for (ti, tr) in zip(infection_times, removal_times)
        if ismissing(ti)
            ismissing(tr) && throw(ArgumentError("a case has both its infection and removal times missing"))
            push!(removal_only, clamp(step_of(tr), 1, ns))
            continue
        end
        k = step_of(ti)
        k ≤ ns || throw(ArgumentError("infection time $ti is beyond the grid of $ns steps of length $Δt"))
        censor_at === nothing || ti < censor_at ||
            throw(ArgumentError("infection time $ti is at or after censor_at = $censor_at"))
        counts[k] += 1
        ismissing(tr) && continue
        tr ≥ ti || throw(ArgumentError("removal time $tr precedes infection time $ti"))
        if censor_at !== nothing && tr ≥ censor_at
            push!(intervals, max(1, round(Int, (censor_at - ti) / Δt))); push!(cens, true)
        else
            push!(intervals, max(1, round(Int, (tr - ti) / Δt))); push!(cens, false)
        end
    end
    return LineList(ns, Float64(Δt), counts, n, intervals, cens, removal_only)
end
