# Small numerical helpers shared by the forward maps and the likelihood (internal).

"""
    normalise_pmf(f, n)

Non-negative, finite copy of `f` normalised to sum to one; `NaN`, infinite and negative entries
are set to zero first. If the remaining total is not above `1e-15` (or not finite), the uniform
distribution on `n` points is returned.
"""
function normalise_pmf(f::AbstractVector{T}, n::Integer) where {T}
    f_pos = ifelse.(isnan.(f) .| .!isfinite.(f), zero(T), max.(f, zero(T)))
    sF = sum(f_pos)
    return (sF > T(1e-15) && isfinite(sF)) ? f_pos ./ sF : fill(one(T) / n, n)
end

"""Stable `log(exp(a) + exp(b))` (`-Inf` if both are `-Inf`)."""
function logaddexp(a::Real, b::Real)
    m = max(a, b)
    return isfinite(m) ? m + log(exp(a - m) + exp(b - m)) : m
end

"""A valid probability `p ∈ (0, 1)` (finite primal value)."""
_is_probability(p) = 0 < p < 1

"""Logger that passes on only messages at or above `level` (used to silence Turing's info messages)."""
struct _MinLevelLogger{L<:AbstractLogger} <: AbstractLogger
    logger::L
    level::LogLevel
end
Logging.min_enabled_level(l::_MinLevelLogger) = max(l.level, Logging.min_enabled_level(l.logger))
Logging.shouldlog(l::_MinLevelLogger, level, args...) = level ≥ l.level && Logging.shouldlog(l.logger, level, args...)
Logging.handle_message(l::_MinLevelLogger, args...; kwargs...) = Logging.handle_message(l.logger, args...; kwargs...)
Logging.catch_exceptions(l::_MinLevelLogger) = Logging.catch_exceptions(l.logger)
