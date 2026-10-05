"""
$(TYPEDEF)

Step control of [`PALC`](@ref) by the quality of its corrector, after Allgower and Georg (*Numerical Continuation Methods*, 1990) and Gambit's logit tracer (`PathTracer::TracePath`, `src/solvers/path/path.cc`). It is off by default and passed as `PALC(step_control = CorrectorQuality(...))`; `step_control = nothing` returns to the control of `ds` by the number of Newton iterations (see the parameter `a` of [`ContinuationPar`](@ref)). The quality of the corrector is measured by the lengths ``d_1, d_2, \\dots`` of its Newton steps, in the norm of the arc length constraint.

# Tests

The step is rejected, as if the corrector had not converged (`ds` is halved and the step is retried), when
- the first Newton step has length ``d_1 \\geq`` `max_distance`. The corrector is stopped at once. This is the distance from the predictor to the curve, on the scale of `ds` and in the units of the state, which is why the test is off (`max_distance = nothing`) unless a threshold is given (the constructor default),
- a later Newton step contracts too slowly: ``d_k / (d_{k-1} + \\text{tol}\\,\\eta) >`` `max_contraction`. The corrector is stopped at once. `tol` is `newton_options.tol`. The rate has no units, so its default `0.6` needs no tuning.

A test is switched off with `nothing`, in which case it neither rejects a step nor slows `ds`.
With `max_growth = nothing` the tests only reject steps and `ds` follows the Newton iteration count after every accepted step, as without step control.

# Step size

After an accepted step (and unless `max_growth = nothing`), `ds` is multiplied by `1 / deceleration` with

`deceleration = max(1 / max_growth, max_growth * sqrt(d₁ / max_distance), max_growth * sqrt(c_k / max_contraction) for k ≥ 2)`

where `c_k = d_k / (d_{k-1} + tol η)`. So `ds` grows by at most the factor `max_growth` per step and shrinks as the distance or the contraction approach their limits. A step after which the predictor was already a solution (no Newton step) grows by `max_growth`. The result is clamped to `[dsmin, dsmax]`.

!!! note "Where it applies"
    The tests and the step size rule apply to the steps of the `PALC` corrector. A step that falls back on the `Natural` corrector (at the bounds `p_min`, `p_max`) is not tested. Neither is a step of the bisection that locates special points, whose `ds` is prescribed.

# Fields

$(TYPEDFIELDS)

# Constructor

    CorrectorQuality(; max_distance = nothing, max_contraction = 0.6, max_growth = 1.1, η = 0.1)

`max_growth = nothing` leaves the step size to the Newton iteration count. Gambit's values are `max_distance = 0.4` (in log probabilities), `max_contraction = 0.6`, `max_growth = 1.1` and `η = 0.1`.
"""
struct CorrectorQuality{T <: Real}
    "Largest allowed length of the first Newton step, or `nothing` for no test."
    max_distance::Union{Nothing, T}
    "Largest allowed contraction rate of the Newton steps after the first, or `nothing` for no test."
    max_contraction::Union{Nothing, T}
    "Largest factor by which `ds` grows in one step, also the factor in the deceleration, or `nothing` to leave the step size to the Newton iteration count. Must exceed 1."
    max_growth::Union{Nothing, T}
    "Offset `tol * η` in the contraction rate, which avoids dividing by a vanishing Newton step."
    η::T

    function CorrectorQuality{T}(max_distance::Union{Nothing, Real}, max_contraction::Union{Nothing, Real}, max_growth::Union{Nothing, Real}, η::Real) where {T <: Real}
        @assert _is_positive(max_distance) "max_distance must be positive or nothing"
        @assert _is_positive(max_contraction) "max_contraction must be positive or nothing"
        @assert isnothing(max_growth) || max_growth > 1 "max_growth must exceed 1 or be nothing"
        @assert η >= 0 "η must be non-negative"
        return new{T}(_as(T, max_distance), _as(T, max_contraction), _as(T, max_growth), convert(T, η))
    end
end

_is_positive(::Nothing) = true
_is_positive(x::Real) = x > 0
_as(::Type{T}, ::Nothing) where {T <: Real} = nothing
_as(::Type{T}, x::Real) where {T <: Real} = convert(T, x)

function CorrectorQuality(; max_distance::Union{Nothing, Real} = nothing, max_contraction::Union{Nothing, Real} = 0.6, max_growth::Union{Nothing, Real} = 1.1, η::Real = 0.1)
    given = filter(!isnothing, (max_distance, max_contraction, max_growth, η))
    T = float(promote_type(map(typeof, given)...))
    return CorrectorQuality{T}(max_distance, max_contraction, max_growth, η)
end

"""`control` with its thresholds in the float type `T`, so that the step size stays in the state's type."""
function CorrectorQuality{T}(control::CorrectorQuality)::CorrectorQuality{T} where {T <: Real}
    return CorrectorQuality{T}(control.max_distance, control.max_contraction, control.max_growth, control.η)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
What a corrector has shown of its quality so far, for [`CorrectorQuality`](@ref): the length of its last Newton step, the largest deceleration asked for by the steps up to it, and their number. Immutable, [`observe`](@ref) returns the next one.
"""
struct StepQuality{T <: Real}
    previous::T
    deceleration::T
    steps::Int
end

"""The quality of a corrector that has not taken a Newton step: it asks for the largest growth, `1 / deceleration = max_growth`."""
StepQuality(control::CorrectorQuality{T}) where {T <: Real} = StepQuality{T}(zero(T), _initial_deceleration(control.max_growth), 0)

_initial_deceleration(max_growth::Real) = inv(max_growth)
_initial_deceleration(::Nothing) = 0
_deceleration(max_growth::Real, ratio::Real) = max_growth * sqrt(ratio)
_deceleration(::Nothing, ratio::Real) = 0

"""
$(TYPEDSIGNATURES)

Contraction rate of a Newton step of length `distance` that follows one of length `quality.previous`. It is only defined after the first step.
"""
function contraction_rate(control::CorrectorQuality{T}, quality::StepQuality{T}, distance::T, tol::Real)::T where {T <: Real}
    return distance / (quality.previous + convert(T, tol) * control.η)
end

# a test that is off rejects nothing and asks for no deceleration
_reaches(::Nothing, value::Real) = false
_reaches(limit::Real, value::Real) = value >= limit
_exceeds(::Nothing, value::Real) = false
_exceeds(limit::Real, value::Real) = value > limit
_load(::Nothing, value::T) where {T <: Real} = zero(T)
_load(limit::Real, value::Real) = value / limit

"""
$(TYPEDSIGNATURES)

Whether the step must be rejected, and the corrector stopped, because of the Newton step of length `distance` that follows the ones in `quality`: the first must be shorter than `max_distance`, the following ones must contract by less than `max_contraction`.
"""
function rejects_step(control::CorrectorQuality{T}, quality::StepQuality{T}, distance::T, tol::Real)::Bool where {T <: Real}
    if quality.steps == 0
        return _reaches(control.max_distance, distance)
    end
    return _exceeds(control.max_contraction, contraction_rate(control, quality, distance, tol))
end

"""
$(TYPEDSIGNATURES)

The quality after the Newton step of length `distance`, which `rejects_step` accepted.
"""
function observe(control::CorrectorQuality{T}, quality::StepQuality{T}, distance::T, tol::Real)::StepQuality{T} where {T <: Real}
    if quality.steps == 0
        ratio = _load(control.max_distance, distance)
    else
        ratio = _load(control.max_contraction, contraction_rate(control, quality, distance, tol))
    end
    deceleration = max(quality.deceleration, _deceleration(control.max_growth, ratio))
    return StepQuality{T}(distance, deceleration, quality.steps + 1)
end

"""
$(TYPEDSIGNATURES)

Factor by which an accepted step multiplies `ds`, or `nothing` when `control` leaves the step size to the Newton iteration count.
"""
function growth_factor(control::CorrectorQuality, quality::StepQuality{T})::Union{Nothing, T} where {T <: Real}
    if isnothing(control.max_growth)
        return nothing
    end
    return inv(quality.deceleration)
end

"""
$(TYPEDSIGNATURES)

Whether the growth of `ds` after the step is limited by `max_growth`: no test asked for a deceleration beyond the lowest one, `1 / max_growth`.
"""
growth_capped(control::CorrectorQuality, quality::StepQuality) = quality.deceleration <= _initial_deceleration(control.max_growth)
