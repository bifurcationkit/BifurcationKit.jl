"""
$(TYPEDEF)

Step acceptance tests and step size rule of Allgower and Georg (*Numerical Continuation Methods*, 1990) for the corrector of [`PALC`](@ref). It is off by default: pass it as `PALC(step_acceptance = AllgowerGeorg())`. The default values are the ones of Gambit's logit tracer (`PathTracer::TracePath`, `src/solvers/path/path.cc`). With it, `ds` is controlled by the *quality* of the corrector, measured by the lengths ``d_1, d_2, \\dots`` of its Newton steps, instead of by their number (see the parameter `a` of [`ContinuationPar`](@ref)).

# Tests

The step is rejected, as if the corrector had not converged (`ds` is halved and the step is retried), when
- the first Newton step has length ``d_1 \\geq`` `max_distance`. The corrector is stopped at once. This is the distance from the predictor to the curve, in the norm of the arc length constraint, so it is on the scale of `ds`,
- a later Newton step contracts too slowly: ``d_k / (d_{k-1} + \\text{tol}\\,\\eta) >`` `max_contraction`. The corrector is stopped at once. `tol` is `newton_options.tol`,
- the converged point lies behind or beside the step: the cosine between the tangent at the previous point and the chord to the new point, in the norm of the arc length constraint, is below `min_alignment`. With `min_alignment = 0` the continuation never turns back on itself (the orientation test of Allgower-Georg). A larger value also rejects a sharp turn, for example `0.8` allows a turn of about 37 degrees between tangent and chord. The chord stands for the new tangent, which is not available before the next predictor; for a [`Secant`](@ref) predictor it is the new tangent.

# Step size

After an accepted step, `ds` is multiplied by `1 / deceleration` with

`deceleration = max(1 / max_growth, max_growth * sqrt(d₁ / max_distance), max_growth * sqrt(c_k / max_contraction) for k ≥ 2)`

where `c_k = d_k / (d_{k-1} + tol η)`. So `ds` grows by at most the factor `max_growth` per step and shrinks as the distance or the contraction approach their limits. A step after which the predictor was already a solution (no Newton step) grows by `max_growth`. The result is clamped to `[dsmin, dsmax]`.

Each test is switched off alone with `max_distance = Inf`, `max_contraction = Inf` or `min_alignment = -1`.

!!! note "Where it applies"
    The tests and the step size rule apply to the steps of the `PALC` corrector. A step that falls back on the `Natural` corrector (at the bounds `p_min`, `p_max`) is not tested.

# Fields

$(TYPEDFIELDS)

# Constructor

    AllgowerGeorg(; max_distance = 0.4, max_contraction = 0.6, max_growth = 1.1, min_alignment = 0.0, η = 0.1)
"""
struct AllgowerGeorg{T <: Real}
    "Largest allowed length of the first Newton step (`Inf` switches the test off)."
    max_distance::T
    "Largest allowed contraction rate of the Newton steps after the first (`Inf` switches the test off)."
    max_contraction::T
    "Largest factor by which `ds` grows in one step, also the factor in the deceleration. Must exceed 1."
    max_growth::T
    "Smallest allowed cosine between the tangent and the chord of the step, in [-1, 1] (`-1` switches the test off)."
    min_alignment::T
    "Offset `tol * η` in the contraction rate, which avoids dividing by a vanishing Newton step."
    η::T

    function AllgowerGeorg{T}(max_distance::Real, max_contraction::Real, max_growth::Real, min_alignment::Real, η::Real) where {T <: Real}
        @assert max_distance > 0 "max_distance must be positive"
        @assert max_contraction > 0 "max_contraction must be positive"
        @assert max_growth > 1 "max_growth must exceed 1"
        @assert -1 <= min_alignment <= 1 "min_alignment must belong to [-1, 1]"
        @assert η >= 0 "η must be non-negative"
        return new{T}(convert(T, max_distance), convert(T, max_contraction), convert(T, max_growth), convert(T, min_alignment), convert(T, η))
    end
end

function AllgowerGeorg(; max_distance::Real = 0.4, max_contraction::Real = 0.6, max_growth::Real = 1.1, min_alignment::Real = 0.0, η::Real = 0.1)
    T = float(promote_type(typeof(max_distance), typeof(max_contraction), typeof(max_growth), typeof(min_alignment), typeof(η)))
    return AllgowerGeorg{T}(max_distance, max_contraction, max_growth, min_alignment, η)
end

"""`acceptance` with its thresholds in the float type `T`, so that the step size stays in the state's type."""
function AllgowerGeorg{T}(acceptance::AllgowerGeorg)::AllgowerGeorg{T} where {T <: Real}
    return AllgowerGeorg{T}(acceptance.max_distance, acceptance.max_contraction, acceptance.max_growth, acceptance.min_alignment, acceptance.η)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
What a corrector has shown of its quality so far, for [`AllgowerGeorg`](@ref): the length of its last Newton step, the largest deceleration asked for by the steps up to it, and their number. Immutable, [`observe`](@ref) returns the next one.
"""
struct StepQuality{T <: Real}
    previous::T
    deceleration::T
    steps::Int
end

"""The quality of a corrector that has not taken a Newton step: it asks for the largest growth, `1 / deceleration = max_growth`."""
StepQuality(acceptance::AllgowerGeorg{T}) where {T <: Real} = StepQuality{T}(zero(T), inv(acceptance.max_growth), 0)

"""
$(TYPEDSIGNATURES)

Contraction rate of a Newton step of length `distance` that follows one of length `quality.previous`. It is only defined after the first step.
"""
function contraction_rate(acceptance::AllgowerGeorg{T}, quality::StepQuality{T}, distance::T, tol::Real)::T where {T <: Real}
    return distance / (quality.previous + convert(T, tol) * acceptance.η)
end

"""
$(TYPEDSIGNATURES)

Whether the step must be rejected, and the corrector stopped, because of the Newton step of length `distance` that follows the ones in `quality`: the first must be shorter than `max_distance`, the following ones must contract by less than `max_contraction`.
"""
function rejects_step(acceptance::AllgowerGeorg{T}, quality::StepQuality{T}, distance::T, tol::Real)::Bool where {T <: Real}
    if quality.steps == 0
        return distance >= acceptance.max_distance
    end
    return contraction_rate(acceptance, quality, distance, tol) > acceptance.max_contraction
end

"""
$(TYPEDSIGNATURES)

The quality after the Newton step of length `distance`, which `rejects_step` accepted.
"""
function observe(acceptance::AllgowerGeorg{T}, quality::StepQuality{T}, distance::T, tol::Real)::StepQuality{T} where {T <: Real}
    if quality.steps == 0
        ratio = distance / acceptance.max_distance
    else
        ratio = contraction_rate(acceptance, quality, distance, tol) / acceptance.max_contraction
    end
    deceleration = max(quality.deceleration, acceptance.max_growth * sqrt(ratio))
    return StepQuality{T}(distance, deceleration, quality.steps + 1)
end

"""
$(TYPEDSIGNATURES)

Factor by which an accepted step multiplies `ds`.
"""
growth_factor(quality::StepQuality)::typeof(quality.deceleration) = inv(quality.deceleration)
