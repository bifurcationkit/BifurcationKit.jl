"""
$(TYPEDEF)

Abstract type for the predictions returned by [`predictor`](@ref).

A prediction is an [`EquilibriumPredictor`](@ref), a
[`PeriodicOrbitPredictor`](@ref) or an [`EmptyPredictor`](@ref) (no bifurcated
branch, typically at a Fold point).
"""
abstract type AbstractNormalFormPredictor end

"""
$(TYPEDEF)

Prediction for a branch of equilibria bifurcating from a simple branch point.

$(TYPEDFIELDS)
"""
struct EquilibriumPredictor{Tx0, Tx1, Tp, Ta, Te} <: AbstractNormalFormPredictor
    "point on the trivial branch."
    x0::Tx0
    "guess on the bifurcated branch."
    x1::Tx1
    "predicted parameter value."
    pnew::Tp
    "amplitude of the predicted point."
    amp::Ta
    "method-specific fields (`dsfactor`, `δp`, `xm1`, `pm1`, `p0`, ...), or `nothing` if there are none. They are accessible via `pred.extras`, e.g. `pred.extras.dsfactor`."
    extras::Te
end

"""
$(TYPEDEF)

Prediction for a branch of periodic orbits bifurcating from a bifurcation point
of periodic orbits.

$(TYPEDFIELDS)
"""
struct PeriodicOrbitPredictor{Torb, Tp, Tprob, Ta, Tpo, Te} <: AbstractNormalFormPredictor
    "guess for the periodic orbit."
    orbitguess::Torb
    "predicted parameter value."
    pnew::Tp
    "periodic orbit problem used for the continuation."
    prob::Tprob
    "amplitude multiplier used to build the guess."
    ampfactor::Ta
    "reference periodic orbit."
    po::Tpo
    "method-specific fields (`ζc`, `δp`, ...), or `nothing` if there are none. They are accessible via `pred.extras`, e.g. `pred.extras.ζc`."
    extras::Te
end

"""
$(TYPEDEF)

Returned by [`predictor`](@ref) when no bifurcated branch can be predicted, for
example at a Fold point of equilibria or periodic orbits.

Use `pred isa EmptyPredictor` to test for this case.
"""
struct EmptyPredictor <: AbstractNormalFormPredictor end

"""
$(TYPEDSIGNATURES)

Return the predicted parameter value of a normal form predictor.

Throws an `ArgumentError` for an [`EmptyPredictor`](@ref).
"""
get_predicted_param(pr::Union{EquilibriumPredictor, PeriodicOrbitPredictor}) = pr.pnew
get_predicted_param(::EmptyPredictor) = throw(ArgumentError("EmptyPredictor does not contain a predicted parameter."))


