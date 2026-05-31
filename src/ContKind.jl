abstract type AbstractContinuationKind end
abstract type AbstractOneParamCont <: AbstractContinuationKind end
abstract type AbstractTwoParamCont <: AbstractContinuationKind end
abstract type AbstractTwoParamPeriodicOrbitCont <: AbstractTwoParamCont end

struct EquilibriumCont <: AbstractOneParamCont end
struct PeriodicOrbitCont <: AbstractOneParamCont end
struct BoundaryValueProblemCont <: AbstractOneParamCont end
struct TravellingWaveCont <: AbstractOneParamCont end

struct FoldCont <: AbstractTwoParamCont end
struct HopfCont <: AbstractTwoParamCont end
struct PDCont <: AbstractTwoParamCont end
struct NSCont <: AbstractTwoParamCont end

struct FoldPeriodicOrbitCont <: AbstractTwoParamPeriodicOrbitCont end
struct PDPeriodicOrbitCont <: AbstractTwoParamPeriodicOrbitCont end
struct NSPeriodicOrbitCont <: AbstractTwoParamPeriodicOrbitCont end

# these structs allow to wrap the discretization (shooting, collocation, etc) to the problem: 
# computing periodic orbits, solving BVP, etc
# we rely on wrapper types but see the issue:
# https://github.com/JuliaLang/julia/issues/37790
struct PeriodicOrbit{Tdisc}
    disc::Tdisc
end

struct TravellingWave{Tdisc}
    disc::Tdisc
end
@inline get_discretization(model::Union{PeriodicOrbit, TravellingWave}) = model.disc