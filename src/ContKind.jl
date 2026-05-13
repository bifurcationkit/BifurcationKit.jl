abstract type AbstractContinuationKind end
abstract type OneParamCont <: AbstractContinuationKind end
abstract type TwoParamCont <: AbstractContinuationKind end
abstract type TwoParamPeriodicOrbitCont <: TwoParamCont end

struct EquilibriumCont <: OneParamCont end # TODO rename abstract
struct PeriodicOrbitCont <: OneParamCont end
struct TravellingWaveCont <: OneParamCont end

struct FoldCont <: TwoParamCont end
struct HopfCont <: TwoParamCont end
struct PDCont <: TwoParamCont end
struct NSCont <: TwoParamCont end

struct FoldPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end
struct PDPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end
struct NSPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end

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