abstract type AbstractContinuationKind end
abstract type OneParamCont <: AbstractContinuationKind end
abstract type TwoParamCont <: AbstractContinuationKind end

# Codim1
struct EquilibriumCont <: OneParamCont end
struct PeriodicOrbitCont <: OneParamCont end
struct TravellingWaveCont <: OneParamCont end

# Codim2
struct FoldCont <: TwoParamCont end
struct HopfCont <: TwoParamCont end
struct PDCont <: TwoParamCont end

# Codim2 periodic orbit
struct FoldPeriodicOrbitCont <: TwoParamCont end
struct PDPeriodicOrbitCont <: TwoParamCont end
struct NSPeriodicOrbitCont <: TwoParamCont end
