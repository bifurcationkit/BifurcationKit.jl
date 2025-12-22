abstract type AbstractContinuationKind end
abstract type OneParamCont <: AbstractContinuationKind end
abstract type TwoParamCont <: AbstractContinuationKind end
abstract type TwoParamPeriodicOrbitCont <: TwoParamCont end

struct EquilibriumCont <: OneParamCont end
struct PeriodicOrbitCont <: OneParamCont end
struct TravellingWaveCont <: OneParamCont end

struct FoldCont <: TwoParamCont end
struct HopfCont <: TwoParamCont end
struct PDCont <: TwoParamCont end
struct NSCont <: TwoParamCont end

struct FoldPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end
struct PDPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end
struct NSPeriodicOrbitCont <: TwoParamPeriodicOrbitCont end
