# Library

## Parameters

```@docs
NewtonPar
```

```@docs
ContinuationPar
```

## Results

```@docs
ContResult
```

## Problems

```@docs
DeflationOperator
```

```@docs
DeflatedProblem
```

```@docs
PeriodicOrbitTrapProblem
```

```@docs
ShootingProblem
```

```@docs
PoincareShootingProblem
```

## Misc.

```@docs
PrecPartialSchurKrylovKit
```

```@docs
PrecPartialSchurArnoldiMethod
```

```@docs
Flow
```

```@docs
FloquetQaDTrap
```

```@docs
FloquetQaDShooting
```

## Newton

```@docs
newton
```

## Newton with deflation

```@docs
newton(Fhandle, Jhandle, x0::vectype, options:: NewtonPar{T}, defOp::DeflationOperator{T, Tf, vectype}; kwargs...) where {T, Tf, vectype}
```

### Newton for Fold / Hopf

```@docs
newtonFold(F, J, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm) where {T, vectype}
```

```@docs
newtonFold(F, J, br::ContResult, ind_fold::Int64, options::NewtonPar;Jt = nothing, d2F = nothing, kwargs...)
```

```@docs
newtonHopf(F, J, hopfpointguess::BorderedArray{vectypeR, T}, eigenvec, eigenvec_ad, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm) where {vectypeR, T}
```

```@docs
newtonHopf(F, J, br::ContResult, ind_hopf::Int64, options::NewtonPar ; Jt = nothing, d2F = nothing, normN = norm)
```

### Newton for Periodic Orbits

```@docs
newton(probPO::PeriodicOrbitTrapProblem, orbitguess, options::NewtonPar, linearalgo::Symbol = :BorderedLU; kwargs...)
```

```@docs
newton(prob::T, orbitguess, options::NewtonPar; kwargs...) where {T <: AbstractShootingProblem}
```

## Continuation

```@docs
continuation
```

### Continuation for Fold / Hopf

```@docs
continuationFold(F, J, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...) where {T,vectype}
```

```@docs
continuationFold(F, J, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...)
```

```@docs
continuationHopf(F, J, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ;  Jt = nothing, d2F = p2 -> nothing, kwargs...)
```

```@docs
continuationHopf(F, J, hopfpointguess::BorderedArray{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...) where {T,Tb,vectype}
```

### Continuation for periodic orbits

```@docs
continuationPOTrap(probPO, orbitguess, p0::Real, contParams::ContinuationPar; linearalgo = :BorderedLU, printSolution = (u,p) -> u[end], kwargs...)
```

```@docs
continuationPOShooting(prob, orbitguess, p0::Real, contParams::ContinuationPar; printPeriod = true, kwargs...)
```