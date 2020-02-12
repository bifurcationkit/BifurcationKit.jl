# Library

## Structs

```@docs
NewtonPar
```

```@docs
ContinuationPar
```

```@docs
ContResult
```

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
Flow
```

```@docs
ShootingProblem
```

```@docs
PoincareShootingProblem
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

## Newton for Fold / Hopf

```@docs
newtonFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; normN = norm, d2F_is_known = true ) where {T,vectype}
```

```@docs
newtonFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
```

## Newton for Periodic Orbits

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

## Continuation for Fold / Hopf

```@docs
continuationFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ; d2F_is_known = true, kwargs...) where {T,vectype}
```

```@docs
continuationFold(F, J, Jt, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ;kwargs...) where {T,vectype}
```

```@docs
continuationFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
```

```@docs
continuationHopf(F, J, Jt, d2F, hopfpointguess::BorderedArray{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...) where {T,Tb,vectype}
```

## Continuation for periodic orbits

```@docs
continuationPOTrap(probPO, orbitguess, p0::Real, contParams::ContinuationPar, linearalgo = :BorderedLU; printSolution = (u,p) -> u[end], kwargs...)
```

```@docs
continuationPOShooting(prob, orbitguess, p0::Real, contParams::ContinuationPar; printPeriod = true, kwargs...)
```

## Plotting

```@docs
plotBranch(contres::ContResult; kwargs...)
```

```@docs
plotBranch!(contres::ContResult; kwargs...)
```

```@docs
plotBranch(brs::Vector; kwargs...)
```

