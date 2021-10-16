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
PeriodicOrbitOCollProblem
```

```@docs
ShootingProblem
```

```@docs
PoincareShootingProblem
```

## Newton

```@docs
newton
```

## [Continuation](@id Library-Continuation)

```@docs
continuation
```

## Events

```@docs
BifurcationKit.DiscreteEvent
```

```@docs
BifurcationKit.ContinuousEvent
```

```@docs
BifurcationKit.SetOfEvents
```

```@docs
BifurcationKit.PairOfEvents
```

## Branch switching (branch point)

```@docs
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jᵗ = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = getvectortype(br), scaleζ = norm, verbosedeflation = false, maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter), perturb = identity, kwargs...)
```

## Branch switching (Hopf point)
```@docs
continuation(F, dF, d2F, d3F, br::BifurcationKit.AbstractBranchResult, ind_bif::Int, _contParams::ContinuationPar, prob::BifurcationKit.AbstractPeriodicOrbitProblem ; Jᵗ = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, usedeflation = false, nev = _contParams.nev, updateSectionEveryStep = 0, kwargs...)
```

## Bifurcation diagram

```@docs
bifurcationdiagram
```

```@docs
bifurcationdiagram!
```

```@docs
getBranch
```

```@docs
getBranchesFromBP
```

```@docs
BifurcationKit.SpecialPoint
```

## Utils for periodic orbits

```@docs
getPeriod
```

```@docs
getAmplitude
```

```@docs
getMaximum
```

```@docs
SectionSS
```

```@docs
SectionPS
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
FloquetQaD
```

```@docs
guessFromHopf(br, ind_hopf, eigsolver::AbstractEigenSolver, M, amplitude; phase = 0)
```

```@docs
computeNormalForm
```