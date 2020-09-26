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

```@docs
guessFromHopf(br, ind_hopf, eigsolver::AbstractEigenSolver, M, amplitude; phase = 0)
```

```@docs
computeNormalForm
```

## Newton

```@docs
newton
```

```@docs
newton(probPO::PeriodicOrbitTrapProblem, orbitguess, par, options::NewtonPar; linearPO::Symbol = :BorderedLU, kwargs...)
```

## Continuation

```@docs
continuation
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
GenericBifPoint
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