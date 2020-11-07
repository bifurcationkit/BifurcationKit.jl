BifurcationKit.jl, Changelog
========================

All notable changes to this project will be documented in this file.

## [0.0.1] - ??
- remove fields `n_unstable`, `n_imag` and `stability` from `ContResult` and put it in the field `branch`.

## [0.0.1] - 2020-10-25
- the keyword argument `Jt` for the jacobian transpose is written `Jᵗ`

## [0.0.1] - 2020-9-18
- new way to use the argument `printSolution` in `continuation`. You can return (Named) tuple now.

## [0.0.1] - 2020-9-17
- add new type GenericBifPoint to record bifurcation points and also an interval which contains the bifurcation point
- add `kwargs` to arguments `finaliseSolution`
- add `kwargs` to callback from `newton`. In particular, newton passes `fromNewton=true`, newtonPALC passes `fromNewton = false`
- save intervals for the location of bifurcation points in the correct way, (min, max)

## [0.0.1] - 2020-9-16
- better estimation of d2f/dpdx in normal form computation
- change of name `HyperplaneSections` -> `SectionPS` for Poincare Shooting

## [0.0.1] - 2020-9-12
- clamp values in [pMin, pMax] for continuation
- put arrow at the end of the branch (plotting)

## [0.0.1] - 2020-9-6
- add eta parameter in ContinuationPar 
- change name `PALCStateVariables` into `ContState` and `PALCIterable` into `ContIterable`
- add Deflated Continuation

## [0.0.1] - 2020-8-21
- add Multiple predictor (this is needed to implement the `pmcont` algorithm from `pde2path` (Matlab)

## [0.0.1] - 2020-7-26
- add Polynomial predictor

## [0.0.1] - 2020-7-19
- add Branch switching for non-simple branch points

## [0.0.1] - 2020-7-9
The package is registered.

## [0.0.1] - 2020-6-20

### Deprecated

- Rename option `ContinuationPar`: `saveSolEveryNsteps` --> `saveSolEveryStep`
- Rename option `ContinuationPar`: `saveEigEveryNsteps` --> `saveEigEveryStep`
- Rename option `ContinuationPar`: `plotEveryNsteps` --> `plotEveryStep` 

## [0.0.1] - 2020-6-10

- change the name of the package into `BifurcationKit.jl`

### Deprecated

- The options `computeEigenvalue` in `ContinuationPar` has been removed. It is now controlled with `detectBifurcation`.

## [0.0.1] - 2020-5-2


### Added

- automatic branch switching from simple Hopf points 
- automatic normal form computation for any kernel dimension


## [0.0.1] - 2020-4-27


### Added

- automatic branch switching from simple branch points (equilibrium)
- automatic normal form computation 

