BifurcationKit.jl, Changelog
========================

All notable changes to this project will be documented in this file.

## [0.1.0] - 2020-7-9
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

