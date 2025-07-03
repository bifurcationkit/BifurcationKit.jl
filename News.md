BifurcationKit.jl, Changelog
========================

All notable changes to this project will be documented in this file (hopefully). No performance improvements will be notified but mainly the addition of new methods, the modifications of internal structs, etc.

## [0.5.0]
- add `ShiftInvert`, new general composite eigen solver
- add option `use_adapted_mesh` to `generate_ci_problem(pb::PeriodicOrbitOCollProblem`
- add option `optimal_period` to `generate_ci_problem(shooting::ShootingProblem`
- 🚦🚦🚦 change jacobian parameter in PeriodicOrbitTrapProblem from Symbol to custom type. See `?PeriodicOrbitTrapProblem` for more information
- correct bug in `Natural` which improves `AutoSwitch`
- remove (some) closures from codim1 continuation
- add structures/normal form for CuspPO, R1, R2, etc

## [0.4.16]
- make `WrapPOColl` a `AbstractWrapperFDProblem`
- 🚦🚦🚦 change `jacobian_ma` argument from Symbol to type. See docs for codim 2 continuation. For example, `continuation(br ,1; jacobian_ma = :minaug)` becomes  `continuation(br ,1; jacobian_ma = MinAug())`
- 🚦🚦🚦 add `update!` function to `BifurcationProblem`. This allows to adapt the problem during continuation

## [0.4.15]
- solve compile time issue for continuation of Fold of periodic orbits
- move test functions out of continuation function for `MinAug`
- emove the saving of AD generated BifFunction. Rely on dispatch instead.

## [0.4.4]
- change type parameters in AbstractCodim2EigenSolver
- add Krylov.jl as possible linear solver

## [0.4.3]
- add jacobian option MinAugMatrixBased to Fold/Hopf continuation
- remove reference to RecursiveVec

## [0.4.2]
- change bordered linear solvers' interface
- `record_from_solution` has been changed to the following definition

## [0.4.0]
- Setfield.jl is not anymore the main component of BifurcationKit to support parameter axis. It has been changed in favour of Accessors.jl

## [0.3.7]
- remove `Requires.jl` and use extensions. This requires julia>=1.9

## [0.3.5]
- add field `save_solution` to `BifurcationProblem`. Allows to save problem state along with the current solution. Useful for periodic orbits for example where we can save the phase condition and the mesh when the latter is adapted.

## [0.3.4]
- add function `_keep_opts_cont` to filter continuation options
- order 2 prediction for periodic orbit from Hopf bifurcation
- add `jacobian(Π::PoincaréMap{ <: WrapPOSh }, x, pars)`
- `hasstability` becomes `_hasstability`
- `getvectortype` becomes `_getvectortype`
- add condensation of parameters, great speedup for collocation of periodic orbits
- add `in_bisection` field in struct `ContState`
- allow to do codim 2 continuation of PO with collocation and mesh adaptation
- `update_section_every_step` becomes a UInt
- add fields in `PeriodDoublingProblemMinimallyAugmented` and `NeimarkSackerProblemMinimallyAugmented` for holding Resonance test values
- add specific finalizer for Fold of PO when using Collocation or shooting
- add struct `FinalisePO`
- `update_minaug_every_step = 1` by default for Hopf / Fold continuation
- `update_minaug_every_step = 1` is default for for PD / NS continuation

## [0.2.8] - 2023-05-18
- add `getDelta` to the interface of `AbstractFlow`
- remove `finDiffEps` from `ContinuationPar`. 

## [0.2.8] - 2023-04-23
- use jvp function name in `Flow` interface
- add radius to section of Poincare Shooting
- add _mesh field to reconstruct POColl problem (adapted mesh) from previous solution
- add new jacobian parametrization using `struct`s instead of Symbol 
- remove `θ` from ContinuationPar

## [0.2.8] - 2023-04-17
- add delta keyword to BifurcationProblem constructor

MISSSINF 

## [0.2.0] - 2022-07-23
- new interface based on the problem `BifurcationProblem`

## [0.1.8] - 2021-12-12
- switch from `DiffEqBase` to `SciMLBase`
- change function name `closesttozero` to `rightmost`

## [0.1.8] - 2021-11-27
- ⛳️ add a new interface for Flows
- add custom distance for `DeflationOperator`
- add possibility to use forward diff (AD) with deflation operator

## [0.1.8] - 2021-11-20
- the method for periodic orbits `getM` becomes `getMeshSize`

## [0.1.7] - 2021-11-6
- add abstract types `AbstractDirectLinearSolver` and `AbstractIterativeLinearSolver`
- the function `getTrajectory` becomes `getPeriodicOrbit`
- add struct `SolPeriodicOrbit` to allow for unified plotting interface with all methods for computing periodic orbits
- ⛳️ the keyword argument 	`linearPO` is renamed into `jacobianPO`
- add newton / continuation methods for `TWProblem`
- add `GEigArpack` generalized eigensolver

## [0.1.5] - 2021-10-23
- remove documentation from package, it is now located in BifurcationKitDocs.jl

## [0.1.5] - 2021-10-16
- change function name problemForBS into reMake for aBS of periodic orbits
- add function `generateSolution` to generate guess for computing orbits from a function solution `t -> orbit(t)`
- ⛳️ add orthogonal collocation method for periodic orbits
- add additional method for computing Floquet multipliers based on generalized eigenvalue problem

## [0.1.5] - 2021-09-26
- add new problem for symmetries `TWProblem`

## [0.1.5] - 2021-09-25
- add example for wave computation

## [0.1.5] - 2021-09-25
- refactoring, extractTimeSlices becomes getTimeSlices

## [0.1.5] - 2021-09-05
- add a simple callback to limit residuals in Newton iterations `cbMaxNorm`
- ⛳️ add branch switching for branches of PO at BP / PD
- auto generate more tutorials

## [0.1.5] - 2021-07-18
- rename `get3Jet` into `getJet`
- remove `BlockArrays.setblock!` occurrences which are deprecated

## [0.1.5] - 2021-07-10
- add `perturbGuess` option to `multicontinuation`
- change option `printSolution` to `recordFromSolution` in continuation and similar functions

## [0.1.5] - 2021-06-26
- add new function getFirstPointsOnBranch to allow fine grained control of aBS
- add full automatic differentiation for Deflated Problems

## [0.1.5] - 2021-06-20
- ⛳️ add computing full transcritical/pitchfork branch (not half) in `bifurcationDiagram`

## [0.1.4] - 2021-06-06
- move toward automatic generation of docs with figures
- add `applytoX, applytoY` option to plotting

## [0.1.4] - 2021-05-30
- add function `get3Jet` to compute Taylor expansion
- `getLensParam` becomes `getLensSymbol`
- add detection of codim 2 singularities

## [0.0.1] - 2021-05-16
- rename `HopfBifPoint` -> `Hopf`
- rename `GenericBifPoint` into `SpecialPoint` and `bifpoint` to `specialpoint`
- add applytoY keyword to plot recipe

## [0.0.1] - 2021-05-9
- remove `p->nothing` as default argument in `continuationHopf`
- add bordered linear solver option in `newtonHopf`

## [0.0.1] - 2021-05-2
- remove type piracy for `iterate`
- put the computation of eigenvalues in the iterator
- correct mistake in bracketing interval in `locateBifurcation!`
- remove `GMRESIterativeSolvers!` from linearsolvers

## [0.0.1] - 2021-04-3
- correct bug in the interval locating the bifurcation point (in bisection method)

## [0.0.1] - 2021-01-24
- ⛳️ add `bothside` kwargs to continuation to compute a branch on both sides of initial guess
- update the Minimally augmented problem during the continuation. This is helpful otherwise the codim 2 continuation fails.
- [WIP] detection of Bogdanov-Takens and Fold-Hopf bifurcations
- remove field `foldpoint` from ContResult

## [0.0.1] - 2020-11-29
- improve bordered solvers for POTrap based on the cyclic matrix

## [0.0.1] - 2020-11-7
- ⛳️ update phase condition during continuation for shooting problems and Trapezoid method

## [0.0.1] - 2020-11-7
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
- ⛳️ add Deflated Continuation

## [0.0.1] - 2020-8-21
- ⛳️ add Multiple predictor (this is needed to implement the `pmcont` algorithm from `pde2path` (Matlab)

## [0.0.1] - 2020-7-26
- ⛳️ add Polynomial predictor

## [0.0.1] - 2020-7-19
- ⛳️ add Branch switching for non-simple branch points

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

- ⛳️ automatic branch switching from simple Hopf points
- ⛳️ automatic normal form computation for any kernel dimension


## [0.0.1] - 2020-4-27


### Added

- ⛳️ automatic branch switching from simple branch points (equilibrium)
- ⛳️ automatic normal form computation
