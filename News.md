BifurcationKit.jl, Changelog
========================

All notable changes to this project will be documented in this file (hopefully). No performance improvements will be notified but mainly the addition of new methods, the modifications of internal structs, etc.

## [Unreleased]

### Breaking changes
- `get_adjoint_basis` is replaced by the methods `_get_kernel_basis_1d_from_eigensolver` / `_get_kernel_basis_1d_from_bls` (resp. the Nd helpers), which now compute the right and left (adjoint) kernel vectors together
- `__compute_bordered_vectors_fold` / `__compute_bordered_vectors_hopf` are now also used to build the kernel basis of the normal forms (`start_with_eigen = Val(false)`)

### Added
- add `∫_gauss` to integrate arrays sampled at the Gauss points, i.e. in the row layout of the collocation operator, consistently with `∫`
- add the keyword `start_with_eigen = Val(true)` to `get_normal_form1d`, `get_normal_formNd` and the codim 2 normal forms (`hopf`, `cusp`, `bogdanov_takens`, `bautin`, `zero_hopf`, `hopf_hopf`): with `start_with_eigen = Val(false)`, the (right/left) kernel basis is built with a bordered linear system (keywords `bls`, `bls_adjoint`, `bls_block`) instead of the eigensolver, which is more robust for large-scale problems. The 1d basis is computed with the bordered-vector solvers `__compute_bordered_vectors_fold` / `__compute_bordered_vectors_hopf` and the Nd basis with `__compute_nd_basis_from_bls`. `bautin_normal_form` also accepts `bls`/`bls_adjoint` like `hopf_normal_form`
- compute the parameter derivative `dFdp` in the Moore-Penrose algorithm (`newton_moore_penrose`) with the (user provided) `R01`/`R01!` instead of first-order finite differences, and accumulate the number of linear iterations (`itlineartot`)

### Fixed
- evaluate all the integrands of the NS Iooss normal form (`neimark_sacker_normal_form_iooss`) at the Gauss points and build the RHS of the homological equations `h20`/`h11` in the row layout of the collocation operator. This removes the former `Icoll` mass-matrix pre-scaling and the ad-hoc `h20 ./= 2Ntst` / `h11 ./= 2Ntst` renormalizations.
- evaluate all the integrands of the PD Iooss normal form (`period_doubling_normal_form_iooss`) at the Gauss points and build the RHS of the homological equations `h₂`/`h₀₁` in the row layout of the collocation operator, normalizing `ψ₁★` with `∫_gauss` so that `<ψ₁★, F(u₀)> = 1/2`. This removes the former `Icoll` mass-matrix pre-scaling and fixes the scaling of the normal form coefficients.

## [0.8.3]

### Breaking changes
- `detect_codim2_parameters` is renamed `modify_contparams_for_codim2`
- the method `_compute_bordered_vectors` of the Fold/Hopf/PD/NS MA formulations is refactored into the methods `__compute_bordered_vectors_fold` / `__compute_bordered_vectors_hopf`

### Added
- add the bordered-vector kernels `__compute_bordered_vectors_fold` / `__compute_bordered_vectors_hopf` which compute the right/left null vectors of the Fold / Hopf jacobians by a bordered linear system (factoring the Fold/Hopf initial eigenvector computation). These methods are also used to build the kernel basis of the normal forms with `start_with_eigen = Val(false)`
- allow plotting during NS (Neimark-Sacker) continuation
- Plots backend: plot several codim 1 branches in one figure
- improve codim 2 continuation of periodic orbits (PD/NS Jacobian borders, events on Fold of PO, simplified `get_normal_form` for codim 2 points of PO)

### Fixed
- correct `update!` for `NSMAProblem` / `PDMAProblem`
- robust mesh adaptation for `Collocation` (improved predictor)
- correct the Hopf predictor at Hopf-Hopf points

## [0.8.2]

- filter infinite eigenvalues in generalized eigenvalue computations (`DefaultEig`)

## [0.8.1]

### Breaking changes
- 🚦🚦🚦 `update!(prob, x)` becomes `restore_problem!(prob, x, pars)`
- `getlinsolver`/`getbls` become `get_bordered_linsolver`
- `OneParamCont`, `TwoParamCont`, `TwoParamPeriodicOrbitCont` become `AbstractOneParamCont`, `AbstractTwoParamCont`, `AbstractTwoParamPeriodicOrbitCont`
- `_getsolution` becomes `saved_solution`
- `tangent` field in `MoorePenrose` becomes `predictor`
- remove `autodiff` keyword argument for computing normal forms
- `FlowDE` construction is now keyword-based
- merge BT `nfsupp` into `nf`

### Added
- add derivatives of the flow w.r.t. the parameter and higher-order differentials `R01`, `R11`, `R20`, `R30` to `Flow`/`FlowDE`, computed with ForwardDiff by default and usable in the Poincaré return map and the normal forms
- use `monodromy_matrix!` in Shooting and Poincaré Shooting
- rework the Poincaré return map around `_evolve_flow_prm`; add `R01`, `R11`, `R20`, `R30` methods for `PoincaréMap` and collocation
- rework the Jet traits: `R01`/`R02`/`R11` are now computed by dispatch on `AutoDiff()`/`FiniteDifferences()` and generalized to any `AbstractBifurcationProblem`
- save/restore of the Poincaré section for `PoincareShooting` (`POSavedSolutionAndState_PSH`, `BVPSavedSolutionAndState_PSH`), fixing issue #334
- add `Accumulator` to `DeflationOperator`
- add eigenvalue solvers `EigenWave` and `GEigenWave` for waves
- add `__sort_spectrum` and correct the spectrum sorting of `DefaultEig`
- add mechanism to switch the plot backend: `set_plot_backend!`, `get_plot_backend` with `BK_Plots()`, `BK_Makie()`, `BK_NoPlot()`
- Makie: add `plot_stability_segments`
- Plots: add `:dots` style for the unstable part of the branch
- add defaults to `ContState` fields and `EmptyContState`
- refactor the generalized eigenvalue computation into `GeneralizedEigenSolver.jl` and the tangents into `Tangents.jl`
- use an out-of-place formulation for waves

## [0.8.0]

- add a new BVP interface: `BVPModel`, `PeriodicOrbitModel`, `DiscretizedBVP`, `discretize`, `generate_solution`, the discretizers `Shooting`, `Trapeze`, `Collocation` and the problem `BVPBifProblem`
- add mesh adaptation for BVP problems
- add user-specified time interval for BVP problems
- add `BoundaryValueProblemCont` for the continuation of BVP problems
- add wrapper for deflated continuation of BVP problems
- add `TimeMesh` structure to support non-uniform meshes for `Trapeze`
- `SolPeriodicOrbit` becomes `BVPSolution`, `POSolution` becomes `POInterpolation`, `POSolutionAndState` becomes `POSavedSolutionAndState`
- rename `Trap` into `Trapeze` in the BVP interface
- `jacobian = :auto` becomes `jacobian = AutoDiffDense()` in BVP Shooting
- add `normal_form` alias for `get_normal_form`

## [0.7.3]
- Remove `AbstractPeriodicOrbitDiscretization`, make subtypes inherit directly from `AbstractBoundaryValueDiscretization`
- `AbstractPODifferentialDiscretization` → `AbstractDifferentialDiscretization`
- `AbstractPOFiniteDifferencesDiscretization` → `AbstractFiniteDifferencesDiscretization`
- `AbstractPOShootingDiscretization` → `AbstractShootingDiscretization`

## [0.7.2]

- Change names `*ShootingProblem` for `*Shooting`, `PeriodicOrbitOCollProblem` for `Collocation` and `PeriodicOrbitTrapProblem` for `Trapeze`
## [0.6.1]
- emove fields params and lens from PeriodicOrbitFunctionalTrap, PeriodicOrbitFunctionalSh, PeriodicOrbitFunctionalColl, WrapTW
- remove `param` field from `*MAProblem` structs (`FoldMAProblem`, `HopfMAProblem`, `PDMAProblem`, `NSMAProblem`, `BTMAProblem`) and from `WrapTW`; `getparams` now delegates to `getparams(get_formulation(prob))`
- add `re_make` methods for `AbstractMABifurcationProblem`, `AbstractWaveProblem`, `AbstractWrapperPeriodicOrbitProblem`, `AbstractPODifferentialDiscretization`, `AbstractPOShootingDiscretization`, `TWModel`, `AbstractMinimallyAugmentedFormulation`
- add `getlens` method for `TWModel`
## [0.6.0]
- add `AbstractBoundaryValueDiscretization`, `AbstractPeriodicOrbitDiscretization`, `AbstractPODifferentialDiscretization`, `AbstractPOFiniteDifferencesDiscretization`
- `AbstractPoincareShootingProblem` becomes `AbstractPoincareShootingDiscretization`
- `AbstractShootingProblem` becomes `AbstractPOShootingDiscretization`
- `BTProblemMinimallyAugmented` becomes `BTMinimallyAugmentedFormulation`
- `TWProblem` becomes `TWModel`
- remove `modify_po_finalise`, `modify_po_record`, `modify_tw_record` and `Finalizer`
- change `Fold/Hopf/PeriodDoubling/NeimarkSackerProblemMinimallyAugmented` into `Fold/Hopf/PeriodDoubling/NeimarkSackerMinimallyAugmentedFormulation`
- `AbstractProblemMinimallyAugmented` becomes `AbstractMinimallyAugmentedFormulation`
- make `AbstractMABifurcationProblem{T, Tjac}` dependent on 2 parameters
- add method `finalise_solution(iter::ContIterable, state::AbstractContinuationState, contRes)`
- remove `update_minaug_hopf`, `update_minaug_fold`, `update_min_aug_ns`, `update_min_aug_pd`
- `WrapPOColl` becomes `PeriodicOrbitFunctionalColl`
- `WrapPOSh` becomes `PeriodicOrbitFunctionalSh`
- `WrapPOTrap` becomes `PeriodicOrbitFunctionalTrap`
- `correct_bifurcation` becomes `_correct_event_labels`
- add `PeriodicOrbit{Tdisc}`, `TravellingWave{Tdisc}` to bridge discretization and problem
- add `MASolution`, `MASolutionFreq` to hold codim1 MA solutions (for mesh adaptation in codim2)
- add `RecordForFold`, `RecordForHopf`, `RecordForPeriodicOrbits`, `RecordForNS`, `RecordForPD`, `RecordForTW` to replace closure-based callbacks
- add accessor methods `get_discretization`, `get_formulation`, `get_solution`, `getparams`
- add `update!` for `PDMAProblem`, `NSMAProblem`, `HopfMAProblem` and periodic orbit wrappers
- `TWModel` inherits from `AbstractTravelingWaveDiscretization` instead of `AbstractBifurcationProblem`
- change signature of `continuation` for periodic orbits: `probPO::AbstractPeriodicOrbitProblem` becomes `disc::AbstractPeriodicOrbitDiscretization`
- `DefaultLS` uses `VI.Zero()` / `VI.One()` instead of literal 0 / 1
- closure removal for `record_from_solution` in all continuation types (codim1 and codim2, periodic orbits, waves)
- add mesh adaptation in codim2 continuation
- change initialization of `delta` in `BifurcationProblem` to use `getdelta`
- simplify `is_event_crossed` and refactor events
- remove dead code in `_continuation(gh::Bautin)`
- various bug fixes: correction of jacobian selection in PD continuation, fix plotting in NS continuation, correct PD formulation, correct linear bordered solver `solve_bls_block`, correct type stability of Hopf predictor, correct type assert in Bautin normal form, do not update MA problem when in bisection
## [0.5.8]
- add `AbstractBifurcationPointCodim2`, `NdBranchPoint` for bifurcation points with dim(Ker) > 1
- add method `minus(x::POSolutionAndState, y::POSolutionAndState)` for `detect_loop` with collocation and mesh adaptation
- add branching to curve of periodic orbits from curve of Hopf points
- set `PALC(tangent = Bordered())` as default in `autoswitch` constructor
- remove type constraint `Tlens <: AllOpticTypes` in `BifurcationPoints`
- improve type stability of `get_normal_formNd`, `biorthogonalise`, etc.
- refactor `BranchSwitching.jl`
- simplify `multicontinuation` to return a `Branch` instead of `Vector{Branch}`
- bump compat for `RecursiveArrayTools` to 4
## [0.5.6] (future)
- reorganise tests:
  - each test belongs to a test category (like in previous versions)
  - the test directory arborescence reflects this organisation
  - the runtests.jl file is now generic and runs accepts arguments which define which tests will be run
  - the .github/workflows/ci.yml can now run a subset of tests (using the previous runtests.jl modifications) bases on the labels of PR/commit/..
  - examples of valid labels: `Run test(s): wave`,  `Run test(s): wave | newton`
## [0.5.5]
- most Standard shooting code works with `VI.MinimalVec`
## [0.5.4]
- modify dispatch for get_time_slice and get_time_slices
## [0.5.3]
- make BorderedArray comply with VectorInterface.jl (VI)
- most of BK code complies with VI
- bug correction in mesh adaptation for collocation
- massive improvement in computation of reduced equations. Improve type stability as well.
## [0.5.2]
- export `ODEBifProblem`
- add new jacobian types
- `jad` becomes `jacobian_adjoint`
## [0.5.1]
- `_eig_floquet_col` is now `_eig_floquet_coll`
- `FloquetCollGEV` is now called `FloquetGEV`
- `extract_period` is now called `_extract_period`
- the struct `COPCACHE` has been changed a lot.
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
- ⛳️ the keyword argument   `linearPO` is renamed into `jacobianPO`
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
