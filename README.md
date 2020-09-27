# BifurcationKit.jl

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rveltz.github.io/BifurcationKit.jl/dev) | ![Build Status](https://travis-ci.com/rveltz/BifurcationKit.jl.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/rveltz/BifurcationKit.jl/badge.svg?branch=master)](https://coveralls.io/github/rveltz/BifurcationKit.jl?branch=master) |


This Julia package aims at performing **automatic bifurcation analysis** of large dimensional equations F(u,λ)=0 where λ∈ℝ.  

It incorporates continuation algorithms (PALC, deflated continuation, ...) which provide a *predictor* (u1,λ1) from a known solution (u0,λ0). A Newton-Krylov method is then used to correct this predictor and a Matrix-Free eigensolver is used to compute stability and bifurcation points.

By leveraging on the above method, it can also seek for periodic orbits of Cauchy problems by casting them into an equation F(u,λ)=0 of high dimension. **It is by now, one of the only softwares which provides shooting methods AND methods based on finite differences to compute periodic orbits.**

The current package focuses on large scale nonlinear problems and multiple hardwares. Hence, the goal is to use Matrix Free methods on **GPU** (see [PDE example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials2b/#The-Swift-Hohenberg-equation-on-the-GPU-(non-local)-1) and [Periodic orbit example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Continuation-of-periodic-orbits-on-the-GPU-(Advanced)-1)) or on a **cluster** to solve non linear PDE, nonlocal problems, compute sub-manifolds...

**If you use this package for your work, please cite it!! Open source development strongly depends on this. It is referenced on HAL-Inria as follows:**

```
@misc{veltz:hal-02902346,
  TITLE = {{BifurcationKit.jl}},
  AUTHOR = {Veltz, Romain},
  URL = {https://hal.archives-ouvertes.fr/hal-02902346},
  INSTITUTION = {{Inria Sophia-Antipolis}},
  YEAR = {2020},
  MONTH = Jul,
  KEYWORDS = {pseudo-arclength-continuation ; periodic-orbits ; floquet ; gpu ; bifurcation-diagram ; deflation ; newton-krylov},
  PDF = {https://hal.archives-ouvertes.fr/hal-02902346/file/354c9fb0d148262405609eed2cb7927818706f1f.tar.gz},
  HAL_ID = {hal-02902346},
  HAL_VERSION = {v1},
}
```

## Installation 

This package requires Julia >= v1.3.0

To install it, please run

`] add BifurcationKit`

To install the bleeding edge version, please run

`] add BifurcationKit#master`

## Website

The package is located [here](https://github.com/rveltz/BifurcationKit.jl).

## Examples of bifurcation diagrams


| ![](https://rveltz.github.io/BifurcationKit.jl/dev/BDSH1d.png)   |  ![](https://rveltz.github.io/BifurcationKit.jl/dev/mittlemannBD-1.png) | 
|:-------------:|:-------------:|
| [Automatic Bif. Diagram in 1D Swift Hohenberg](https://rveltz.github.io/BifurcationKit.jl/dev/Swift-Hohenberg1d/#Swift-Hohenberg-equation-1d-1) |  [Automatic Bif. Diagram in 2d Bratu](https://rveltz.github.io/BifurcationKit.jl/dev/mittelmannAuto/#Automatic-diagram-of-2d-Bratu–Gelfand-problem-(Intermediate)-1) |
| ![](https://rveltz.github.io/BifurcationKit.jl/dev/sh2dbranches.png)   |  ![](https://rveltz.github.io/BifurcationKit.jl/dev/bru-po-cont-3br.png) | 
| [Snaking in 2D Swift Hohenberg](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials2) |  [Periodic orbits in Brusselator](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials3/) |  
| ![](https://rveltz.github.io/BifurcationKit.jl/dev/br_pd3.png) |![](https://rveltz.github.io/BifurcationKit.jl/dev/cgl-sh-br.png) | 
| [Period doubling BVAM Model](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsPD)  |  [Ginzburg-Landau 2d](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/)  |  


## Main features

- Newton-Krylov solver with generic linear / eigen *preconditioned* solver. Idem for the arc-length continuation.
- Newton-Krylov solver with nonlinear deflation and preconditioner. It can be used for branch switching for example.
- Deflated continuation
- Bifurcation points are located using a bisection algorithm
- Branch, Fold, Hopf bifurcation point detection of stationary solutions.
- Automatic branch switching at simple branch points
- Automatic branch switching at simple Hopf points to periodic orbits
- **Automatic bifurcation diagram computation**
- Fold / Hopf continuation based on Minimally Augmented formulation, with Matrix Free / Sparse Jacobian.
- Periodic orbit computation and continuation using Shooting or Finite Differences.
- Branch, Fold, Neimark-Sacker, Period Doubling bifurcation point detection of periodic orbits.
- Computation and Continuation of Fold of periodic orbits

Custom state means, we can use something else than `AbstractArray`, for example your own `struct`. 

**Note that you can combine most of the solvers, like use Deflation for Periodic orbit computation or Fold of periodic orbits family.**


|Features|Matrix Free|Custom state| [Tutorial](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/) | GPU |
|---|---|---|---|---|
| (Deflated) Newton| Y | Y | 4, 5| :heavy_check_mark:|
| Continuation (Natural, Secant, Tangent) | Y | Y | All |:heavy_check_mark:  |
| Deflated Continuation | Y | Y | |:heavy_check_mark:  |
| Branching / Fold / Hopf point detection | Y | Y | All / All / 6 - 9 | :heavy_check_mark: |
| Fold Point continuation | Y | Y | 1, 8 | |
| Hopf continuation | Y | `AbstractArray` | 5 | |
| Branch switching at Branch / Hopf points | Y | `AbstractArray` | 3 | |
| <span style="color:red">**Automatic bifurcation diagram computation**</span> | Y | `AbstractArray` |  Yes | |
| Periodic Orbit (FD) Newton / continuation | Y | `AbstractVector` | 6, 8 | :heavy_check_mark:|
| Periodic Orbit with Parallel Poincaré / Standard Shooting Newton / continuation | Y | `AbstractArray` |  6, 7, 9 | | 
| Fold, Neimark-Sacker, Period doubling detection | Y | `AbstractVector` | 6 - 9  | |
| Continuation of Fold of periodic orbits | Y | `AbstractVector` | 8 | |

## To do or grab
Without a priority order:

- [ ] improve compatibility with `DifferentialEquations.jl`
- [ ] Add interface to other iterative linear solvers (cg, minres,...) from IterativeSolvers.jl
