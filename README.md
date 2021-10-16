# BifurcationKit.jl

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rveltz.github.io/BifurcationKit.jl/dev) | [![Build status](https://github.com/rveltz/BifurcationKit.jl/workflows/CI/badge.svg)](https://github.com/rveltz/BifurcationKit.jl/actions) [![codecov](https://codecov.io/gh/rveltz/BifurcationKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/rveltz/BifurcationKit.jl) |

This Julia package aims at performing **automatic bifurcation analysis** of possibly large dimensional equations F(u, λ)=0 where λ∈ℝ by taking advantage of iterative methods, sparse formulation and specific hardwares (*e.g.* GPU).

It incorporates continuation algorithms (PALC, deflated continuation, ...) which provide a *predictor* (u1, λ1) from a known solution (u0, λ0). A Newton-Krylov method is then used to correct this predictor and a Matrix-Free eigensolver is used to compute stability and bifurcation points.

By leveraging on the above method, it can also seek for periodic orbits of Cauchy problems by casting them into an equation F(u, λ)=0 of high dimension. **It is by now, one of the only softwares which provides shooting methods AND methods based on finite differences or collocation to compute periodic orbits.**

The current package focuses on large scale nonlinear problems and multiple hardwares. Hence, the goal is to use Matrix Free methods on **GPU** (see [PDE example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials2b/#The-Swift-Hohenberg-equation-on-the-GPU-(non-local)-1) and [Periodic orbit example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Continuation-of-periodic-orbits-on-the-GPU-(Advanced)-1)) or on a **cluster** to solve non linear PDE, nonlocal problems, compute sub-manifolds...


## Support and citation
If you use this package for your work, we ask that you cite the following paper. Open source development as part of academic research strongly depends on this. Please also consider starring this repository if you like our work, this will help us to secure funding in the future. It is referenced on HAL-Inria as follows:

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
| [Automatic Bif. Diagram in 1D Swift Hohenberg](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/Swift-Hohenberg1d/#d-Swift-Hohenberg-equation-(Automatic)) |  [Automatic Bif. Diagram in 2D Bratu](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/mittelmannAuto/#Automatic-diagram-of-2d-Bratu–Gelfand-problem-(Intermediate)) |
| ![](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/sh2dbranches.png)   |  ![](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/bru-po-cont-3br.png) |
| [Snaking in 2D Swift Hohenberg](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorials2/#d-Swift-Hohenberg-equation:-snaking,-Finite-Differences) |  [Periodic orbits in 1D Brusselator](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorials3/#d-Brusselator-(automatic)) 
| ![](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/br_pd3.png) |![](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/cgl-sh-br.png) |
| [Period doubling BVAM Model](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorialsPD/#Period-doubling-in-the-Barrio-Varea-Aragon-Maini-model)  |  [Periodic orbits in 2D Ginzburg-Landau](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorialsCGL/#d-Ginzburg-Landau-equation-(finite-differences))  |
| ![](https://rveltz.github.io/BifurcationKit.jl/dev/carrier.png) | ![](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/GPU-branch.png) |
| [Deflated Continuation in Carrier problem](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorialCarrier/#Deflated-Continuation-in-the-Carrier-Problem)  |  [2D Swift Hohenberg on GPU](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/tutorials2b/#d-Swift-Hohenberg-equation-(non-local)-on-the-GPU,-periodic-BC-(Advanced))  |


## Main features

- Newton-Krylov solver with generic linear / eigen *preconditioned* solver. Idem for the arc-length continuation.
- Newton-Krylov solver with nonlinear deflation and preconditioner. It can be used for branch switching for example.
- Continuation written as an [iterator](https://rveltz.github.io/BifurcationKit.jl/dev/iterator/)
- Monitoring user functions along curves computed by continuation, see [events](https://rveltz.github.io/BifurcationKit.jl/dev/EventCallback/)
- Continuation methods: PALC, Moore Spence, Deflated continuation
- Bifurcation points are located using a bisection algorithm
- Branch, Fold, Hopf bifurcation point detection of stationary solutions.
- Automatic branch switching at branch points (whatever the dimension of the kernel)
- Automatic branch switching at simple Hopf points to periodic orbits
- **Automatic bifurcation diagram computation of equilibria**
- Fold / Hopf continuation based on Minimally Augmented formulation, with Matrix Free / Sparse Jacobian.
- detection of Bogdanov-Takens, Bautin and Cusp bifurcations
- Periodic orbit computation and continuation using Shooting, Finite Differences or Orthogonal Collocation.
- Branch, Fold, Neimark-Sacker, Period Doubling bifurcation point detection of periodic orbits.
- Continuation of Fold of periodic orbits

Custom state means, we can use something else than `AbstractArray`, for example your own `struct`.

**Note that you can combine most solvers, like use Deflation for Periodic orbit computation or Fold of periodic orbits family.**


|Features|Matrix Free|Custom state| [Tutorial](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/) | GPU |
|---|---|---|---|---|
| (Deflated) Krylov-Newton| Yes| Yes| All| :heavy_check_mark:|
| Continuation PALC (Natural, Secant, Tangent, Polynomial) | Yes| Yes| All |:heavy_check_mark:  |
| Deflated Continuation | Yes| Yes| [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialCarrier/#Deflated-Continuation-in-the-Carrier-Problem-1) |:heavy_check_mark:  |
| Branching / Fold / Hopf point detection | Yes| Yes| All / All / [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/#Bifurcation-diagrams-with-periodic-orbits-1) | :heavy_check_mark: |
| Fold Point continuation | Yes| Yes| [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials1/#Temperature-model-(simplest-example-for-equilibria)-1), [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Complex-Ginzburg-Landau-2d-1) | :heavy_check_mark: |
| Hopf continuation | Yes| `AbstractArray` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials3/#Continuation-of-Hopf-points-1) | |
| Branch switching at Branch / Hopf points | Yes| `AbstractArray` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/#Bifurcation-diagrams-with-periodic-orbits-1) | :heavy_check_mark: |
| <span style="color:red">**Automatic bifurcation diagram computation of equilibria**</span> | Yes| `AbstractArray` |  [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/#Automatic-bifurcation-diagram-1) | |
| Periodic Orbit (Trapezoid) Newton / continuation | Yes| `AbstractVector` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials3/#Brusselator-1d-(automatic)-1), [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Complex-Ginzburg-Landau-2d-1) | :heavy_check_mark:|
| Periodic Orbit (Collocation) Newton / continuation | Yes| `AbstractVector` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/ode/tutorialsODE/#Neural-mass-equation-(Hopf-aBS)) | |
| Periodic Orbit with Parallel Poincaré / Standard Shooting Newton / continuation | Yes| `AbstractArray` |  [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/#Bifurcation-diagrams-with-periodic-orbits-1) | |
| Fold, Neimark-Sacker, Period doubling detection | Yes| `AbstractVector` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials/#Bifurcation-diagrams-with-periodic-orbits-1)  | |
| Continuation of Fold of periodic orbits | Yes| `AbstractVector` | [:arrow_heading_up:](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Complex-Ginzburg-Landau-2d-1) | :heavy_check_mark: |

## To do or grab
Without a priority order:

- [ ] improve compatibility with `DifferentialEquations.jl`
- [ ] Add interface to other iterative linear solvers (cg, minres,...) from IterativeSolvers.jl
