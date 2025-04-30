# BifurcationKit.jl

| **Documentation** | **Build Status** | **Coverage** | **Version / Stats** |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![Build status](https://badge.buildkite.com/fc3fdd514333077cbb6600923ed0ef631a59de5226b197665f.svg)](https://buildkite.com/julialang/bifurcationkit-dot-jl) | [![codecov](https://codecov.io/gh/bifurcationkit/BifurcationKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bifurcationkit/BifurcationKit.jl) | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![Build status](https://github.com/rveltz/BifurcationKit.jl/workflows/CI/badge.svg)](https://github.com/rveltz/BifurcationKit.jl/actions) |   | [![deps-img]][deps-url] [![downloads][downloads-img]][downloads-url]|

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev

[ver-img]: https://juliahub.com/docs/BifurcationKit/version.svg
[ver-url]: https://juliahub.com/ui/Packages/BifurcationKit/UDEDl

[deps-img]: https://juliahub.com/docs/General/BifurcationKit/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/BifurcationKit?t=2

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FBifurcationKit&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-url]: https://juliapkgstats.com/pkg/BifurcationKit

This Julia package aims at performing **automatic bifurcation analysis** of possibly large dimensional equations F(u, λ)=0 where λ is real by taking advantage of iterative methods, dense / sparse formulation and specific hardwares (*e.g.* GPU).

It incorporates continuation algorithms (PALC, deflated continuation, ...) based on a Newton-Krylov method to correct the predictor step and a Matrix-Free/Dense/Sparse eigensolver is used to compute stability and bifurcation points.

> The idea is to be able to seamlessly switch the continuation algorithm a bit like changing the time stepper (Euler, RK4,...) for ODEs.

`BifurcationKit` can also seek for [periodic orbits](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/periodicOrbit/) of Cauchy problems. **It is by now, one of the only software which provides shooting methods *and* methods based on finite differences / collocation to compute periodic orbits.**

The current focus is on large scale nonlinear problems and multiple hardwares. Hence, the goal is to provide Matrix Free methods on **GPU** (see [PDE example](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials2b/#The-Swift-Hohenberg-equation-on-the-GPU-(non-local)-1) and [Periodic orbit example](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorialsCGL/#Continuation-of-periodic-orbits-on-the-GPU-(Advanced)-1)) or on **cluster** to study non linear PDE, nonlocal problems, compute sub-manifolds...

> Despite this focus, the package can easily handle low dimensional problems and specific optimizations are regularly added.

## SciML Integration

**BifurcationKit.jl** is listed in Third patry parameter analysis of [SciML Docs](https://docs.sciml.ai/DiffEqDocs/stable/#DifferentialEquations.jl:-Efficient-Differential-Equation-Solving-in-Julia) under the "Analysis" menu. [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/tutorials/bifurcation_diagram_computation/) and [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/steady_state_functionality/bifurcation_diagrams/) rely on **BifurcationKit.jl** as external dependency.

## 📚 Support and citation
If you use `BifurcationKit.jl` in your work, we ask that you cite the following paper on [HAL-Inria](https://hal.archives-ouvertes.fr/hal-02902346) with *bibtex* entry [CITATION.bib](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/CITATION.bib). Open source development as part of academic research strongly depends on this. Please also consider starring this repository if you like our work, this will help us to secure funding in the future.

## 📦 Installation

This package requires Julia >= v1.3.0

To install it, please run

`] add BifurcationKit`

To install the bleeding edge version, please run

`] add BifurcationKit#master`

## 🧩 Plugins

Most plugins are located in the organization [bifurcationkit](https://github.com/bifurcationkit):

- [HclinicBifurcationKit.jl](https://github.com/bifurcationkit/HclinicBifurcationKit.jl) bifurcation analysis of homoclinic / heteroclinic orbits of ordinary differential equations (ODE)
- [DDEBifurcationKit.jl](https://github.com/bifurcationkit/DDEBifurcationKit.jl) bifurcation analysis of delay differential equations (DDE)
- [AsymptoticNumericalMethod.jl](https://github.com/bifurcationkit/AsymptoticNumericalMethod.jl) provides the numerical continuation algorithm **Asymptotic Numerical Method** (ANM) which can be used directly in `BifurcationKit.jl`
- [GridapBifurcationKit.jl](https://github.com/bifurcationkit/GridapBifurcationKit) bifurcation analysis of PDEs solved with the Finite Elements Method (FEM) using the package [Gridap.jl](https://github.com/gridap/Gridap.jl).
- [PeriodicSchurBifurcationKit.jl](https://github.com/bifurcationkit/PeriodicSchurBifurcationKit.jl) state of the art computation of Floquet coefficients, useful for computing the stability of periodic orbits.

## Overview of capabilities

The list of capabilities is available [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/capabilities/).

## Examples of bifurcation diagrams (ODEs and PDEs)

| ![](https://github.com/bifurcationkit/BifurcationKitDocs.jl/blob/main/docs/src/tutorials/ode/nm-per.png?raw=true)   |  ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/ode/com-fig3.png?raw=true) |
|:-------------:|:-------------:|
| [simple ODE example](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/ode/tutorialsODE/#Neural-mass-equation-(Hopf-aBS)) |  [Codimension 2 (ODE)](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/ode/tutorialCO/#CO-oxydation-(codim-2)) |
| ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/BDSH1d.png)   |  ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/mittlemannBD-1.png) |
| [Automatic Bif. Diagram in 1D Swift Hohenberg](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/Swift-Hohenberg1d/#d-Swift-Hohenberg-equation-(Automatic)) |  [Automatic Bif. Diagram in 2D Bratu](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/mittelmannAuto/#Automatic-diagram-of-2d-Bratu–Gelfand-problem-(Intermediate)) |
| ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/sh2dbranches.png)   |  ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/bru-po-cont-3br.png) |
| [Snaking in 2D Swift Hohenberg](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials2/#d-Swift-Hohenberg-equation:-snaking,-Finite-Differences) |  [Periodic orbits in 1D Brusselator](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials3/#d-Brusselator-(automatic))
| ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/br_pd3.png) |![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/cgl-sh-br.png) |
| [Period doubling BVAM Model](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorialsPD/#Period-doubling-in-the-Barrio-Varea-Aragon-Maini-model)  |  [Periodic orbits in 2D Ginzburg-Landau](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorialsCGL/#d-Ginzburg-Landau-equation-(finite-differences))  |
| ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/carrier.png) | ![](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/GPU-branch.png) |
| [Deflated Continuation in Carrier problem](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorialCarrier/#Deflated-Continuation-in-the-Carrier-Problem)  |  [2D Swift Hohenberg on GPU](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials2b/#d-Swift-Hohenberg-equation-(non-local)-on-the-GPU,-periodic-BC-(Advanced))  |

