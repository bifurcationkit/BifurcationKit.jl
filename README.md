# PseudoArcLengthContinuation.jl

![Build Status](https://travis-ci.com/rveltz/PseudoArcLengthContinuation.jl.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/rveltz/PseudoArcLengthContinuation.jl/badge.svg?branch=master)](https://coveralls.io/github/rveltz/PseudoArcLengthContinuation.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev)

This Julia package aims at solving equations F(u,λ)=0 where λ∈ℝ starting from an initial guess (u0,λ0). It relies on the pseudo arclength continuation algorithm which provides a *predictor* (u1,λ1) from (u0,λ0). A Newton method is then used to correct this predictor.

The package actually does a little more. By leveraging on the above method, it can also seek for periodic orbits of Cauchy problems by casting them into an equation F(u,λ)=0 of high dimension. **It is by now, one of the only softwares which provides shooting methods AND methods based on finite differences to compute periodic orbits.**

The current package focuses on large scale nonlinear problems and multiple hardwares. Hence, the goal is to use Matrix Free methods on **GPU** (see [PDE example](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev/tutorials2/index.html#The-Swift-Hohenberg-equation-on-the-GPU-1) and [Periodic orbit example](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev/tutorialsCGL/#Continuation-of-periodic-orbits-on-the-GPU-(Advanced)-1)) or on a **cluster** to solve non linear PDE, nonlocal problems, compute sub-manifolds...

**If you use this package for your work, please cite it!! Open source development strongly depends on this. It is referenced on HAL-Inria as follows:**

```
@misc{veltz:hal-02071874,
  TITLE = {{PseudoArcLengthContinuation.jl}},
  AUTHOR = {Veltz, Romain},
  URL = {https://hal.inria.fr/hal-02071874},
  YEAR = {2019},
  MONTH = Mar,
  KEYWORDS = {Pseudo Arclength Continuation},
  PDF = {https://hal.inria.fr/hal-02071874/file/PseudoArcLengthContinuation.jl-master.zip},
  HAL_ID = {hal-02071874},
  HAL_VERSION = {v1},
}
```

## Installation 

This package requires Julia >= v1.3.0

To install it, please run

`] add https://github.com/rveltz/PseudoArcLengthContinuation.jl`

## Website

The package is located [here](https://github.com/rveltz/PseudoArcLengthContinuation.jl).

## Main features

- Matrix Free Newton solver with generic linear / eigen *preconditioned* solver. Idem for the arc-length continuation.
- Matrix Free Newton solver with deflation and preconditioner. It can be used for branch switching for example.
- Bifurcation points are located using a bisection algorithm
- Branch, Fold, Hopf bifurcation point detection of stationary solutions.
- Fold / Hopf continuation based on Minimally Augmented formulation, with Matrix Free / Sparse Jacobian.
- Periodic orbit computation and continuation using Shooting or Finite Differences.
- Branch, Fold, Neimark-Sacker, Period Doubling bifurcation point detection of periodic orbits.
- Computation and Continuation of Fold of periodic orbits

Custom state means, we can use something else than `AbstractArray`, for example your own `struct`. 

**Note that you can combine most of the solvers, like use Deflation for Periodic orbit computation or Fold of periodic orbits family.**

|Features|Matrix Free|Custom state| Tutorial |
|---|---|---|---|
| Newton | Y | Y | All |
| Newton + Deflation| Y | Y | 3, 4|
| Continuation (Natural, Secant, Tangent) | Y | Y | All |
| Branching point detection | Y | Y | All |
| Fold point detection | Y | Y | All |
| Hopf detection | Y | Y | 5 - 8 |
| Fold Point continuation | Y | Y | 1, 7 |
| Hopf continuation | Y | `AbstractArray` | 5 |
| Periodic Orbit (FD) Newton / continuation | Y | `AbstractVector` | 5, 7 |
| Periodic Orbit with Poincaré / Standard Shooting Newton / continuation | Y | `AbstractArray` |  5, 6, 8 |
| Fold, Neimark-Sacker, Period doubling detection | Y | `AbstractVector` | 5 - 8  |
| Continuation of Fold of periodic orbits | Y | `AbstractVector` | 7 |

## To do or grab
Without a priority order:

- [ ] improve plotting by using recipies
- [ ] improve compatibility with `DifferentialEquations.jl`
- [ ] Add interface to other iterative linear solvers (cg, minres,...) from IterativeSolvers.jl
- [ ] Check different `struct` and look for potential improvements (type stability, barriers...)
- [ ] Compute Hopf Normal Form and allow branching from Hopf point using this
- [ ] Inplace implementation
- [ ] Write continuation loop as an iterator
