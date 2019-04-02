# PseudoArcLengthContinuation.jl

![Build Status](https://travis-ci.com/rveltz/PseudoArcLengthContinuation.jl.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/rveltz/PseudoArcLengthContinuation.jl/badge.svg?branch=master)](https://coveralls.io/github/rveltz/PseudoArcLengthContinuation.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev)

This Julia package aims at solving equations F(u,λ)=0 where λ∈ℝ starting from an initial guess (u0,λ0). It relies on the pseudo arclength continuation algorithm which provides a *predictor* (u1,λ1) from (u0,λ0). A Newton method is then used to correct this predictor.

The current package focuses on large scale problems and multiple hardwares. Hence, the goal is to use Matrix Free methods on **GPU** (see [example](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev/#Example-5:-the-Swift-Hohenberg-equation-on-the-GPU-1)) or on a **cluster** to solve non linear PDE (for example).

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

To install it, please run

`] add https://github.com/rveltz/PseudoArcLengthContinuation.jl`

## Website

The package is located [here](https://github.com/rveltz/PseudoArcLengthContinuation.jl).

## Main features

- Matrix Free Newton solver with generic linear / eigen solver. Idem for the arc-length continuation.
- Matrix Free Newton solver with deflation. It can be used for branch switching for example.
- Fold / Hopf bifurcation detection
- Fold / Hopf with MatrixFree / Sparse Jacobian continuation with Minimally Augmented. formulation.
- Periodic orbit computation and continuation using Simple Shooting (not very stable yet) or Finite Differences.

Custom state means, can we use something else than `AbstractVector`:


|Feature|Matrix Free|Custom state| Example |
|---|---|---|---|
| Newton | Y | Y |1 - 5 |
| Newton + Deflation| Y | Y | 1, 2, 5|
| Continuation (Natural, Secant, Tangent) | Y | Y | 1 - 5 |
| Branching point detection | Y | Y |  |
| Fold detection | Y | Y | 1 - 5 |
| Hopf detection | Y | Y | 3 |
| Fold continuation | Y | `AbstractArray` | 1 |
| Hopf continuation | Y | `AbstractVector` | 3 |
| Periodic Orbit Newton | Y | `AbstractVector` | 3 |
| Periodic Orbit continuation | Y | `AbstractVector` | 3 |

## To do or grab
- [x] Improve Sparse Matrix creation of the Jacobian for the Periodic Orbit problem with Finite Differences
- [ ] Compute Hopf Normal Form
- [ ] Implement Preconditioner for the Matrix Free computation of Periodic Orbits based on Finite Differences
- [ ] Inplace implementation
- [ ] Provide a way to add constraints and combine functionals
- [ ] Improve `computeHopf` to allow for general state (not `AbstractArray`). Also, the implementation allocates a new `struct` for each parameter.
- [ ] write continuation loop as an iterator
