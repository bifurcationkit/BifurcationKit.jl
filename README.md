# PseudoArcLengthContinuation.jl

![Build Status](https://travis-ci.com/rveltz/PseudoArcLengthContinuation.jl.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/rveltz/PseudoArcLengthContinuation.jl/badge.svg?branch=master)](https://coveralls.io/github/rveltz/PseudoArcLengthContinuation.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rveltz.github.io/PseudoArcLengthContinuation.jl/dev)

This package aims at solving equations F(u,λ)=0 where λ∈ℝ starting from an initial guess (u0,λ0). It relies on the pseudo arclength continuation algorithm which provides a *predictor* (u1,λ1) from (u0,λ0). A Newton method is then used to correct this predictor.

The current package focuses on large scale problem and multiple hardwares. Hence, the goal is to use Matrix Free methods on GPU or a cluster to solve non linear PDE (for example).

**If you use this package for your work, please cite it!! Open source development strongly depends on this.**

## Installation 

To install it, please run

`] add https://github.com/rveltz/PseudoArcLengthContinuation.jl`

## Main features

- Matrix Free Newton solver with generic linear / eigen solver. Idem for the arc-length continuation
- Matrix Free Newton solver with deflation. It can be used for branch switching for example.
- Fold / Hopf bifurcation detection
- Fold / Hopf with MatrixFree / Sparse Jacobian continuation with Minimally Augmented formulation
- Periodic orbit computation and continuation using Simple Shooting (not very stable yet) or Finite Differences.
- Custom state means, can we use something else than `AbstractVector`


|Feature|Matrix Free|Custom state|
|---|---|---|
| Newton | Y | Y |
| Newton + Deflation| Y | Y |
| Continuation (Natural, Secant, Tangent) | Y | Y |
| Branching point detection | Y | Y |
| Fold detection | Y | Y |
| Hopf detection | Y | Y |
| Fold continuation | Y | N |
| Hopf continuation | Y | N |
| Periodic Orbit Newton | Y | N |
| Periodic Orbit continuation | Y | N |

## To do
- [ ] Improve Sparse Matrix creation of the Jacobian for the Periodic Orbit problem with Finite Differences
- [ ] Compute Hopf Normal Form
- [ ] Implement Preconditioner for the Matrix Free computation of Periodic Orbits based on Finite Differences
- [ ] Inplace implementation
- [ ] Provide a way to add constraints and combine functionals
- [ ] Improve `computeHopf` and `computeFold` to allow for general state (not `AbstractArray`). Also, the implementation allocates a new `struct` for each parameter.
- [ ] write continuation loop as an iterator
