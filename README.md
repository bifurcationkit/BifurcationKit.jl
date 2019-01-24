# PseudoArcLengthContinuation.jl

![Build Status](https://travis-ci.com/rveltz/PseudoArcLengthContinuation.svg?token=JVdfPsGga24TLMZxCLqE&branch=master)
[![Coverage Status](https://coveralls.io/repos/github/rveltz/PseudoArcLengthContinuation.jl/badge.svg?branch=master)](https://coveralls.io/github/rveltz/PseudoArcLengthContinuation.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://rveltz.github.io/PseudoArcLengthContinuation.jl/latest) 

This package aims at solving equations $F(u,\lambda)=0$ where $\lambda \in\mathbb R$ starting from an initial guess $(u_0,\lambda_0)$. It relies on the pseudo arclength continuation algorithm which provides a *predictor* $(u_1,\lambda_1)$ from $(u_0,\lambda_0)$. A Newton method is then used to correct this predictor.

The current package focuses on large scale problem and multiple hardwares. Hence, the goal is to use Matrix Free methods on GPU or a cluster to solve non linear PDE (for example).

**If you use this package for your work, please cite it!! Open source development will die otherwise.**

## Installation 

To install it, please run

`] add PseudoArcLengthContinuation.jl`

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
- Improve Sparse Matrix creation of the Jacobian for the Periodic Orbit problem with Finite Differences
- Compute Hopf Normal Form
- Implement Preconditioner for the Matrix Free computation of Periodic Orbits based on Finite Differences