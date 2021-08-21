# Continuation methods: introduction

The goal of these methods[^Kuz],[^Govaerts],[^Rabinowitz],[^Mei],[^Keller] is to find solutions $x\in\mathbb R^n$ to nonlinear equations

$$\mathbb R^n\ni F(x,p) = 0 \quad\tag{E}$$

as function of a real parameter $p$. Given a known solution $(x_0,p_0)$, we can continue it by computing a 1d curve of solutions $\gamma = (x(s),p(s))_{s\in I}$ passing through $(x_0,p_0)$.

For the sequel, it is convenient to use the following formalism [^Kuz]

1. prediction of the next point
2. correction
3. step size control.


## Natural continuation

> More information is available at [Predictors - Correctors](@ref)

We just use this simple continuation method to exemplify the  formalism.
Knowing $(x_0, p_0)$, we form the predictor $(x_0, p_0+ds)$ for some $ds$ and use it as a guess using a Newton corrector applied to $x\to F(x, p_0+ds)=0$. The corrector is thus the newton algorithm.

This continuation method is set by the option `tangentAlgo = NaturalPred()` in `continuation`.

## Linear Algebra

Let us discuss here more about the norm and dot product. First, the option `normC` [`continuation`](@ref) specifies norm that is used to evaluate the residual in the following way: $max(normC(F(x,p)), |N(x,p)|)<tol$. It is thus used as a stopping criterion for a corrector. The dot product (resp. norm) used in $N$ and in the (iterative) linear solvers is `LinearAlgebra.dot` (resp. `LinearAlgebra.norm`). It can be changed by importing these functions and redefining it. Not that by default, the $\mathcal L^2$ norm is used. These details are important because of the constraint $N$ which incorporates the factor `length`. For some custom composite type implementing a Vector space, the dot product could already incorporates the `length` factor in which case you should either redefine the dot product or change $\theta$.

## Step size control

As explained above, each time the corrector phased failed, the step size ``ds`` is halved. This has the disadvantage of having lost Newton iterations (which costs time) and impose small steps (which can be slow as well). To prevent this, the step size is controlled internally with the idea of having a constant number of Newton iterations per point. This is in part controlled by the aggressiveness factor `a` in `ContinuationPar`. Further tuning is performed by using `doArcLengthScaling=true` in `ContinuationPar`. This adjusts internally $\theta$ so that the relative contributions of ``x`` and ``p`` are balanced in the constraint $N$.


### References

[^Kuz]:> Kuznetsov, Elements of Applied Bifurcation Theory.

[^Govaerts]:> Govaerts, Numerical Methods for Bifurcations of Dynamical Equilibria; Allgower and Georg, Numerical Continuation Methods

[^Rabinowitz]:> Rabinowitz, Applications of Bifurcation Theory; Dankowicz and Schilder, Recipes for Continuation

[^Mei]:> Mei, Numerical Bifurcation Analysis for Reaction-Diffusion Equations

[^Keller]:> Keller, Lectures on Numerical Methods in Bifurcation Problems
