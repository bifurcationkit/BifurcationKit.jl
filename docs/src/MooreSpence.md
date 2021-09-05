# Moore Spence continuation

!!! warning "WIP"
    This is work in progress. The available options are limited.

This is one of the various continuation methods implemented in `BifurcationKit.jl`. it is set by the option `tangentAlgo = MooreSpence()` in [`continuation`](@ref).

For solving 

$$\mathbb R^n\ni F(x,p) = 0 \quad\tag{E}$$

using a Newton algorithm, we miss an equation. Hence, we proceed as follows [^Meijer]. Starting from a predictor $(x_1,p_1)$, we look for the solution to (E) that is closest to $(x_1,p_1)$. Hence, we optimise

$$\min_{(x,p)} \{ \|(x,p)-(x_1,p_1)\| \text{ such that } F(x,p)=0\} \tag{MS}$$  

It can be interpreted as a PALC in which the hyperplane is adapted at every step.  

## Predictor

The possible predictors are listed in [Predictors - Correctors](@ref).

## Corrector

The corrector is the Gauss Newton algorithm applied to (MS).

## Linear Algebra

Let us discuss more about the norm and dot product. First, the option `normC` [`continuation`](@ref) specifies the norm used to evaluate the distance in (MS). The dot product (resp. norm) used in the (iterative) linear solvers is `LinearAlgebra.dot` (resp. `LinearAlgebra.norm`). It can be changed by importing these functions and redefining it. Note that by default, the ``L^2`` norm is used.

The linear solver for the linear problem associated to (MS) is set by the option `linearAlgo` in [`continuation`](@ref): it is one of [Bordered linear solvers (BLS)](@ref).


## Step size control

Each time the corrector fails, the step size ``ds`` is halved. This has the disadvantage of having lost Newton iterations (which costs time) and imposing small steps (which can be slow as well). To prevent this, the step size is controlled internally with the idea of having a constant number of Newton iterations per point. This is in part controlled by the aggressiveness factor `a` in `ContinuationPar`. 


### References

[^Meijer]:> Meijer, Dercole, and Oldeman, “Numerical Bifurcation Analysis.”
