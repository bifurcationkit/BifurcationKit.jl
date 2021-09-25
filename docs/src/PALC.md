# Pseudo arclength continuation

This is one of the various continuation methods implemented in `BifurcationKit.jl`. It is set by the option `tangentAlgo = BorderedPred()` in [`continuation`](@ref).

For solving 

$$\mathbb R^n\ni F(x,p) = 0 \quad\tag{E}$$

using a Newton algorithm, we miss an equation. The simplest way is to select an hyperplane in the space $\mathbb R^n\times \mathbb R$ passing through $(x_0,p_0)$:

$$N(x, p) = \frac{\theta}{length(x)} \langle x - x_0, dx_0\rangle + (1 - \theta)\cdot(p - p_0)\cdot dp_0 - ds = 0$$

with $\theta\in[0,1]$ and where $ds$ is the pseudo arclength (see [^Keller]).

!!! warning "Parameter `theta`"
    The parameter `theta` in the struct `ContinuationPar` is very important. It should be tuned for the continuation to work properly especially in the case of large problems where the ``\langle x - x_0, dx_0\rangle`` component in the constraint might be favored too much. Also, large `theta`s favour `p` as the corresponding term in the constraint ``N`` involves the term ``1-\theta``.
    
![](PALC.png)
    

## Predictor

The possible predictors are listed in [Predictors - Correctors](@ref).

## Corrector

The corrector is the newton algorithm for finding the roots $(x,p)$ of

$$\begin{bmatrix} F(x,p) \\	N(x,p)\end{bmatrix} = 0\tag{PALC}$$

## Linear Algebra

Let us discuss more about the norm and dot product. First, the option `normC` [`continuation`](@ref) specifies the norm used to evaluate the residual in the following way: $max(normC(F(x,p)), |N(x,p)|)<tol$. It is thus used as a stopping criterion for the corrector. The dot product (resp. norm) used in $N$ and in the (iterative) linear solvers is `LinearAlgebra.dot` (resp. `LinearAlgebra.norm`). It can be changed by importing these functions and redefining it. Note that by default, the $\mathcal L^2$ norm is used. These details are important because of the constraint $N$ which incorporates the factor `length`. For some custom composite type implementing a Vector space, the dot product could already incorporates the `length` factor in which case you should either redefine the dot product or change $\theta$.

The linear solver for the linear problem associated to (PALC) is set by the option `linearAlgo` in [`continuation`](@ref): it is one of [Bordered linear solvers (BLS)](@ref).



## Step size control

Each time the corrector fails, the step size ``ds`` is halved. This has the disadvantage of having lost Newton iterations (which costs time) and imposing small steps (which can be slow as well). To prevent this, the step size is controlled internally with the idea of having a constant number of Newton iterations per point. This is in part controlled by the aggressiveness factor `a` in `ContinuationPar`. 


### References

[^Keller]:> Keller, Herbert B. Lectures on Numerical Methods in Bifurcation Problems. Springer, 1988
