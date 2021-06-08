# [Deflated problems](@id Deflated-problems)

!!! unknown "References"
    P. E. Farrell, A. Birkisson, and S. W. Funke. **Deflation techniques for finding distinct solutions of nonlinear partial differential equations**. SIAM J. Sci. Comput., 2015.,

Assume you want to solve $F(x)=0$ with a Newton algorithm but you want to avoid the algorithm to return some already known solutions $x_i,\ i=1\cdots n$. 

The idea proposed in the paper quoted above is to penalize these solutions by looking for the zeros of the function $G(x):={F(x)}{M(x)}$ where

$$M(x) = \prod_{i=1}^n\left(\|x - x_i\|^{-p} + \alpha\right)$$

and $\alpha>0$. Obviously $F$ and $G$ have the same zeros away from the $x_i$s but the factor $M$ penalizes the residual of the Newton iterations of $G$, effectively producing zeros of $F$ different from $x_i$.

## Encoding of the functional

A composite type [`DeflationOperator`](@ref) implements this functional. Given a deflation operator `M = DeflationOperator(p, dot, α, xis)`, you can build a deflated functional `pb = DeflatedProblem(F, J, M)` which you can use to access the values of $G$ by doing `pb(x)`. A Matrix-Free / Sparse linear solver is implemented which works on the GPU.

> the `dot` argument in `DeflationOperator` lets you specify a dot product from which the norm is derived in the expression of $M$.

See example [Snaking computed with deflation](@ref).

Note that you can add new solution `x0` to `M` by doing `push!(M, x0)`. Also `M[i]` returns `xi`.

## Computation with `newton`

Most newton functions can be used with a deflated problem, see for example [Snaking computed with deflation](@ref). The idea is to pass the deflation operator `M`. For example, we have the following overloaded method, which works on GPUs: 

```
newton(F, J, x0, p0, options::NewtonPar, defOp::DeflationOperator, linsolver = DeflatedLinearSolver(); kwargs...)
```

If you pass a linear solver other than the default one `::DeflatedLinearSolver`, a Matrix-Free is used in place of the dedicated solver `DeflatedLinearSolver` which is akin to a Bordering method.

We refer to [`newton`](@ref) for more information about the arguments.

!!! tip "Tip"
    You can use this method for periodic orbits as well by passing the deflation operator `M` to the newton method
