# Bordered linear solvers (BLS)

> The bordered linear solvers must be subtypes of `AbstractBorderedLinearSolver <: AbstractLinearSolver`. 

The methods provided here solve bordered linear equations. More precisely, one is interested in the solution $u$ to $J\cdot u = v$ where

$$\tag E J=\left(\begin{array}{ll}
{A} & {b} \\
{c} & {d}
\end{array}\right) \text { and } v=\left(\begin{array}{l}
{v_1} \\
{v_2}
\end{array}\right)$$

Such linear solver `bdlsolve` will be called like `sol, success, itnumber = bdlsolve(A, b, c, d, v1, v2)` throughout the package.


## Full matrix `MatrixBLS`
This easiest way to solve $(E)$ is by forming the matrix $J$. In case it is sparse, it should be relatively efficient. You can create such bordered linear solver using `bls = MatrixBLS(ls)` where `ls::AbstractLinearSolver` is a linear solver (which defaults to `\`) used to solve invert $J$.

## Bordering method `BorderingBLS `

The general solution to $(E)$ when $A$ is non singular is $x_1=A^{-1}v_1, x_2=A^{-1}b$, $u_2 = \frac{1}{d - (c,x_2)}(v_2 - (c,x_1))$ and $u_1=x_1-u_2x_2$. This is the default method used in the package. It is very efficient for large scale problems because it is entirely Matrix-Free and one can use preconditioners. You can create such bordered linear solver using `bls = BorderingBLS(ls)` where `ls::AbstractLinearSolver` is a linear solver which defaults to `\`. The intermediate solutions $x_1=A^{-1}v_1, x_2=A^{-1}b$ are formed using `ls`.

> 1. Using such method with `ls` being a GMRES method is the main way to solve (E) in this package.
> 2. In the case where `ls = DefaultLS()`, the factorisation of `A` is cached so the second linear solve is very fast 

## Full Matrix-Free `MatrixFreeBLS`

In cases where $A$ is singular but $J$ is not, the bordering method may fail. It can thus be advantageous to form the Matrix-Free version of $J$ and call a generic linear solver to find the solution to $(E)$. You can create such bordered linear solver using `bls = MatrixFreeBLS(ls)` where `ls::AbstractLinearSolver` is a (Matrix Free) linear solver which is used to invert `J`.

> For now, this linear solver only works with `AbstractArray`

