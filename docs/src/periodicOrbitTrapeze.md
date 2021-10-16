# Periodic orbits based on trapezoidal rule

We have implemented a method where we compute `M` slices of a periodic orbit. This is done by the structure `PeriodicOrbitTrapProblem` for which the problem of finding periodic orbits is discretized using Finite Differences based on a trapezoidal rule.

The general method is very well exposed in [^Uecker],[^Lust] We adopt the notations of the first reference.

We look for periodic orbits as solutions $(x(0),T)$ of

$$M_a\dot x = T\cdot F(x),\ x(0)=x(1)$$

where $M_a$ is a mass matrix (default is the identity one).

In order to have a unique solution, we need to remove the phase freedom. This is done by imposing a *phase* condition $\sum_i\langle x_{i} - x_{\pi,i}, \phi_{i}\rangle = 0$ for some $x_\pi,\phi$ which are chosen (wisely).

By discretizing the above problem, we obtain

$$\begin{array}{l}
		0= M_a\left(x_{j}-x_{j-1}\right)-\frac{h}{2} \left(F\left(x_{j}\right)+F\left(x_{j-1}\right)\right)\equiv G_j(x),\quad j=1,\cdots,m-1 \\
0= x_m-x_1 \equiv G_m(x) \\
0= \sum_i\langle x_{i} - x_{\pi,i}, \phi_{i}\rangle=0
\end{array}$$
where $x_0=x_m$ and $h=T/m$. The Jacobian of the system of equations *w.r.t.* $(x_0,T)$ is given by

$$\mathcal{J}=\left(\begin{array}{cc}{A_1} & {\partial_TG} \\ {\star} & {d}\end{array}\right)$$

where

$$A_{\gamma}:=\left(\begin{array}{ccccccc}
{M_{1}} & {0} & {0} & {0} & {\cdots} & {-H_{1}} & {0} \\
{-H_{2}} & {M_{2}} & {0} & {0} & {\cdots} & {0} & {0} \\
{0} & {-H_{3}} & {M_{3}} & {0} & {\cdots} & {0} & {0} \\
{\vdots} & {\cdots} & {\ddots} & {\ddots} & {\ddots} & {\vdots} & {\vdots} \\
{0} & {\cdots} & {\cdots} & {\ddots} & {\ddots} & {0} & {0} \\
{0} & {\cdots} & {\cdots} & {0} & {-H_{m-1}} & {M_{m-1}} & {0} \\
{-\gamma I} & {0} & {\cdots} & {\cdots} & {\cdots} & {0} & {I}
\end{array}\right)$$

with $M_i := M_a-	\frac h2dF(x_i)$ and $H_i := M_a+\frac h2dF(x_{i-1})$.

We solve the linear equation $\mathcal J\cdot sol = rhs$ with a bordering strategy (*i.e.* the linear solver is a subtype of `<: AbstractBorderedLinearSolver`) which in turn requires to solve $A_\gamma z=b$ where $z=(x,x_m)$. We also solve this equation with a bordering strategy but this time, it can be simplified as follows. If we write $b=(f,g)$, one gets $J_c x=f$ and $x_m=g+\gamma x_1$ where $x_1$ is the first time slice of $x$ and $J_c$ is the following **cyclic matrix**:

$$J_c:=\left(\begin{array}{ccccccc}
{M_{1}} & {0} & {0} & {0} & {\cdots} & {-H_{1}} \\
{-H_{2}} & {M_{2}} & {0} & {0} & {\cdots} & {0} \\
{0} & {-H_{3}} & {M_{3}} & {0} & {\cdots} & {0} \\
{\vdots} & {\cdots} & {\ddots} & {\ddots} & {\ddots} & {\vdots} \\
{0} & {\cdots} & {\cdots} & {\ddots} & {\ddots} & {0} \\
{0} & {\cdots} & {\cdots} & {0} & {-H_{m-1}} & {M_{m-1}} \\
\end{array}\right)$$

Our code thus provides methods to invert $J_c$ and $A_\gamma$ using a sparse solver or a Matrix-Free solver. A preconditioner can be used.

## Encoding of the functional

The functional is encoded in the composite type [`PeriodicOrbitTrapProblem`](@ref). See the link for more information, in particular on how to access the underlying functional, its jacobian and other matrices related to it like $A_\gamma, J_c$...

## Preconditioning

We strongly advise you to use a preconditioner to deal with the above linear problem. See [2d Ginzburg-Landau equation (finite differences)](@ref) for an example.


## Floquet multipliers computation

A **not very precise** algorithm for computing the Floquet multipliers is provided. The method, dubbed Quick and Dirty (QaD), is not numerically very precise for large / small Floquet exponents.

It amounts to computing the eigenvalues of 

$$\mathcal{M}=M_{1}^{-1} H_{1} M_{m-1}^{-1} H_{m-1} \cdots M_{2}^{-1} H_{2}.$$

The method allows, nevertheless, to detect bifurcations of periodic orbits. It seems to work reasonably well for the tutorials considered here. For more information, have a look at [`FloquetQaD`](@ref).


!!! note "Algorithm"
    A more precise algorithm, based on the periodic Schur decomposition will be implemented in the future.


## Computation with `newton`

We provide a simplified call to `newton` to locate the periodic orbits. Compared to the regular `newton` function, there is an additional option `linearalgo` to select one of the many ways to deal with the above linear problem. The default solver `linearalgo` is `:BorderedLU`.

Have a look at the [Continuation of periodic orbits (Finite differences)](@ref) example for the Brusselator for a basic example and at [2d Ginzburg-Landau equation (finite differences)](@ref) for a more advanced one.

The docs for this specific `newton` are located at [`newton`](@ref).

## Computation with `newton` and deflation

We also provide a simplified call to `newton` to locate the periodic orbit with a deflation operator.

```@docs
newton(probPO::PeriodicOrbitTrapProblem, orbitguess, options::NewtonPar, defOp::DeflationOperator, linearPO = :BorderedLU; kwargs...)
```


## Continuation

Have a look at the [Continuation of periodic orbits (Finite differences)](@ref) example for the Brusselator. We refer to [`continuation`](@ref) for more information regarding the arguments.

## References

[^Uecker]:> Uecker, Hannes. Hopf Bifurcation and Time Periodic Orbits with Pde2path – Algorithms and Applications. Communications in Computational Physics 25, no. 3 (2019) 

[^Lust]:> Lust, Kurt, Numerical Bifurcation Analysis of Periodic Solutions of Partial Differential Equations, PhD thesis, 1997. 