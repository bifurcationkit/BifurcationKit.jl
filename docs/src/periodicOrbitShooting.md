# Periodic orbits based on the shooting method

A set of shooting algorithms is provided which are called either *Simple Shooting (SS)* if a single shooting is used and *Multiple Shooting (MS)* otherwise. 

!!! unknown "References"
    For the exposition, we follow the PhD thesis **Numerical Bifurcation Analysis of Periodic Solutions of Partial Differential Equations**, *Lust, Kurt*, 1997. 

We aim at finding periodic orbits for the Cauchy problem 

$$\tag{1} \frac{d x}{d t}=f(x)$$ 

and we write $\phi^t(x_0)$ the associated flow (or semigroup of solutions).

!!! tip "Tip about convenience functions"
    For convenience, we provide some functions `plotPeriodicShooting` for plotting, `getAmplitude` (resp. `getMaximum`) for getting the amplitude (resp. maximum) of the solution encoded by a shooting problem. See the tutorials for examples of use.

## Standard Shooting
### Simple shooting
A periodic orbit is found when we have a couple $(x, T)$ such that $\phi^T(x) = x$ and the trajectory is non constant. Therefore, we want to solve the equations $G(x,T)=0$ given by

$$\tag{SS}
\begin{array}{l}{\phi^T(x)-x=0} \\ {s(x,T)=0}\end{array}.$$

The section $s(x,T)=0$ is a phase condition to remove the indeterminacy of the point on the limit cycle.

### Multiple shooting
This case is similar to the previous one but more sections are used. To this end, we partition the unit interval with $M+1$ points
$$0=s_{0}<s_{1}<\cdots<s_{m-1}<s_{m}=1$$ and consider the equations $G(x_1,\cdots,x_M,T)=0$

$$\begin{aligned} 
\phi^{\delta s_1T}(x_{1})-x_{2} &=0 \\ 
\phi^{\delta s_2T}(x_{2})-x_{3} &=0 \\ & \vdots \\ 
\phi^{\delta s_{m-1}T}(x_{m-1})-x_{m} &=0 \\ 
\phi^{\delta s_mT}(x_{m})-x_{1} &=0 \\ s(x_{1}, T) &=0. \end{aligned}$$

where $\delta s_i:=s_{i+1}-s_i$. The Jacobian of the system of equations *w.r.t.* $(x,T)$ is given by 

$$\mathcal{J}=\left(\begin{array}{cc}{\mathcal J_c} & {\partial_TG} \\ {\star} & {d}\end{array}\right)$$

where the cyclic matrix $\mathcal J_c$ is

$$\mathcal J_c := 
\left(\begin{array}{ccccc}
{M_{1}} & {-I} & {} & {} \\ 
{} & {M_{2}} & {-I} & {}\\ 
{} & {} & {\ddots} & {-I}\\ 
{-I} & {} & {} & {M_{m}}\\ 
\end{array}\right)$$

and $M_i=\partial_x\phi^{\delta s_i T}(x_i)$.

### Encoding of the functional

The functional is encoded in the composite type [`ShootingProblem`](@ref). In particular, the user can pass its own time stepper or one can use the different ODE solvers in  [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl) which makes it very easy to choose a solver tailored for the a specific problem. See the link [`ShootingProblem`](@ref) for more information ;  for example on how to access the underlying functional, its jacobian...

## Poincaré shooting

> The algorithm is based on the one described in **Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.**, Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó (2004) and **Matrix-Free Continuation of Limit Cycles for Bifurcation Analysis of Large Thermoacoustic Systems.** Waugh, Iain, Simon Illingworth, and Matthew Juniper (2013). 


We look for periodic orbits solutions of (1) using the hyperplanes $\Sigma_i=\{x\ / \ n_i\cdot(x-x^c_{i})=0\}$ for $i=1,\cdots,M$, centered on $x^c_i$, which intersect transversally an initial periodic orbit guess. We write $\Pi_i:\Sigma_i\to\Sigma_{mod(i+1,M)}$, the Poincaré return map to $\Sigma_{mod(i+1,M)}$. The main idea of the algorithm is to use the fact that the problem is $(N-1)\cdot M$ dimensional if $x_i\in\mathbb R^N$ because each $x_i$ lives in $\Sigma_i$. Hence, one has to constrain the unknowns to these hyperplanes otherwise the Newton algorithm does not converge well.

To this end, we introduce the projection operator $R_i:\mathbb R^N\to \mathbb R^{N-1}$ such that 

$$R_{i}\left(x_{1}, x_{2}, \ldots, x_{k_i-1}, x_{k_i}, x_{k_i+1}, \ldots, x_{N}\right)=\left(x_{1}, x_{2}, \ldots, x_{k_i-1}, x_{k_i+1}, \ldots, x_{N}\right)$$

where $k_i=argmax_p |n_{i,p}|$. The inverse operator is defined as (where $\bar x:=R_i(x)$)

$$E_{i}(\bar x) := E_{i}\left(x_{1}, x_{2}, \ldots, x_{k_i-1}, x_{k_i+1}, \ldots, x_{N}\right)=
\left(x_{1}, x_{2}, \ldots, x_{k_i-1}, x^c_{i,k_i}-\frac{\bar{n}_i \cdot\left(\overline{x}-\overline{x}^c_{i}\right)}{n_{i,k_i}}, x_{k_i+1}, \ldots, x_{N}\right).$$ 

We note that $R_i\circ E_i = I_{\mathbb R^{N-1}}$ and $E_i\circ R_i = I_{\mathbb R^{N}}$.

We then look for solutions of the following problem:

$$\begin{aligned} 
\bar x_1 - R_M\Pi_M(E_M(\bar x_M))&=0 \\ 
\bar x_2 - R_1\Pi_1(E_i(\bar x_1))&=0 \\ & \vdots \\ 
\bar x_M - R_{M-1}\Pi_{M-1}(E_{M-1}(\bar x_{M-1}))&=0. 
\end{aligned}$$



### Encoding of the functional

The functional is encoded in the composite type [`PoincareShootingProblem`](@ref). In particular, the user can pass their own time stepper or he can use the different ODE solvers in  [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl) which makes it very easy to choose a tailored solver: the partial Poincaré return maps are implemented using **callbacks**. See the link [`PoincareShootingProblem`](@ref) for more information, in particular on how to access the underlying functional, its jacobian...

## Floquet multipliers computation


### Standard shooting
The Floquet multipliers are computed as the eigenvalues of $M_M\cdots M_1$.

> Unlike the case with [Finite differences](https://rveltz.github.io/BifurcationKit.jl/dev/periodicOrbitTrapeze/), the matrices $M_i$ are not sparse.

### Poincaré shooting
The (non trivial) Floquet exponents are eigenvalues of the Poincare return map $\Pi:\Sigma_1\to\Sigma_1$. We have $\Pi = \Pi_M\circ\Pi_{M-1}\circ\cdots\circ\Pi_2\circ\Pi_1$. Its differential is thus

$$d\Pi(x)\cdot h = d\Pi_M(x_{M})d\Pi_{M-1}(x_{M-1})\cdots d\Pi_1(x_1)\cdot h$$

### Numerical method

A **not very precise** algorithm for computing the Floquet multipliers is provided. The method, dubbed Quick and Dirty (QaD), is not numerically very precise for large / small Floquet exponents. 

It amounts to computing the eigenvalues of $M_M\cdots M_1$ (resp. $d\Pi$) for the Standard (resp. Poinncaré) Shooting.


The method allows, nevertheless, to detect bifurcations of periodic orbits. It seems to work reasonably well for the tutorials considered here. For more information, have a look at [`FloquetQaD`](@ref).

!!! note "Algorithm"
    A more precise algorithm, based on the periodic Schur decomposition will be implemented in the future.

## Computation with `newton`

We provide a simplified call to `newton` to locate the periodic orbit. Have a look at the tutorial [Continuation of periodic orbits (Standard Shooting)](@ref) for a simple example on how to use the above methods. 

The docs for this specific `newton` are located at [`newton`](@ref).

## Computation with `newton` and deflation

We also provide a simplified call to `newton` to locate the periodic orbit with a deflation operator:

```
newton(prob:: AbstractShootingProblem, orbitguess, par0, options::NewtonPar; kwargs...)
```

and

```
newton(prob:: AbstractShootingProblem, orbitguess, par0, options::NewtonPar, defOp::DeflationOperator; kwargs...)```

## Continuation

Have a look at the [Continuation of periodic orbits (Standard Shooting)](@ref) example for the Brusselator.

The docs for this specific `newton` are located at [`continuation`](@ref).
