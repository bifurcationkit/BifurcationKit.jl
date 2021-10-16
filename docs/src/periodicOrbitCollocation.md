# Periodic orbits based on orthogonal collocation

We have implemented a method where we compute `Ntst` time slices of a periodic orbit. This is done by the structure `PeriodicOrbitOCollProblem` for which the problem of finding periodic orbits is discretized using orthogonal collocation.

!!! danger "WIP"
    This is work in progress and only the docs are available for now. This warning will be removed when the functionality is available.

!!! warning "Large scale"
    The current implementation is not very optimized for large scale problems. This will be improved in the future.    

The general method is very well exposed in [^Dankowicz],[^Doedel]. We adopt the notations of [^Dankowicz] but use the implementation of [^Doedel] because it is more economical (less equations) when it forces the continuity of the solution.

We look for periodic orbits as solutions $(x(0), T)$ of

$$\dot x = T\cdot F(x),\ x(0)=x(1).$$

We consider a partition of the time domain

$$0=\tau_{1}<\cdots<\tau_{j}<\cdots<\tau_{N_{tst}+1}=1$$

where the points are referred to as **mesh points**. On each mesh interval $[\tau_j,\tau_{j+1}]$ for $j=1,\cdots,N_{tst}$, we define the affine transformation

$$\tau=\tau^{(j)}(\sigma):=\tau_{j}+\frac{(1+\sigma)}{2}\left(\tau_{j+1}-\tau_{j}\right), \sigma \in[-1,1].$$

The functions $x^{(j)}$ defined on $[-1,1]$ by $x^{(j)}(\sigma) \equiv x(\tau_j(\sigma))$ satisfies the following equation on $[-1,1]$:

$$\dot x^{(j)} = T\frac 	{\tau_{j+1}-\tau_j}{2}\cdot F(x^{(j)})\tag{$E_j$}$$

with the continuity equation $x^{(j+1)}(-1) = x^{(j)}(1)$.

We now aim at  solving $(E_j)$ by using an approximation with a polynomial of degree $m$. Following [^Dankowicz], we define a (uniform) partition:

$$-1=\sigma_{1}<\cdots<\sigma_{i}<\cdots<\sigma_{m+1}=1$$

and the associated $m+1$ Lagrange polynomials of degree $m$:

$$\mathcal{L}_{i}(\sigma):=\prod_{k=1, k \neq i}^{m+1} \frac{\sigma-\sigma_{k}}{\sigma_{i}-\sigma_{k}}, i=1, \ldots, m+1.$$

We then introduce the approximation $p_j$ of $x^{(j)}$:

$$\mathcal p_j(\sigma)\equiv \sum\limits_{k=1}^{m+1}\mathcal L_k(\sigma)x_{j,k}$$

and the problem to be solved at the **collocation nodes** $z_l$, $l=1,\cdots,m$:

$$\forall 1\leq l\leq m,\quad, 1\leq j\leq N_{tst},\quad \dot p_j(z_l) = \frac{\tau_{j+1}-\tau_j}{2}\cdot F(p_j(z_l))\tag{$E_j^2$}.$$

The **collocation nodes** $(z_l)$ are associated with a Gauss–Legendre quadrature.

In order to have a unique solution, we need to remove the phase freedom. This is done by imposing a *phase* condition.

## Encoding of the functional

The functional is encoded in the composite type [`PeriodicOrbitOCollProblem`](@ref). See the link for more information, in particular on how to access the underlying functional, its jacobian...

## Floquet multipliers computation

The algorithm is a simplified version of the procedure described in [^Fairgrieve]. It boils down to solving a large generalized eigenvalue problem. There is clearly room for improvements here.

The method allows, nevertheless, to detect bifurcations of periodic orbits. It seems to work reasonably well for the tutorials considered here.


## Computation with `newton`

We provide a simplified call to `newton` to locate the periodic orbits. Compared to the regular `newton` function, there is an additional option `linearalgo` to select one of the many ways to deal with the above linear problem. The default solver `linearalgo` is `:autodiffDense`.

The docs for this specific `newton` are located at [`newton`](@ref).

## Computation with `newton` and deflation

We also provide a simplified call to `newton` to locate the periodic orbit with a deflation operator.

## Continuation

We refer to [`continuation`](@ref) for more information regarding the arguments.

## References

[^Dankowicz]:> Dankowicz, Harry, and Frank Schilder. Recipes for Continuation. Computational Science and Engineering Series. Philadelphia: Society for Industrial and Applied Mathematics, 2013.

[^Doedel]:> Doedel, Eusebius, Herbert B. Keller, and Jean Pierre Kernevez. “NUMERICAL ANALYSIS AND CONTROL OF BIFURCATION PROBLEMS (II): BIFURCATION IN INFINITE DIMENSIONS.” International Journal of Bifurcation and Chaos 01, no. 04 (December 1991): 745–72.

[^Fairgrieve]:> Fairgrieve, Thomas F., and Allan D. Jepson. “O. K. Floquet Multipliers.” SIAM Journal on Numerical Analysis 28, no. 5 (October 1991): 1446–62. https://doi.org/10.1137/0728075.
