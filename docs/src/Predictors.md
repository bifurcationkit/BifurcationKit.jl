# Predictors - Correctors

As explained in the docs of [continuation](@docs), the continuation method works with the following pattern (see [^Allgower1990]):

1. compute **tangent**
2. call **predictor** (based on tangent, mostly)
3. call **corrector**


[^Allgower1990]: > Allgower and Georg, Numerical Continuation Methods, 1990

There are several couple predictor-tangent/corrector which can be used in **BifurcationKit.jl** as we now explain. The tangent computation is formally included in the predictor whereas it is a distinct function in the code.

## 1) Natural, zeroth order predictor

This is the dumbest predictor based on the formula $(x_1,p_1) = (x_0, p_0 + ds)$ with Newton corrector ; it fails at Turning points. This is set by the option `tangentAlgo = NaturalPred()` in [continuation](@docs). For matrix based jacobian, it is not faster than the pseudo-arclength predictor because the factorisation of the jacobian is catched. For Matrix-free methods, this predictor can be faster than the following ones until it hits a Turning point.

## 2) First order predictor

This one is based on a computation of the tangent $\tau = (dx,dp)$ to the curve of solutions by $(x_1,p_1) = (x_0,p_0) + ds\cdot \tau$. This predictor passes Turning points when used with PALC Newton corrector.
**BifurcationKit.jl** provides two ways to compute the tangent $(dx_1, dp_1)$. 

### Secant predictor
The first one is called secant and is parametrized by the option `tangentAlgo = SecantPred()`. It is computed by $(dx_1, dp_1) = (z_1, p_1) - (z_0, p_0)$ and normalized by the norm $\|(x, p)\|^2_\theta = \frac{\theta}{length(x)} \langle x,x\rangle + (1 - \theta)\cdot p^2$. 

### Bordered predictor
Another method is to compute $(dx_1, dp_1)$ by solving solving the bordered linear system $$\begin{bmatrix} F_x & F_p	\\ \frac{\theta}{length(x)}dx_0 & (1-\theta)dp_0\end{bmatrix}\begin{bmatrix}dx_1 \\  dp_1\end{bmatrix} =\begin{bmatrix}0 \\ 1\end{bmatrix}$$ it is set by the option `tangentAlgo = BorderedPred()`.

## 3) Polynomial predictor

The polynomial predictor is based on a fit of an $n$th-order polynomial $P$ function to the last $k$ solution vectors, where $n < k$ using least square regression. The arclength $s$ is used for the polynomial which then fits the solution $(x_i,p_i,s_i)$ as $P(s_i)\approx (x_i,p_i)$. To keep $s$ in suitable range (see [^Waugh]), we rescale it as $s\to \frac{s-\bar s}{\sigma}$ where $\sigma$ is the standard deviation of the $s_i$.

This predictor is parametrized by `tangentAlgo = PolynomialPred(n, k, v0, algo = SecantPred() )` where `algo` is the predictor used for the first $k$ solutions before the polynomial predictor is operational and `v0` is an example of guess.

[^Waugh]: > Waugh, Illingworth, and Juniper, “Matrix-Free Continuation of Limit Cycles for Bifurcation Analysis of Large Thermoacoustic Systems.”

## 4) Multiple predictor (aka `pmcont` in `pde2path`)

The predictor is especially designed [^Uecker2014] to avoid spurious branch switching and pass singular points especially in PDE where branch point density can be quite high. It is based on the idea that one uses many predictors
$$(x_i,p_i) = (x_0,p_0) + i\cdot ds\cdot \tau,\ i\leq nb$$
and use a PALC Newton with the following twist. The criterion is that in each Newton step, the residual has to decrease by a factor $0<\alpha<1$ 
$$\| F(u_n,p_n)\|\leq \alpha \| F(u_{n-1},p_{n-1}) \|$$
otherwise the corrector fails. The solution that is returned is the one for the highest $i$. We refer to [^Uecker2014] for an exposition of the step size adaption strategy.

This predictor is parametrized by `tangentAlgo = MultiplePred(α, nb, τ, algo = SecantPred() )` where `τ` is an initial tangent vector (used to set the types) and `algo` is a predictor.

[^Uecker2014]: > 1.Uecker, H. pde2path - A Matlab Package for Continuation and Bifurcation in 2D Elliptic Systems. NMTMA 7, 58–106 (2014).
