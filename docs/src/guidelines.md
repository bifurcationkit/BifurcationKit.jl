# Guidelines

The goal of this package is to find solutions $x$ to nonlinear equations 

$$F(x,p) = 0 \quad\tag{E}$$

in large dimensions as function of a *real* parameter $p$. We want to be able to do so on GPU, distributed systems...

The core of the package is built around a Newton-Krylov solver (see [`newton`](@ref)) which allows to solve equations of the form $F(x)=0$, or a solution $x_0$ to (E) for a particular $p_0$.

Once such a solution $(x_0,p_0)$ is known, we can continue it by building a 1d curve of solutions $\gamma = (x(s),p(s))_{s\in I}$ passing through $(x_0,p_0)$ (see [`continuation`](@ref)).

> In practice, you don't need to know exactly $(x_0,p_0)$ to compute $\gamma$. Indeed, [`continuation`](@ref) will call [`newton`](@ref) to refine any initial guess that you will pass.

## Bifurcation analysis of Equilibria

We can detect if the curve of solutions $\gamma$ crosses another curve of solutions $\gamma^{bif}$ *without knowing* $\gamma^{bif}$! The intersection point $(x^b,p^b)\in\gamma$ is called a bifurcation point and is such that $\partial_xF(x^b,p^b)$ is non invertible. When calling [`continuation`](@ref), `γ, _ = continuation(...)` with the option `detectBifurcation > 0` inside [`ContinuationPar`](@ref), the bifurcation points are automatically detected and stored in `γ.bifpoints`.

!!! warning "Eigenvalues"
    The rightmost eigenvalues are computed by default to detect bifurcations. Hence, the number of eigenvalues with positive real parts must be finite (*e.g.* small). This might require to consider $-F(x,p)=0$ instead of (E).

### Branch switching 

In the simple cases, *e.g.* when $dim\ker \partial_xF(x^b,p^b) = 1$, we can compute automatically the **bifurcated branch** $\gamma^{bif}$ by calling [`continuation`](@ref) and passing $\gamma$. This is explained in [Branch switching from simple branch point to equilibria](@ref). Recursively, we can compute the curves of solutions which are connected to $(x_0,p_0)$, this is called a **bifurcation diagram**. This bifurcation diagram can be automatically computed using the function [`bifurcationdiagram`](@ref) with minimum input from the user. More information is provided in [Automatic Bifurcation diagram computation](@ref) and examples of use are [Swift-Hohenberg equation 1d](@ref) and [Automatic diagram of 2d Bratu–Gelfand problem (Intermediate)](@ref).

When $d\equiv dim\ker \partial_xF(x^b,p^b) > 1$, there is no automatic method to perform branch switching. Nevertheless, we can reduce (E) to a system of $d$ dimensional multivariate polynomial equations in $d$ unknowns whose solutions gives the local topology of branches in the neighborhood of the bifurcation point $(x^b,p^b)$. We can then use the solutions of this **reduced equation** as initial guesses and call again [`continuation`](@ref) to compute the bifurcated branches. This is explained in [Branch switching from non simple branch point to equilibria](@ref) and an example of shown in [A generalized Bratu–Gelfand problem in two dimensions](@ref).	
> In the case $d=1$, the reduced equation can be further simplified into a **normal form**. This is also automatically computed by the package.


## Bifurcation analysis of Cauchy problems

The goal of this section is to study the dynamics of Cauchy problems

$$\frac{d}{dt}x - F(x,p) = 0 \quad\tag{C}$$

The equilibria are time independent solutions of (C) hence solving (E). The previous section can be applied to compute curves of equilibria. However, we can do more. By discretizing time, we can recast (C) in the general form (E) and look for time dependent solutions as well. 

We can detect the existence of periodic solutions close to $\gamma$. This is done automatically and those bifurcation points are stored in `γ.bifpoint` as well with the name of **Hopf bifurcation points**.  

### Branch switching at Hopf points

We will not review the bifurcation of equilibria (see above). Therefore, we focus on computing the branch of periodic solutions branching of a Hopf point. This is done automatically by calling again [`continuation`](@ref), passing $\gamma$ and choosing a time discretization algorithm (see [Periodic orbits computation](@ref)). Some details about this branch switching is given in [Branch switching from Hopf point to periodic orbits](@ref).

### Branch switching at bifurcation points of periodic orbits

Let us consider the case where a branch of periodic orbits $\gamma^{po}$ have been computed (see for example previous section) and several bifurcation points have been detected (branch point, period doubling and Neimark Sacker). Can we compute bifurcated branches from $\gamma^{po}$? Automatically?

We do not provide an *automatic* branch switching for those points for all methods (Shooting, Finite differences). However, for branch points of periodic orbits, you can call [`continuation`](@ref) by passing $\gamma^{po}$ and some simple arguments (amplitude of the periodic orbits) to perform branch switching in a semi-automatic way. For the case of [Periodic orbits based on finite differences](@ref), see [Branch switching from Branch point of curve of periodic orbits](@ref).

!!! tip "Manual Branch switching"
    You can perform **manual** branch switching by computing the nearby solutions close to a bifurcation point using a deflated newton (see [Deflated problems](@ref)), which provides a way to compute solutions other than a set of already known solutions.  You can then use these solutions to compute branches by calling `continuation`. Many, if not all tutorials give example of doing so like [A generalized Bratu–Gelfand problem in two dimensions](@ref) or [Brusselator 1d (automatic)](@ref).