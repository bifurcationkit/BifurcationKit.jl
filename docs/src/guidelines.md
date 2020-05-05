# Guidelines

The goal of this package is to find solutions $x$ to nonlinear equations 

$$F(x,p) = 0 \quad\tag{E}$$

in large dimensions as function of a *real* parameter $p$. We want to be able to do so on GPU, distributed systems...

The core of the package is built around a Newton-Krylov solver (see [`newton`](@ref)) which allows to solve equations of the form $F(x)=0$, or a solution $x_0$ to (E) for a particular $p_0$.

Once such a solution $(x_0,p_0)$ is known, we can continue this solution by building a 1d curve of solutions $\gamma = (x(s),p(s))_{s\in I}$ passing through $(x_0,p_0)$ (see [`continuation`](@ref)).

> In practice, you don't need to know exactly $(x_0,p_0)$ to compute $\gamma$. Indeed, [`continuation`](@ref) will call [`newton`](@ref) to refine any initial guess that you will pass.

## Bifurcation analysis of Equilibria

We can detect if the curve of solutions $\gamma$ crosses another curve of solutions $\gamma^{bif}$ *without knowing* $\gamma^{bif}$! The intersection point $(x^b,p^b)\in\gamma$ is called a bifurcation point and is such that $\partial_xF(x^b,p^b)$ is non invertible. When calling [`continuation`](@ref), `γ, _ = continuation(...)` with the option `detectBifurcation > 0` inside [`ContinuationPar`](@ref), the bifurcation points are automatically detected and stored in `γ.bifpoints`.

### Branch switching 

In the simple cases, *e.g.* when $dim\ker \partial_xF(x^b,p^b) = 1$, we can compute automatically the **bifurcated branch** $\gamma^{bif}$ by calling [`continuation`](@ref) and passing $\gamma$. This is explained in [Simple bifurcation branch switching](@ref). Recursively, we can compute the curves of solutions which are connected to $(x_0,p_0)$, this is called a **bifurcation diagram**.

When $d\equiv dim\ker \partial_xF(x^b,p^b) > 1$, there is no automatic method to perform branch switching. Nevertheless, we can reduce (E) to a $d$ dimensional multivariate polynomials equations in $d$ unknowns whose solutions gives the local topology of branches in the neighborhood of the bifurcation point $(x^b,p^b)$. We can then use the solutions of this **reduced equation** as initial guesses and call again [`continuation`](@ref) to compute the bifurcated branches. This is explained in [Non-simple bifurcation branch switching](@ref) and an example of shown in [A generalized Bratu–Gelfand problem in two dimensions](@ref).	
> In the case $d=1$, the reduced equation can be further simplified into a **normal form**. This is also automatically computed by the package.


## Bifurcation analysis of Cauchy problems

The goal of this section is to study the dynamics of Cauchy problems

$$\frac{d}{dt}x - F(x,p) = 0 \quad\tag{C}$$

The equilibria are time independent solutions of (C) hence solving (E). The previous section can be applied to compute curves of equilibria. However, we can do more. By discretizing time, we can recast (C) in the general form (E) and look for time dependent solutions as well. 

We can detect the existence of periodic solutions close to $\gamma$. This is done automatically and those bifurcation points are stored in `` as well with the name of **Hopf bifurcation points**.  

### Branch switching at Hopf points

We will not review the bifurcation of equilibria (see above). Therefore, we focus on computing the branch of periodic solutions branching of a Hopf point. This is done automatically by calling again [`continuation`](@ref), passing $\gamma$ and choosing a time discretization (see [Periodic orbits computation](@ref)). Some details about this branch switching is given in [Simple Hopf branch switching](@ref).

### Branch switching at bifurcation points of periodic orbits

Once a branch of periodic orbits $\gamma^{po}$ have been computed (see for example previous section), several bifurcation points are detected (branch point, period doubling and Neimark Sacker). We do not provide an automatic branch switching for those points. However, for branch points of periodic orbits, you can call [`continuation`](@ref) by passing $\gamma^{po}$ and some simple arguments (amplitude of the periodic orbits) to perform branch switching in a semi-automatic way.

!!! tip "Manual Branch switching"
    You can perform **manual** branch switching by computing the nearby solutions close to a bifurcation point using a deflated newton (see [Deflated problems](@ref)), which provide a way to compute solutions other than a set of already known solutions.  You can then use these solutions to compute branches by calling `continuation`. Many, if not all, tutorials give example of doing so like [A generalized Bratu–Gelfand problem in two dimensions](@ref) or [Brusselator 1d](@ref).