# Constrained problems

!!! compat "Experimental"
    This feature is still experimental. It has not been tested thoroughly, especially the case of multiple constraints and matrix-free functionals.

This section is dedicated to the study of an equation (in `x`) `F(x,p)=0` where one wishes to add a constraint `g(x,p)=0`. Hence, one is interested in solving in the couple $(x,p)$:

$$\left\{
\begin{array}{l}
F(x,p)=0 \\
g(x,p)=0
\end{array}\right.$$

There are several situations where this proves useful:

1. the pseudo-arclength continuation method is such a constrained problem, see [`continuation`](@ref) for more details.
2. when the equation $F(x)$ has a continuous symmetry described by a Lie group $G$ and action $g\cdot x$ for $g\in G$. One can reduce the symmetry of the problem by considering the constrained problem:
$$\left\{
\begin{array}{l}
F(x) + p\cdot T\cdot x=0 \\
\langle T\cdot x_{ref},x-x_{ref}\rangle=0
\end{array}\right.$$
where $T$ is a generator of the Lie algebra associated to $G$ and $x_{ref}$ is a reference solution. This is known as the *freezing method*.

!!! unknown "Reference"
    See Beyn and Thümmler, **Phase Conditions, Symmetries and PDE Continuation.** for more information on the *freezing method*.

## Encoding of the functional

A composite type which implements this functional:

```@docs
BorderedProblem
```


