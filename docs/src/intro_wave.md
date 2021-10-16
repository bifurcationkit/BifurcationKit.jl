# Freezing problems, symmetries and waves

!!! warning "WIP"
    This is work in progress. The available options are limited if any.

This section is dedicated to the study of an equation (in `x`) `F(x,p)=0` where one wishes to freeze a continuous symmetry. When the equation $F(x, p) = 0$ has a continuous symmetry described by a Lie group $G$ and action $g\cdot x$ for $g\in G$, one can reduce the symmetry of the problem by considering the constrained problem:

$$\left\{
\begin{array}{l}
F(x, p) - s\cdot T\cdot x=0 \\
\langle T\cdot x_{ref},x-x_{ref}\rangle=0
\end{array}\right.$$

where $T$ is a generator of the Lie algebra associated to $G$, $x_{ref}$ is a reference solution and $s$ is the speed. This is known as the *freezing method*.

!!! unknown "Reference"
    See Beyn and Thümmler, **Phase Conditions, Symmetries and PDE Continuation.** for more information on the *freezing method*.

## Freezing symmetries

The method of freezing is handled by the type `TWProblem`. It allows to compute and continue waves as function of a parameter.
