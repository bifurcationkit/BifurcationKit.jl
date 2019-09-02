# PseudoArcLengthContinuation.jl

This package aims at solving equations $F(u,\lambda)=0$ where $\lambda \in\mathbb R$ starting from an initial guess $(u_0,\lambda_0)$. It relies on the pseudo arclength continuation algorithm which provides a *predictor* $(u_1,\lambda_1)$ from $(u_0,\lambda_0)$. A Newton method is then used to correct this predictor.

The current package focuses on large scale problem and multiple hardware. Hence, the goal is to use Matrix Free / Sparse methods on GPU or a cluster in order to solve non linear equations (PDE for example).

Finally, we leave it to the user to take advantage of automatic differentiation.

## Other softwares

We were inspired by [pde2path](http://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/). One can also mention the venerable AUTO, or also, [XPPAUT](http://www.math.pitt.edu/~bard/xpp/xpp.html), [MATCONT](http://www.matcont.ugent.be/) and [COCO](https://sourceforge.net/projects/cocotools/) or [Trilinos](https://trilinos.org/). Most continuation softwares are listed on [DSWeb](https://dsweb.siam.org/Software). There is also this MATLAB continuation [code](https://www.dropbox.com/s/inqwpl0mp7o1oy0/AvitabileICMNS2016Workshop.zip?dl=0) by [D. Avitabile](https://www.maths.nottingham.ac.uk/plp/pmzda/index.html).


In Julia, we have for now a [wrapper](https://github.com/JuliaDiffEq/PyDSTool.jl) to PyDSTools, and also [Bifurcations.jl](https://github.com/tkf/Bifurcations.jl).

One design choice is that we do not require `u` to be a subtype of an `AbstractArray` as this would forbid the use of spectral methods like the one from `ApproxFun.jl`. So far, our implementation does not allow this for Hopf continuation and computation of periodic orbits. It will be improved later.

## A word on performance

The examples which follow have not been written with the goal of performance but rather simplicity. One could surely turn them into more efficient codes. The intricacies of PDEs make the writing of efficient code highly problem dependent and one should take advantage of every particularity of the problem under study.

For example, in the first example below, one could use `BandedMatrices.jl` for the jacobian and an inplace modification when the jacobian is called ; using a composite type would be favored. Porting them to GPU would be another option.


