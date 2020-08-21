# BifurcationKit.jl

This Julia package aims at performing **automatic bifurcation analysis** of large dimensional equations F(u,λ)=0 where λ∈ℝ.  

It incorporates a pseudo arclength continuation algorithm which provides a *predictor* (u1,λ1) from a known solution (u0,λ0). A Newton-Krylov method is then used to correct this predictor and a Matrix-Free eigensolver is used to compute stability and bifurcation points.

By leveraging on the above method, it can also seek for periodic orbits of Cauchy problems by casting them into an equation F(u,λ)=0 of high dimension. **It is by now, one of the only softwares which provides shooting methods AND methods based on finite differences to compute periodic orbits.**

The current package focuses on large scale nonlinear problems and multiple hardwares. Hence, the goal is to use Matrix Free methods on **GPU** (see [PDE example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorials2b/index.html#The-Swift-Hohenberg-equation-on-the-GPU-1) and [Periodic orbit example](https://rveltz.github.io/BifurcationKit.jl/dev/tutorialsCGL/#Continuation-of-periodic-orbits-on-the-GPU-(Advanced)-1)) or on a **cluster** to solve non linear PDE, nonlocal problems, compute sub-manifolds...

One design choice is that we do not require `u` to be a subtype of an `AbstractArray` as this would forbid the use of spectral methods like the one from `ApproxFun.jl`. So far, our implementation does not allow this for Hopf continuation and computation of periodic orbits. It will be improved later.

Finally, we leave it to the user to take advantage of automatic differentiation as this field is moving too fast for now, albeit there are several well established packages like `ForwardDiff.jl` and `Zygote.jl` to name just a few.

## Installation 

This package requires Julia >= v1.3.0 because of the use of methods added to abstract types (see [#31916](https://github.com/JuliaLang/julia/pull/31916)).

To install it, please run

`] add BifurcationKit`

## Citing this work
If you use this package for your work, please **cite** it!! Open source development strongly depends on this. It is referenced on HAL-Inria as follows:

```
@misc{veltz:hal-02902346,
  TITLE = {{BifurcationKit.jl}},
  AUTHOR = {Veltz, Romain},
  URL = {https://hal.archives-ouvertes.fr/hal-02902346},
  INSTITUTION = {{Inria Sophia-Antipolis}},
  YEAR = {2020},
  MONTH = Jul,
  KEYWORDS = {pseudo-arclength-continuation ; periodic-orbits ; floquet ; gpu ; bifurcation-diagram ; deflation ; newton-krylov},
  PDF = {https://hal.archives-ouvertes.fr/hal-02902346/file/354c9fb0d148262405609eed2cb7927818706f1f.tar.gz},
  HAL_ID = {hal-02902346},
  HAL_VERSION = {v1},
}
```

## Other softwares

There are many good softwares already available, most of them are listed on [DSWeb](https://dsweb.siam.org/Software). One can mention the venerable AUTO, or also, [XPPAUT](http://www.math.pitt.edu/~bard/xpp/xpp.html), [MATCONT](http://www.matcont.ugent.be/) and [COCO](https://sourceforge.net/projects/cocotools/). For large scale problems, there is [Trilinos](https://trilinos.org/), the versatile [pde2path](http://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/), [CL_MATCONTL](https://github.com/careljonkhout/cl_matcontL) and the python libraries [pyNCT](https://pypi.org/project/PyNCT/) and [pacopy](https://github.com/nschloe/pacopy).


In Julia, we have for now a [wrapper](https://github.com/JuliaDiffEq/PyDSTool.jl) to PyDSTools, and also [Bifurcations.jl](https://github.com/tkf/Bifurcations.jl).

## A word on performance

The examples which follow have not **all** been written with the goal of performance but rather simplicity (except maybe [Complex Ginzburg-Landau 2d](@ref)). One could surely turn them into more efficient codes. The intricacies of PDEs make the writing of efficient code highly problem dependent and one should take advantage of every particularity of the problem under study.

For example, in the first tutorial on [Temperature model](@ref), one could use `BandedMatrices.jl` for the jacobian and an inplace modification when the jacobian is called ; using a composite type would be favored. Porting them to GPU would be another option.

## Main features

- Newton-Krylov solver with generic linear / eigen *preconditioned* solver. Idem for the arc-length continuation.
- Newton-Krylov solver with nonlinear deflation and preconditioner. It can be used for branch switching for example.
- Bifurcation points are located using a bisection algorithm
- Branch, Fold, Hopf bifurcation point detection of stationary solutions.
- Automatic branch switching at simple branch points
- Automatic branch switching at simple Hopf points to periodic orbits
- **Automatic bifurcation diagram computation**
- Fold / Hopf continuation based on Minimally Augmented formulation, with Matrix Free / Sparse Jacobian.
- Periodic orbit computation and continuation using Shooting or Finite Differences.
- Branch, Fold, Neimark-Sacker, Period Doubling bifurcation point detection of periodic orbits.
- Computation and Continuation of Fold of periodic orbits

Custom state means, we can use something else than `AbstractArray`, for example your own `struct` (see [Requested methods for Custom State](@ref)). 

**Note that you can combine most of the solvers, like use Deflation for Periodic orbit computation or Fold of periodic orbits family.**

|Features|Matrix Free|Custom state| Tutorial |
|---|---|---|---|
| (Deflated) Newton| Y | Y | 4, 5| :heavy_check_mark:|
| Continuation (Natural, Secant, Tangent) | Y | Y | All |
| Branching / Fold / Hopf point detection | Y | Y | All / All / 6 - 9 | :heavy_check_mark: |
| Fold Point continuation | Y | Y | 1, 8 |
| Hopf continuation | Y | `AbstractArray` | 6 |
| Branch switching at Branch / Hopf points | Y | `AbstractArray` | 3 |
| <span style="color:red">**Automatic bifurcation diagram computation**</span> | Y | `AbstractArray` |  Yes | |
| Periodic Orbit (FD) Newton / continuation | Y | `AbstractVector` | 7, 8 |
| Periodic Orbit with Parallel Poincaré / Standard Shooting Newton / continuation | Y | `AbstractArray` |  6, 7, 9 |
| Fold, Neimark-Sacker, Period doubling detection | Y | `AbstractVector` | 6 - 9  |
| Continuation of Fold of periodic orbits | Y | `AbstractVector` | 8 |


## Requested methods for Custom State
Needless to say, if you use regular arrays, you don't need to worry about what follows.

We make the same requirements than `KrylovKit.jl`. Hence, we refer to its [docs](https://jutho.github.io/KrylovKit.jl/stable/#Package-features-and-alternatives-1) for more information. We additionally require the following methods to be available:

- `Base.length(x)`: it is used in the constraint equation of the pseudo arclength continuation method (see [`continuation`](@ref) for more details). If `length` is not available for your "vector", define it `length(x) = 1` and adjust tuning the parameter `theta` in `ContinuationPar`.
- `Base.copyto!(dest, in)` this is required to reduce the allocations by avoiding too many copies