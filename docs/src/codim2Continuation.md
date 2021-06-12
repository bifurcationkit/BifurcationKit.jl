# Fold / Hopf Continuation

For this to work best, it is necessary to have an analytical expression for the jacobian. See the tutorial [Temperature model (Simplest example)](@ref) for more details.

In this page, we explain how to perform continuation of Fold / Hopf points and detect the associated bifurcations.

### List of detected bifurcation points
|Bifurcation|index used|
|---|---|
| Bogdanov-Takens | bt |
| Bautin | gh |
| Cusp | cusp |

In a nutshell, all you have to do (see below) is to call `continuation(F, J, br, ind_bif)` to continue the bifurcation point stored in `br.specialpoint[ind_bif]` and set proper options. 

## Fold continuation

The continuation of Fold bifurcation points is based on a **Minimally Augmented**[^Govaerts] formulation which is an efficient way to detect singularities. The continuation of Fold points is based on the formulation $G(u,p) = (F(u,p), g(u,p))\in\mathbb R^{n+1}$ where the test function $g$ is solution of

$$\left[\begin{array}{cc}
dF(u,p) & w \\
v^{\top} & 0
\end{array}\right]\left[\begin{array}{c}
r \\
g(u,p)
\end{array}\right]=\left[\begin{array}{c}0_{n} \\1\end{array}\right]\quad\quad (M_f)$$

where $w,v$ are chosen in order to have a non-singular matrix $(M_f)$. More precisely, $v$ (resp. $w$) should be close to a null vector of `dF(u,p)` (resp. `dF(u,p)'`). During continuation, the vectors $w,v$ are updated so that the matrix $(M_f)$ remains non-singular ; this is controlled with the argument `updateMinAugEveryStep` (see below).

> note that there are very simplified calls for this, see **Newton refinement** below. In particular, you don't need to set up the Fold Minimally Augmented problem yourself. This is done in the background.

!!! warning "Linear Method"
    You can pass the bordered linear solver to solve $(M_f)$ using the option `bdlinsolver ` (see below). Note that the choice `bdlinsolver = BorderingBLS()` can lead to singular systems. Indeed, in this case, $(M_f)$ is solved by inverting `dF(u,p)` which is singular at Fold points.

## Hopf continuation

The continuation of Fold bifurcation points is based on a **Minimally Augmented** (see [^Govaerts] p. 87) formulation which is an efficient way to detect singularities. The continuation of Hopf points is based on the formulation $G(u,\omega,p) = (F(u,\omega,p), g(u,\omega,p))\in\mathbb R^{n+2}$ where the test function $g$ is solution of

$$\left[\begin{array}{cc}
dF(u,p)-i\omega I_n & w \\
v^{\top} & 0
\end{array}\right]\left[\begin{array}{c}
r \\
g(u,\omega,p)
\end{array}\right]=\left[\begin{array}{c}
0_{n} \\
1
\end{array}\right]\quad\quad (M_h)$$

where $w,v$ are chosen in order to have a non-singular matrix $(M_h)$. More precisely, $w$ (resp. $v$) should be a left (resp. right) approximate null vector of $dF(u,p)-i\omega I_n$. During continuation, the vectors $w,v$ are updated so that the matrix $(M_h)$ remains non-singular ; this is controlled with the argument `updateMinAugEveryStep ` (see below).

> note that there are very simplified calls to this, see **Newton refinement** below. In particular, you don't need to set up the Hopf Minimally Augmented problem yourself. This is done in the background.

!!! warning "Linear Method"
    You can pass the bordered linear solver to solve $(M_h)$ using the option `bdlinsolver ` (see below). Note that the choice `bdlinsolver = BorderingBLS()` can lead to singular systems. Indeed, in this case, $(M_h)$ is solved by inverting `dF(u,p)` which is singular at Fold points.


## Newton refinement

Once a Fold/Hopf point has been detected after a call to `br, = continuation(...)`, it can be refined using `newton` iterations. Let us say that `ind_bif` is the index in `br.specialpoint` of a Fold/Hopf point. This guess can be refined as follows:

```julia
outfold, hist, flag =  newton(F, J, br::AbstractBranchResult, ind_bif::Int; 
	issymmetric = false, Jᵗ = nothing, d2F = nothing, 
	normN = norm, options = br.contparams.newtonOptions, 
	bdlinsolver = BorderingBLS(options.linsolver),
	startWithEigen = false, kwargs...)
```

where `par` is the set of parameters used in the call to [`continuation`](@ref) to compute `br`. For the options parameters, we refer to [Newton](@ref).

It is important to note that for improved performances, a function implementing the expression of the **hessian** should be provided. This is by far the fastest. Reader interested in this advanced usage should look at the code `example/chan.jl` of the tutorial [Temperature model (Simplest example)](@ref). 

## Codim 2 continuation

To compute the codim 2 curve of Fold/Hopf points, one can call [`continuation`](@ref) with the following options

```@docs
 continuation(F, J,
				br::BifurcationKit.AbstractBranchResult, ind_bif::Int64,
				lens2::Lens, options_cont::ContinuationPar = br.contparams ;
				kwargs...)
```

where the options are as above except with have an additional parameter axis `lens2` which is used to locate the bifurcation points. 


See [Temperature model (Simplest example)](@ref) for an example of use. 

## Advanced use

Here, we expose the solvers that are used to perform newton refinement or codim 2 continuation in case the above methods fails. This is useful in case it is too involved to expose the linear solver options. An example of advanced use is the continuation of Folds of periodic orbits, see [Continuation of Fold of periodic orbits](@ref).

```@docs
newtonFold
```

```@docs
newtonHopf
```


```@docs
continuationFold
```

```@docs
continuationHopf
```

## References

[^Govaerts]: > Govaerts, Willy J. F. Numerical Methods for Bifurcations of Dynamical Equilibria. Philadelphia, Pa: Society for Industrial and Applied Mathematics, 2000.

