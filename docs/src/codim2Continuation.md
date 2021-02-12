# Fold / Hopf Continuation

For this to work best, it is necessary to have an analytical expression for the jacobian. See the tutorial [Temperature model (simplest example for equilibria)](@ref) for more details.

## Fold continuation

The continuation of Fold bifurcation points is based on a **Minimally Augmented**[^Govaerts] formulation which is an efficient way to detect singularities. The continuation of Fold points is based on the formulation $G(u,p) = (F(u,p), g(u,p))\in\mathbb R^{n+1}$ where the test function $g$ is solution of

$$\left[\begin{array}{cc}
dF(u,p) & w \\
v^{\top} & 0
\end{array}\right]\left[\begin{array}{c}
r \\
g(u,p)
\end{array}\right]=\left[\begin{array}{c}0_{n} \\1\end{array}\right]\quad\quad (M_f)$$

where $w,v$ are chosen in order to have a non-singular matrix $(M_f)$. More precisely, $w$ (resp. $v$) should be a left (resp. right) approximate null vector of $dF(u,p)$. During continuation, the vectors $w,v$ are updated so that the matrix $(M_f)$ remains non-singular ; this is controlled with the argument `updateMinAugEveryStep` (see below).

## Hopf continuation

The continuation of Fold bifurcation points is based on a **Minimally Augmented**[^Govaerts] formulation which is an efficient way to detect singularities. The continuation of Hopf points is based on the formulation $G(u,\omega,p) = (F(u,\omega,p), g(u,\omega,p))\in\mathbb R^{n+2}$ where the test function $g$ is solution of

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


## Newton refinement

Once a Fold/Hopf point has been detected after a call to `br, _ = continuation(...)`, it can be refined using `newton` iterations. Let us say that `ind_bif` is the index in `br.bifpoint` of a Fold/Hopf point. This guess can be refined by newton iterations:

```julia
outfold, hist, flag =  newton(F, J, br::AbstractBranchResult, ind_bif::Int64; 
	Jᵗ = nothing, d2F = nothing, normN = norm, 
	options = br.contparams.newtonOptions, startWithEigen = false, kwargs...)
```

where `par` is the set of parameters used in the call to [`continuation`](@ref) to compute `br`. For the options parameters, we refer to [Newton](@ref).

It is important to note that for improved performances, a function implementing the expression of the **hessian** should be provided. This is by far the fastest. Reader interested in this advanced usage should look at the code `example/chan.jl` of the tutorial [Temperature model (simplest example for equilibria)](@ref). 

## Codim 2 continuation

To compute the codim 2 curve of Fold/Hopf points, one can call [`continuation`](@ref) with the following options

```@docs
 continuation(F, J, br::BifurcationKit.AbstractBranchResult, ind_bif::Int64, lens2::Setfield.Lens, options_cont::BifurcationKit.ContinuationPar ; startWithEigen = false, Jᵗ = nothing, d2F = nothing, kwargs...)
```

where the options are as above except with have an additional parameter axis `lens2` which is used to locate the bifurcation points. 


See [Temperature model (simplest example for equilibria)](@ref) for an example of use. 

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

