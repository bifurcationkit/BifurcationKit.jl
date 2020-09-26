# Fold / Hopf Continuation

For this to work, it is important to have an analytical expression for the jacobian. See the tutorial [Temperature model (simplest example for equilibria)](@ref) for more details.

## Newton refinement

Once a Fold/Hopf point has been detected after a call to `br, _ = continuation(...)`, it can be refined using `newton` iterations. We have implemented a **Minimally Augmented** formulation. A simplified interface is provided.

Let us say that `ind_bif` is the index in `br.bifpoint` of a Fold/Hopf point. This guess can be refined by newton iterations by doing 

```julia
outfold, hist, flag = newton(F, J, br::ContResult, ind_bif::Int64, 
	par, lens::Lens; Jt = nothing, d2F = nothing, normN = norm, 
	options = br.contparams.newtonOptions, kwargs...)
```

where `par` is the set of parameters used in the call to [`continuation`](@ref) to get `br` and `lens` is the parameter axis which is used to find the Fold/Hopf point. For the options parameters, we refer to [Newton](@ref).

It is important to note that for improved performances, a function implementing the expression of the **hessian** should be provided. This is by far the fastest. Reader interested in this advanced usage should look at the code `example/chan.jl` of the tutorial [Temperature model (simplest example for equilibria)](@ref). 

## Codim 2 continuation

To compute the codim 2 curve of Fold/Hopf points, one can call [`continuation`](@ref) with the following options

```julia
br_codim2, _ = continuation(F, J, br, ind_bif, 
	par, lens1::Lens, lens2::Lens, options_cont::ContinuationPar ;
	Jt = nothing, d2F = nothing, kwargs...)
```

where the options are as above except with have two parameter axis `lens1, lens2` which are used to locate the bifurcation points. See [Temperature model (simplest example for equilibria)](@ref) for an example of use. 

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
