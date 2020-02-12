# Fold / Hopf Continuation

For this to work, it is important to have an analytical expression for the jacobian. See the tutorial [Temperature model](@ref) for more details.

## The case of the Fold point

Once a Fold point has been detected after a call to `br, _ = continuation(...)`, it can be refined using `newton` iterations. We have implemented a **Minimally Augmented** formulation. A simplified interface is provided.

### Newton refinement

Let us say that `ind_fold` is the index in `br.bifpoint` of a Fold point. This guess can be refined by calling the following simplified interface. More precisions are provided below for an advanced usage.

```julia
outfold, hist, flag = @time newtonFold(
					(x, p) ->   F(x, p),
					(x, p) -> Jac(x, p),
					br, ind_fold,
					opt_newton)
```

It is important to note that for improved performances, a function implementing the expression of the **hessian** should be provided. This is by far the fastest. Reader interested in this advanced usage should look at the code `example/chan.jl` of the tutorial [Temperature model](@ref). Although it is a simple problem, many different use case are shown in a simple setting. See also [`newtonFold`](@ref).

## The case of the Hopf point

One a Hopf point have been detected after a call to `br, _ = continuation(...)`, it can be refined using `newton` iterations. We have implemented a **Minimally Augmented** formulation. A simplified interface is provided as for the Fold case.

### Newton refinement

Let us say that `ind_hopf` is the index in `br.bifpoint` of a Hopf point. This guess can be refined by calling the simplified interface. More precisions are provided below for an advanced usage. See also [`newtonHopf`](@ref).

```julia
outfold, hist, flag = @time newtonHopf(
					(x, p) ->   F(x, p),
					(x, p) -> Jac(x, p),
					br, ind_hopf,
					opt_newton)
```


## Methods

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
