# Fold / Hopf Continuation

For this to work, it is important to have an analytical expression for the jacobian. See the example `Chan` for more details.

## The case of the Fold point

One a Fold point have been detected after a call to `br, _ = continuation(...)`, it can be refined with the use of `newton` iterations. We have implemented a **Minimally Augmented** formulation. A simplified interface is provided for its use of the later.

### Newton refinement

Let us say that `ind_fold` is the index in `br.bifpoint` of a Fold point. This guess can be refined by calling the simplified interface. More precisions are provided below for an advanced usage.

```julia
outfold, hist, flag = @time Cont.newtonFold((x,p) -> F(x, p),
							(x, p) -> Jac(x, p),
							br, ind_fold,
							opt_newton)
```

## The case of the Hopf point

One a Hopf point have been detected after a call to `br, _ = continuation(...)`, it can be refined with the use of `newton` iterations. We have implemented a **Minimally Augmented** formulation. A simplified interface is provided for its use of the later.

### Newton refinement

Let us say that `ind_hopf` is the index in `br.bifpoint` of a Hopf point. This guess can be refined by calling the simplified interface. More precisions are provided below for an advanced usage.

```julia
outfold, hist, flag = @time Cont.newtonHopf((x,p) -> F(x, p),
							(x, p) -> Jac(x, p),
							br, ind_hopf,
							opt_newton)
```


## Functions

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
