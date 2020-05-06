# Branch switching

## Branch switching from simple branch point to equilibria

You can perform automatic branch switching by calling `continuation` with the following options:

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ;
	Jt = nothing, δ = 1e-8, kwargs...)
```

where `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. This call will compute the branch bifurcating from the `ind_bif `th bifurcation point in `br`. An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

## Branch switching from non simple branch point to equilibria

We do not provide automatic branch switching in this case. The method is to first compute the reduced equation (see [Non-simple branch point](@ref)) and use it to compute the nearby solutions (see tutorial [A generalized Bratu–Gelfand problem in two dimensions](@ref)). You can then use these solutions as initial guess for [`continuation`](@ref).

## Branch switching from Hopf point to periodic orbits

In order to compute the bifurcated branch of periodic solutions at a Hopf bifurcation point, you need to choose a method. Indeed, we provide two methods to compute periodic orbits:

- [Periodic orbits based on finite differences](@ref)
- [Periodic orbits based on the shooting method](@ref)

You can perform automatic branch switching by calling `continuationPOTrap`(and soon `continuationPOShooting`) with the following options:

```julia
continuationPOTrap(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, contParams::ContinuationPar; 
	Jt = nothing, δ = 1e-8, δp = nothing, 
	linearPO = :BorderedLU, M = 21, 
	printSolution = (u, p) -> u[end], 
	linearAlgo = BorderingBLS(), kwargs...)
```

and

```julia
# coming soon
```

where `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. This call will compute the branch bifurcating from the `ind_bif `th bifurcation point in `br`. 

> Some examples of use are provided in [Brusselator 1d](@ref) and [Continuation of periodic orbits (Standard Shooting)](@ref)


