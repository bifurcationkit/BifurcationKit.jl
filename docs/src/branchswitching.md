# Branch switching

## Branch switching from simple branch point to equilibria

You can perform automatic branch switching by calling `continuation` with the following options:

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar;
	Jt = nothing, δ = 1e-8, nev = 5, verbose = false, kwargs...)
```

where `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. This call computes the branch bifurcating from the `ind_bif `th bifurcation point in `br`. An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

## Branch switching from non simple branch point to equilibria

We do not provide automatic branch switching in this case. The method is to first compute the reduced equation (see [Non-simple branch point](@ref)) and use it to compute the nearby solutions (see tutorial [A generalized Bratu–Gelfand problem in two dimensions](@ref)). You can then use these solutions as initial guess for [`continuation`](@ref).

## Branch switching from Hopf point to periodic orbits

In order to compute the bifurcated branch of periodic solutions at a Hopf bifurcation point, you need to choose a method. Indeed, we provide two methods to compute periodic orbits:

- [Periodic orbits based on finite differences](@ref)
- [Periodic orbits based on the shooting method](@ref)

Once you have decided which method you want, you can call the following method.

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, _contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ;
	Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, kwargs...)
```

We refer to [`continuation`](@ref) for more information about the arguments. Here, we just say a few words about how we can specify `prob::AbstractPeriodicOrbitProblem`. For [Periodic orbits based on finite differences](@ref), you can pass `prob = PeriodicOrbitTrapProblem(M = 51)` where `M` is the number of times slices in the periodic orbit. For [Periodic orbits based on the shooting method](@ref), you need more parameters. For example, you can pass `prob = ShootingProblem(2, par, prob, Euler())` or `prob = PoincareShootingProblem(2, par, prob, Euler())` where `prob::ODEProblem` is an ODE problem to specify the Cauchy problem and `par` is the set of parameters passed to the vector field and which must be the same as `br.params`.

Several examples are provided like [Brusselator 1d (automatic)](@ref) or [Complex Ginzburg-Landau 2d](@ref).

!!! tip "Precise options"
    Although very convenient, the automatic branch switching does not allow the very fine tuning of parameters. It must be used as a first attempt before recurring to manual branch switching
    
## Branch switching from Branch point of curve of periodic orbits

We only provide (for now) this method for the case of [`PeriodicOrbitTrapProblem`](@ref). The call is as follows. Please note that a deflation is included in this method to simplify branch switching. 

An example of use is provided in [Brusselator 1d (automatic)](@ref).

```@docs
continuationPOTrapBPFromPO
```


