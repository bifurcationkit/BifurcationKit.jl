# [Branch switching](@id Branch-switching-page)

The precise definition of the methods are given [Branch switching (branch point)](@ref) and [Branch switching (Hopf point)](@ref).

## Branch switching from simple branch point to equilibria

You can perform automatic branch switching by calling `continuation` with the following options:

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar;
	Jᵗ = nothing, δ = 1e-8, nev = 5, verbose = false, kwargs...)
```

where `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. This call computes the branch bifurcating from the `ind_bif `th bifurcation point in `br`. An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

> See [Branch switching (branch point)](@ref) precise method definition

## Branch switching from non simple branch point to equilibria

We provide an *experimental* automatic branch switching method in this case. The method is to first compute the reduced equation (see [Non-simple branch point](@ref)) and use it to compute the nearby solutions. These solutions are seeded as initial guess for [`continuation`](@ref). Hence, you can perform automatic branch switching by calling `continuation` with the following options:

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar;
	Jᵗ = nothing, δ = 1e-8, nev = 5, verbose = false, kwargs...)
```

An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

> See [Branch switching (branch point)](@ref) for precise method definition

## Branch switching from Hopf point to periodic orbits

In order to compute the bifurcated branch of periodic solutions at a Hopf bifurcation point, you need to choose a method. Indeed, we provide two methods to compute periodic orbits:

- [Periodic orbits based on trapezoidal rule](@ref)
- [Periodic orbits based on the shooting method](@ref)

Once you have decided which method you want, you can call the following method.

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, _contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ;
	Jᵗ = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, kwargs...)
```

We refer to [`continuation`](@ref) for more information about the arguments. Here, we just say a few words about how we can specify `prob::AbstractPeriodicOrbitProblem`. For [Periodic orbits based on trapezoidal rule](@ref), you can pass `prob = PeriodicOrbitTrapProblem(M = 51)` where `M` is the number of times slices in the periodic orbit. For [Periodic orbits based on the shooting method](@ref), you need more parameters. For example, you can pass `prob = ShootingProblem(2, par, prob, Euler())` or `prob = PoincareShootingProblem(2, par, prob, Euler())` where `prob::ODEProblem` is an ODE problem to specify the Cauchy problem and `par` is the set of parameters passed to the vector field and which must be the same as `br.params`.

Several examples are provided like [1d Brusselator (automatic)](@ref) or [2d Ginzburg-Landau equation (finite differences)](@ref).

> See [Branch switching (Hopf point)](@ref) precise method definition

!!! tip "Precise options"
    Although very convenient, the automatic branch switching does not allow the very fine tuning of parameters. It must be used as a first attempt before recurring to manual branch switching
    
## Branch switching from Branch / Period-doubling point of curve of periodic orbits

We do not provide (for now) the associated normal forms to these bifurcations of periodic orbits. As a consequence, the user is asked to provide the amplitude of the bifurcated solution.

We provide the branching method for all methods to compute periodic orbits, *i.e.* for [`PeriodicOrbitTrapProblem`](@ref),[`ShootingProblem`](@ref),[`PoincareShootingProblem`](@ref). The call is as follows. Please note that a deflation is included in this method to simplify branch switching. 

An example of use is provided in [Period doubling in Lur'e problem (PD aBS)](@ref).

```julia
continuation(br::AbstractBranchResult, ind_bif::Int, contParams::ContinuationPar; 
	δp = 0.1, ampfactor = 1, usedeflation = false, kwargs...)
```


