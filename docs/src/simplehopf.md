# Simple Hopf branch switching


We expose our method to perform branch switching at a Hopf bifurcation point. At a Hopf branch point $(x_0,p_0)$ for the problem $F(x,p)=0$, we have $\Sigma\ dF(x_0,p_0) = \{\pm i\omega \},\ \omega\in\mathbb R$. At such point, we can compute the **normal form** to transform the initial Cauchy problem

$$\dot x = F(x,p)$$

in large dimensions to a **complex** polynomial vector field: 

$$\dot z = z\left(a \delta p + i\omega + b|z|^2\right)\quad\text{(E)}$$

whose solutions give access to the solutions of the Cauchy problem in a neighborhood of $(x,p)$.

More precisely, if $J \equiv dF(x_0,p_0)$, then we have $J\zeta = i\omega\zeta$ and $J\bar\zeta = -i\omega\bar\zeta$ for some complex eigenvector $\zeta$. It can be shown that $x(t) \approx x_0 + 2\Re(z(t)\zeta)$.


## Normal form computation

The reduced equation (E) can be automatically computed as follows

```julia
computeNormalForm(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int, options::NewtonPar ; Jt = nothing, 
	δ = 1e-8, nev = 5, verbose = false)
```

where `dF, d2F,d3F` are the differentials of `F`, `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. The above call returns a point with information needed to compute the bifurcated branch.

```julia
mutable struct HopfBifPoint{Tv, T, Tω, Tevr, Tevl, Tnf} <: BifurcationPoint
	"Hopf point"
	x0::Tv

	"Parameter value at the Hopf point"
	p::T

	"Frequency of the Hopf point"
	ω::Tω

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζstar::Tevl

	"Normal form coefficient (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end
```

!!! info "Note"
    You should not need to call `computeNormalForm ` except if you want to have the full information about the branch point. Indeed, the call in the next section do it internally.

## Automatic branch switching

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

