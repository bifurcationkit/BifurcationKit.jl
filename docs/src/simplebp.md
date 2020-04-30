# Simple bifurcation branch switching

!!! unknown "References"
    The general method is exposed in Golubitsky, Martin, David G Schaeffer, and Ian Stewart. **Singularities and Groups in Bifurcation Theory**. New York: Springer-Verlag, 1985, VI.1.d page 295

We expose our method to perform branching at a simple branch point. A simple branch point $(x_0,p_0)$ for the problem $F(x,p)=0$ satisfies $\dim \ker dF(x_0,p_0) = 1$. At such point, we can apply **Lyapunov-Schmidt** reduction to transform the initial problem in large dimensions to a **scalar** polynomial: 

$$a\delta p + z\left(b_1\delta p + \frac{b_2}{2}z + \frac{b_3}{6}z^2\right) = 0 \tag{E}$$

whose solutions give access to all solutions in a neighborhood of $(x,p)$.

More precisely, if $\ker dF(x_0,p_0) = \mathbb R\zeta$, one can show that $x_0+z\zeta$ is close to a solution on a new branch, thus satisfying $F(x_0+z\zeta,p_0+\delta p)\approx 0$.

In the above scalar equation,

- if $a\neq 0$, this is a *Saddle-Node* bifurcation
- if $a=0,b_2\neq 0$, the bifurcation point is a *Transcritical* one where the bifurcated branch exists on each side of $p$.
- if $a=0,b_2=0, b_3\neq 0$, the bifurcation point is a *Pitchfork* one where the bifurcated branch only exists on one side of $p$. If it exists at smaller values then $p$, this is a *subcritical Pitchfork* bifurcation. In the other case, this is a *supercritical Pitchfork* bifurcation.

## Normal form computation

The reduced equation (E) can be automatically computed as follows

```julia
computeNF1d(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, options::NewtonPar ; 
	δ = 1e-8, nev = 5, Jt = nothing, verbose = false)
```

where `dF, d2F,d3F` are the differentials of `F`. `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. The above call returns a point with information needed to compute the bifurcated branch.

```julia
mutable struct SimpleBranchPoint{Tv, T, Tevl, Tevr, Tnf} <: BranchPoint
	"bifurcation point"
	x0::Tv

	"Parameter value at the bifurcation point"
	p::T

	"Right eigenvector(s)"
	ζ::Tevr

	"Left eigenvector(s)"
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation point"
	type::Symbol
end
```

!!! info "Note"
    You should not need to call `computeNF1d` except if you want to have the full information about the branch point. Indeed, the call in the next section do it internally.

## Automatic branch switching

You can perform automatic branch switching by calling `continuation` with the following options:

```julia
continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ;
	Jt = nothing, δ = 1e-8, kwargs...)
```

where `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled. This call will compute the branch bifurcating from the `ind_bif `th bifurcation point in `br`. An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

