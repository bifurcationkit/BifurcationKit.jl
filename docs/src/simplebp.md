# Simple bifurcation branch point

!!! unknown "References"
    The general method is exposed in Golubitsky, Martin, David G Schaeffer, and Ian Stewart. **Singularities and Groups in Bifurcation Theory**. New York: Springer-Verlag, 1985, VI.1.d page 295

A simple branch point $(x_0,p_0)$ for the problem $F(x,p)=0$ satisfies $\dim \ker dF(x_0,p_0) = 1$. At such point, we can apply **Lyapunov-Schmidt** reduction to transform the initial problem in large dimensions to a **scalar** polynomial ($\delta p \equiv p-p_0$): 

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
computeNormalForm(F, dF, d2F, d3F, br::ContResult, ind_bif::Int ; δ = 1e-8,
	nev = 5, Jᵗ = nothing, verbose = false, ζs = nothing, lens = br.param_lens)
```

where `dF, d2F,d3F` are the differentials of `F`. `br` is a branch computed after a call to [`continuation`](@ref) with detection of bifurcation points enabled and `ind_bif` is the index of the bifurcation point on the branch `br`. The above call returns a point with information needed to compute the bifurcated branch. For more information about the optional parameters, we refer to [`computeNormalForm`](@ref). The result returns the following:

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
    You should not need to call `computeNormalForm` except if you need the full information about the branch point.
