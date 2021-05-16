# Simple Hopf point


At a Hopf branch point $(x_0,p_0)$ for the problem $F(x,p)=0$, we have $\Sigma\ dF(x_0,p_0) = \{\pm i\omega \},\ \omega > 0$. At such point, we can compute the **normal form** to transform the initial Cauchy problem

$$\dot x = F(x,p)$$

in large dimensions to a **complex** polynomial vector field ($\delta p\equiv p-p_0$): 

$$\dot z = z\left(a \cdot\delta p + i\omega + b|z|^2\right)\quad\text{(E)}$$

whose solutions give access to the solutions of the Cauchy problem in a neighborhood of $(x,p)$.

More precisely, if $J \equiv dF(x_0,p_0)$, then we have $J\zeta = i\omega\zeta$ and $J\bar\zeta = -i\omega\bar\zeta$ for some complex eigenvector $\zeta$. It can be shown that $x(t) \approx x_0 + 2\Re(z(t)\zeta)$ when $p=p_0+\delta p$.


## Normal form computation

The normal form (E) is automatically computed as follows

```julia
computeNormalForm(F, dF, d2F, d3F, br::ContResult, ind_bif::Int ; δ = 1e-8,
	nev = 5, Jᵗ = nothing, verbose = false, ζs = nothing, lens = br.param_lens)
```

where `dF, d2F,d3F` are the differentials of `F`. `br` is a branch computed after a call to `continuation` with detection of bifurcation points enabled and `ind_bif` is the index of the bifurcation point on the branch `br. The above call returns a point with information needed to compute the bifurcated branch. For more information about the optional parameters, we refer to [`computeNormalForm`](@ref). The above call returns a point with information needed to compute the bifurcated branch.

```julia
mutable struct Hopf{Tv, T, Tω, Tevr, Tevl, Tnf} <: BifurcationPoint
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
    You should not need to call `computeNormalForm ` except if you need the full information about the branch point. 
