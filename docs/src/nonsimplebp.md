# Non-simple branch point

!!! unknown "References"
    The general method is exposed in Golubitsky, Martin, David G Schaeffer, and Ian Stewart. **Singularities and Groups in Bifurcation Theory**. New York: Springer-Verlag, 1985, VI.1.d page 295
    
!!! tip "Example"
    An example of use of the methods presented here is provided in [2d generalized Bratu–Gelfand problem](@ref).    

We expose our method to study *non-simple branch points*. Such branch point $(x_0,p_0)$ for the problem $F(x,p)=0$ satisfies $d=\dim \ker dF(x_0,p_0) > 1$ and the eigenvalues have zero imaginary part. At such point, we can apply **Lyapunov-Schmidt** reduction to transform the initial problem in large dimensions to a $d$-dimensional polynomial equation, called the **reduced equation**.

More precisely, it is possible to write $x = u + v$ where $u\in \ker dF(x_0,p_0)$ and $v\approx 0$ belongs to a vector space complement of $\ker dF(x_0,p_0)$. It can be shown that $u$ solves $\Phi(u,\delta p)=0$ with $\Phi(u,\delta p) = (I-\Pi)F(u+\psi(u,\delta p),p_0+\delta p)$ where $\psi$ is known implicitly and $\Pi$ is the spectral projector on $\ker dF(x_0,p_0)$. Fortunately, one can compute the Taylor expansion of $\Phi$ up to order 3. Computing the bifurcation diagram of this $d$-dimensional multivariate polynomials can be done using brute force methods.

Once the zeros of $\Phi$ have been located, we can use them as initial guess for [`continuation`](@ref) but for the original $F$ !!


## Reduced equation computation

The reduced equation (E) can be automatically computed as follows

```julia
computeNormalForm(F, dF, d2F, d3F, br::ContResult, ind_bif::Int ; δ = 1e-8,
	nev = 5, Jᵗ = nothing, verbose = false, ζs = nothing, lens = br.param_lens)
```

where `dF, d2F,d3F` are the differentials of `F`. `br` is a branch computed after a call to [`continuation`](@ref) with detection of bifurcation points enabled and `ind_bif` is the index of the bifurcation point on the branch `br`. The above call returns a point with information needed to compute the bifurcated branch. For more information about the optional parameters, we refer to [`computeNormalForm`](@ref). It returns a point with all requested information:

```julia
mutable struct NdBranchPoint{Tv, T, Tevl, Tevr, Tnf} <: BranchPoint
	"bifurcation point"
	x0::Tv

	"Parameter value at the bifurcation point"
	p::T

	"Right eigenvectors"
	ζ::Tevr

	"Left eigenvectors"
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation point"
	type::Symbol
end
```

## Using the Reduced equation
Once a branch point has been computed `bp = computeNormalForm(...)`, you can do all sort of things. 

- For example, quoted from the file `test/testNF.jl`, you can print the 2d reduced equation as follows:

```julia
julia> BifurcationKit.nf(bp2d)
2-element Array{String,1}:
 " + (3.23 + 0.0im) * x1 * p + (-0.123 + 0.0im) * x1^3 + (-0.234 + 0.0im) * x1 * x2^2"
 " + (-0.456 + 0.0im) * x1^2 * x2 + (3.23 + 0.0im) * x2 * p + (-0.123 + 0.0im) * x2^3"
``` 

- You can evaluate the reduced equation as `bp2d(Val(:reducedForm), rand(2), 0.2)`. This can be used to find all the zeros of the reduced equation by sampling on a grid. 

- Finally, given a $d$-dimensional vector $x$ and a parameter $\delta p$, you can can have access to an initial guess $u$ (see above) by calling `bp2d(rand(2), 0.1)`
