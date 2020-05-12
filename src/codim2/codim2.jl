"""
	newton(F, J, br::ContResult, ind_bif::Int64, par, lens::Lens, eigenvec, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm)

This function turns an initial guess for a Fold/Hopf point into a solution to the Fold/Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `par` parameters used for the vector field
- `lens` parameter axis used to locate the Fold point.
- `eigenvec` guess for the 0 eigenvector
- `options::NewtonPar`

# Optional arguments:
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method.
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
"""
function newton(F, J, br::ContResult, ind_bif::Int64, par, lens::Lens; Jt = nothing, d2F = nothing, normN = norm, options = br.contparams.newtonOptions, kwargs...)
	if length(br.bifpoint) > 0 && br.bifpoint[ind_bif].type == :hopf
		return newtonHopf(F, J, br, ind_bif, par, lens; Jt = Jt, d2F = d2F, options = options, kwargs...)
	elseif br.foldpoint[ind_bif].type == :fold
		return newtonFold(F, J, br, ind_bif, par, lens; Jt = Jt, d2F = d2F, options = options, kwargs...)
	end
	@error "Bifurcation type $(br[ind_bif].type) not yet handle for codim2 newton / continuation"
end


"""
	continuation(F, J, hopfpointguess::BorderedArray, par, lens1::Lens, lens2::Lens, options::ContinuationPar ; Jt = nothing, d2F = nothing, kwargs...)

codim 2 continuation of Hopf points. This function turns an initial guess for a Fold/Hopf point into a curve of Fold/Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `options` parameters for continuation

# Optional arguments:

- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = p -> ((x, p, v1, v2) -> d2F(x, p, v1, v2))` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`.

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""

function continuation(F, J, br::ContResult, ind_bif::Int64, par, lens1::Lens, lens2::Lens, options_cont::ContinuationPar ; Jt = nothing, d2F = nothing, kwargs...)
	if length(br.bifpoint) > 0 && br.bifpoint[ind_bif].type == :hopf
		return continuationHopf(F, J, br, ind_bif, par, lens1, lens2, options_cont; Jt = Jt, d2F = d2F, kwargs...)
	elseif br.foldpoint[ind_bif].type == :fold
		return continuationFold(F, J, br, ind_bif, par, lens1, lens2, options_cont; Jt = Jt, d2F = d2F, kwargs...)
	end
	@error "Bifurcation type $(br[ind_bif].type) not yet handle for codim2 newton / continuation"
end
