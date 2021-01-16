"""
$(SIGNATURES)

This function turns an initial guess for a Fold/Hopf point into a solution to the Fold/Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `par` parameters used for the vector field
- `lens` parameter axis used to locate the Fold/Hopf point.
- `options::NewtonPar`

# Optional arguments:
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of Matrix / Sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver
"""
function newton(F, J, br::AbstractBranchResult, ind_bif::Int64, par, lens::Lens; Jᵗ = nothing, d2F = nothing, normN = norm, options = br.contparams.newtonOptions, kwargs...)
	if length(br.bifpoint) > 0 && br.bifpoint[ind_bif].type == :hopf
		return newtonHopf(F, J, br, ind_bif, par, lens; Jᵗ = Jᵗ, d2F = d2F, normN = normN, options = options, kwargs...)
	elseif br.foldpoint[ind_bif].type == :fold
		return newtonFold(F, J, br, ind_bif, par, lens; Jᵗ = Jᵗ, d2F = d2F, normN = normN, options = options, kwargs...)
	end
	@error "Bifurcation type $(br[ind_bif].type) not yet handled for codim2 newton / continuation"
end


"""
$(SIGNATURES)

codim 2 continuation of Fold / Hopf points. This function turns an initial guess for a Fold/Hopf point into a curve of Fold/Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `options_cont` arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:

- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = p -> ((x, p, v1, v2) -> d2F(x, p, v1, v2))` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`.

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""

function continuation(F, J, br::AbstractBranchResult, ind_bif::Int64, par, lens1::Lens, lens2::Lens, options_cont::ContinuationPar ; Jᵗ = nothing, d2F = nothing, kwargs...)
	if length(br.bifpoint) > 0 && br.bifpoint[ind_bif].type == :hopf
		return continuationHopf(F, J, br, ind_bif, par, lens1, lens2, options_cont; Jᵗ = Jᵗ, d2F = d2F, kwargs...)
	elseif br.foldpoint[ind_bif].type == :fold
		return continuationFold(F, J, br, ind_bif, par, lens1, lens2, options_cont; Jᵗ = Jᵗ, d2F = d2F, kwargs...)
	end
	@error "Bifurcation type $(br[ind_bif].type) not yet handled for codim2 newton / continuation"
end
