abstract type ProblemMinimallyAugmented end

"""
$(SIGNATURES)

This function turns an initial guess for a Fold/Hopf point into a solution to the Fold/Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `lens` parameter axis used to locate the Fold/Hopf point.
- `options::NewtonPar`

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric (for Fold)
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of Matrix / Sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newton(F, J, br::AbstractBranchResult, ind_bif::Int64; Jᵗ = nothing, d2F = nothing, normN = norm, options = br.contparams.newtonOptions, startWithEigen = false, issymmetric = false, kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	if br.specialpoint[ind_bif].type == :hopf
		d2Fc = isnothing(d2F) ? nothing : (x,p,dx1,dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x,p,_dx1,_dx2))(dx1,dx2)
		return newtonHopf(F, J, br, ind_bif; Jᵗ = Jᵗ, d2F = d2Fc, normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	else
		return newtonFold(F, J, br, ind_bif; issymmetric = issymmetric, Jᵗ = Jᵗ, d2F = d2F, normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	end
end

"""
$(SIGNATURES)

codim 2 continuation of Fold / Hopf points. This function turns an initial guess for a Fold/Hopf point into a curve of Fold/Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [`continuation`](@ref)
- `ind_bif` bifurcation index in `br`
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `options_cont` arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric (for Fold)
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = (x, p, v1, v2) -> d2F(x, p, v1, v2)` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a,b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences.
"""
function continuation(F, J,
				br::AbstractBranchResult, ind_bif::Int64,
				lens2::Lens, options_cont::ContinuationPar ;
				startWithEigen = false,
				issymmetric = false,
				Jᵗ = nothing,
				d2F = nothing,
				d3F = nothing,
				kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	if br.specialpoint[ind_bif].type == :hopf
		# redefine the multilinear form to accept complex arguments
		d2Fc = isnothing(d2F) ? nothing : (x,p,dx1,dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x,p,_dx1,_dx2))(dx1,dx2)
		d3Fc = isnothing(d3F) ? nothing : (x,p,dx1,dx2,dx3) -> TrilinearMap((_dx1, _dx2, _dx3) -> d3F(x,p,_dx1,_dx2,_dx3))(dx1,dx2,dx3)
		return continuationHopf(F, J, br, ind_bif, lens2, options_cont; Jᵗ = Jᵗ, d2F = d2Fc, d3F = d3Fc, startWithEigen = startWithEigen, kwargs...)
	else
		return continuationFold(F, J, br, ind_bif, lens2, options_cont; issymmetric = issymmetric, Jᵗ = Jᵗ, d2F = d2F, startWithEigen = startWithEigen, kwargs...)
	end
end
