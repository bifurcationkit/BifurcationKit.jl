"""
For an initial guess from the index of a Fold bifurcation point located in ContResult.bifpoint, returns a point which will be refined using `newtonFold`.
"""
function FoldPoint(br::ContResult, index::Int64)
	@assert br.foldpoint[index].type == :fold "The provided index does not refer to a Fold point"
	bifpoint = br.foldpoint[index]
	return BorderedArray(_copy(bifpoint.u), bifpoint.param)
end

####################################################################################################
# Method using Minimally Augmented formulation
@with_kw struct FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver}
	F::TF								# Function F(x, p) = 0
	J::TJ								# Jacobian of F wrt x
	Jadjoint::TJa						# Adjoint of the Jacobian of F
	d2F::Td2f = nothing					# Hessian of F
	a::vectype							# close to null vector of J^T
	b::vectype							# close to null vector of J
	linsolver::S						# linear solver
	linsolverAdjoint::Sa = linsolver	# linear solver for the jacobian adjoint
	linbdsolver::Sbd					# bordered linear solver
end

hasHessian(pb::FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S, Sa, Sbd}) where {TF, TJ, TJa, Td2f, vectype, S, Sa, Sbd} = Td2f != Nothing

hasAdjoint(pb::FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S, Sa, Sbd}) where {TF, TJ, TJa, Td2f, vectype, S, Sa, Sbd} = TJa != Nothing

FoldProblemMinimallyAugmented(F, J, Ja, a, b, linsolve) = FoldProblemMinimallyAugmented(F, J, Ja, nothing, a, b, linsolve, linsolve, BorderingBLS(linsolve))

FoldProblemMinimallyAugmented(F, J, Ja, d2F, a, b, linsolve) = FoldProblemMinimallyAugmented(F, J, Ja, d2F, a, b, linsolve, linsolve, BorderingBLS(linsolve))

function (fp::FoldProblemMinimallyAugmented)(x::vectype, p::T) where {vectype, T}
	# These are the equations of the minimally augmented (MA) formulation of the Fold bifurcation point
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter at which the jacobian is singular
	# The jacobian of the MA problem is solved with a bordering method
	a = fp.a
	b = fp.b

	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
	n = T(1)
	v, _, _ = fp.linsolver(fp.J(x, p), a)
	bv = dot(b, v)
	bv == T(0) && @error "Error when using Minimally Augmented formulation for Fold bifurcation. The dot product should non-zero."

	σ1 = -n / bv
	return fp.F(x, p), σ1
end

function (foldpb::FoldProblemMinimallyAugmented)(x::BorderedArray)
	res = foldpb(x.u, x.p)
	return BorderedArray(res[1], res[2])
end

# Struct to invert the jacobian of the fold MA problem.
struct FoldLinearSolveMinAug <: AbstractLinearSolver; end

function foldMALinearSolver(x, p::T, pbMA::FoldProblemMinimallyAugmented,
							duu, dup;
							debug_::Bool = false) where T
	# The jacobian should be passed as a tuple as Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, d2F::Bool))
	# The Jacobian J of the vector field is expressed at (x, p)
	# We solve here Jfold⋅res = du := [duu, dup]
	# The Jacobian expression of the Fold problem is Jfold = [J dpF ; σx σp] where σx := ∂_xσ
	# We recall the expression of σx = -< w, d2F(x,p)[v, x2]> where (w, σ2) is solution of J'w + b σ2 = 0 with <a, w> = n
	############### Extraction of function names #################
	# N = length(du) - 1

	F = pbMA.F
	J = pbMA.J

	d2F = pbMA.d2F
	a = pbMA.a
	b = pbMA.b

	# we define the following jacobian. It is used at least 3 times below. This avoid doing 3 times the possibly costly building of J(x, p)
	J_at_xp = J(x, p)

	# we do the following to avoid computing J_at_xp twice in case pbMA.Jadjoint is not provided
	if hasAdjoint(pbMA)
		JAd_at_xp = pbMA.Jadjoint(x, p)
	else
		JAd_at_xp = transpose(J_at_xp)
	end

	n = T(1)
	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J\a>
	v = pbMA.linsolver(J_at_xp, a)[1]
	σ1 = -n / dot(b, v)
	rmul!(v, -σ1)

	# we solve J'w + b σ2 = 0 with <a, w> = n
	# the solution is w = -σ2 J'\b with σ2 = -n/<a, J'\b>
	w = pbMA.linsolverAdjoint(JAd_at_xp, b)[1]
	σ2 = -n / dot(a, w)
	rmul!(w, -σ2)

	δ = T(1e-8)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	################### computation of σx σp ####################
	dpF = minus(F(x, p + ϵ1), F(x, p - ϵ1)); rmul!(dpF, T(1) / T(2ϵ1))
	dJvdp = minus(apply(J(x, p + ϵ3), v), apply(J(x, p - ϵ3), v)); rmul!(dJvdp, T(1) / T(2ϵ3))
	σp = -dot(w, dJvdp) / n

	if hasHessian(pbMA) == false
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions
		prod(size(x)) > 1e4 && @warn "You might want to pass the Hessian, you are using finite differences with $(prod(size(x))) unknowns to compute a gradient"
		# we first need to compute the value of ∂_xσ written σx
		σx = zero(x)
		e  = zero(x)

		# this is the part which does not work if x is not AbstractArray. We use CartesianIndices to support AbstractArray as a type for the solution we are looking for
		for ii in CartesianIndices(x)
			e[ii] = T(1)
			# d2Fve := d2F(x,p)[v,e]
			d2Fve = (apply(J(x + ϵ2 * e, p), v) - apply(J(x - ϵ2 * e, p), v)) / T(2ϵ2)
			σx[ii] = -dot(w, d2Fve) / n
			e[ii] = T(0)
		end

		########## Resolution of the bordered linear system ########
		dX, dsig, flag, it = pbMA.linbdsolver(J_at_xp, dpF, σx, σp, duu, dup)

	else
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is known analytically. Much faster than the previous case

		# we solve it here instead of calling linearBorderedSolver because this removes the need to pass the linear form associated to σx
		x1, x2, _, it = pbMA.linsolver(J_at_xp, duu, dpF)

		d2Fv = d2F(x, p, x1, v)
		σx1 = -dot(w, d2Fv ) / n

		copyto!(d2Fv, d2F(x, p, x2, v))
		σx2 = -dot(w, d2Fv ) / n

		dsig = (dup - σx1) / (σp - σx2)

		# dX = x1 .- dsig .* x2
		dX = copyto!(similar(x1), x1); axpy!(-dsig, x2, dX)
	end

	if debug_
		return dX, dsig, true, sum(it), 0., [J(x, p) dpF ; σx' σp]
	else
		return dX, dsig, true, sum(it), 0.
	end
end

function (foldl::FoldLinearSolveMinAug)(Jfold, du::BorderedArray{vectype, T}, debug_::Bool = false) where {vectype, T}
	out =  foldMALinearSolver(Jfold[1].u,
				 Jfold[1].p,
				 Jfold[2],
				 du.u, du.p;
				 debug_ = debug_)

	if debug_ == false
		return BorderedArray(out[1], out[2]), out[3], out[4], out[5]
	else
		return BorderedArray(out[1], out[2]), out[3], out[4], out[5], out[6]
	end
end

################################################################################################### Newton / Continuation functions
"""
	`newtonFold(F, J, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm)`

This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is the parameter associated to the Fold point
- `dF  = (x, p) -> d_xF(x, p)` associated jacobian
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `BorderedArray` as given by the function FoldPoint
- `eigenvec` guess for the 0 eigenvector
- `options::NewtonPar`

Optional arguments:
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method.
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function newtonFold(F, J, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm) where {T, vectype}
	foldvariable = FoldProblemMinimallyAugmented(
		F,
		J,
		Jt,
		d2F,
		_copy(eigenvec), #copy(eigenvec),
		_copy(eigenvec), #copy(eigenvec),
		options.linsolver)

	foldPb = u -> foldvariable(u)

	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb))

	opt_fold = @set options.linsolver = FoldLinearSolveMinAug()

	# solve the Fold equations
	return newton(x ->  foldPb(x),
				x -> Jac_fold_MA(x, foldvariable),
				foldpointguess,
				opt_fold, normN = normN)
end

"""
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

	`newtonFold(F, J, br::ContResult, index::Int64, options::NewtonPar; Jt = nothing, d2F = nothing)`

where the optional argument `Jt` is the jacobian transpose and the Hessian is `d2F`. The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function newtonFold(F, J, br::ContResult, ind_fold::Int64, options::NewtonPar;Jt = nothing, d2F = nothing, kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.foldpoint[ind_fold]
	eigenvec = bifpt.tau

	# solve the Fold equations
	return newtonFold(F, J, foldpointguess, eigenvec, options; Jt = Jt, d2F = d2F, kwargs...)
end


"""
Codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) ->	F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `BorderedArray` as given by the function FoldPoint
- `p2` parameter p2 for which `foldpointguess` is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`

Optional arguments:

- `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` associated jacobian transpose
- `d2F = p2 -> ((x, p1, v1, v2) -> d2F(x, p1, p2, v1, v2))` this is the hessian of `F` computed at `(x, p1, p2)` and evaluated at `(v1, v2)`.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function continuationFold(F, J, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...) where {T,vectype}
	# Bad way it creates a struct for every p2?
	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb) = (return (u0, pb))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	if isnothing(Jt)
		foldvariable = p2 -> FoldProblemMinimallyAugmented(
			(x, p1) ->  F(x, p1, p2),
			(x, p1) ->  J(x, p1, p2),
			nothing,
			d2F(p2),
			_copy(eigenvec), #copy(eigenvec),
			_copy(eigenvec), #copy(eigenvec),
			options_newton.linsolver)
		foldPb = (u, p2) -> foldvariable(p2)(u)
	else
		foldvariable = p2 -> FoldProblemMinimallyAugmented(
			(x, p1) ->  F(x, p1, p2),
			(x, p1) ->  J(x, p1, p2),
			(x, p1) -> Jt(x, p1, p2),
			d2F(p2),
			_copy(eigenvec), #copy(eigenvec),
			_copy(eigenvec), #copy(eigenvec),
			options_newton.linsolver)
		foldPb = (u, p2) -> foldvariable(p2)(u)
	end

	opt_fold_cont = @set options_cont.newtonOptions.linsolver = FoldLinearSolveMinAug()

	# solve the Fold equations
	return continuation(
		(x, p2) -> foldPb(x, p2),
		(x, p2) -> Jac_fold_MA(x, foldvariable(p2)),
		foldpointguess, p2_0,
		opt_fold_cont,
		printSolution = (u, p) -> u.p,
		plotSolution = (x; kwargs...) -> (xlabel!("p2", subplot=1); ylabel!("p1", subplot=1)  ); kwargs...)
end

"""
Simplified call for continuation of Fold point. More precisely, the call is as follows `continuationFold(F, J, br::ContResult, index::Int64, options; Jt = nothing, d2F = p2 -> nothing)` where the parameters are as for `continuationFold` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function continuationFold(F, J, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.foldpoint[ind_fold]
	eigenvec = bifpt.tau
	return continuationFold(F, J, foldpointguess, p2_0, eigenvec, options_cont ;Jt = Jt, d2F = d2F, kwargs...)
end
