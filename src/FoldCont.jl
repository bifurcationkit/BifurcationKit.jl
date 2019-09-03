using Parameters, Setfield

function FoldPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index][1] == :fold "The index provided does not refer to a Fold point"
	bifpoint = br.bifpoint[index]
	return BorderedArray(copy(bifpoint[5]), bifpoint[3])
end

####################################################################################################Method using Minimally Augmented formulation
struct FoldProblemMinimallyAugmented{TF, TJ, TJa, vectype, S <: LinearSolver}
	F::TF				# Function F(x, p) = 0
	J::TJ				# Jacobian of F wrt x
	Jadjoint::TJa		# Adjoint of the Jacobian of F
	a::vectype			# close to null vector of J^T
	b::vectype			# close to null vector of J
	linsolve::S			# linear solver
end

function (fp::FoldProblemMinimallyAugmented{TF, TJ, TJa, vectype, S })(x::vectype, p::T) where {TF, TJ, TJa, vectype, S  <: LinearSolver, T}
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter for which the jacobian is singular
	# The equations are those of minimally augmented formulation of the turning point problem
	# The jacobian of the MA problem is solved with a bordering method
	a = fp.a
	b = fp.b

	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
	n = T(1)
	v, _, _ = fp.linsolve(fp.J(x, p), a)
	bv = dot(b, v)
	bv == T(0) && (@error "Error with Method using Minimally Augmented formulation for Fold")

	σ1 = -n / bv
	return fp.F(x, p), σ1
end

function (foldpb::FoldProblemMinimallyAugmented{TF, TJ, TJa, vectype, S })(x::BorderedArray{vectype, T}) where {TF, TJ, TJa, vectype, S  <: LinearSolver, T}
	res = foldpb(x.u, x.p)
	return BorderedArray(res[1], res[2])
end

# Method to invert the jacobian of the fold problem. The only parameter which affects inverting the jacobian of the fold MA problem is whether the hessian is known analytically
@with_kw struct FoldLinearSolveMinAug <: LinearSolver
	# whether the Hessian is known analytically
	d2F_is_known::Bool = false
 end

function foldMALinearSolver(x, p::T, pbMA::FoldProblemMinimallyAugmented,
							duu, dup, d2F;
							debug_::Bool = false,
							d2F_is_known::Bool = false) where T
	# We solve Jfold⋅res = du := [duu, dup]
	# the Jacobian J is expressed at (x, p)
	# the Jacobian expression of the Fold problem is Jfold = [J dpF ; σx σp] where σx := ∂_xσ
	# we recall the expression of σx = -< w ,d2F(x,p)[v, x2]>
	############### Extraction of function names #################

	F = pbMA.F
	J = pbMA.J
	Jadjoint = pbMA.Jadjoint
	a = pbMA.a
	b = pbMA.b

	n = T(1)
	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
	v = pbMA.linsolve(J(x, p), a)[1]
	σ1 = -n / dot(b, v)
	rmul!(v, -σ1)

	# we solve J'w + b σ2 = 0 with <a, w> = n
	# the solution is w = -σ2 J\b with σ2 = -n/<a, J^{-1}b>
	w = pbMA.linsolve(Jadjoint(x, p), b)[1]
	σ2 = -n / dot(a, w)
	rmul!(w, -σ2)

	δ = T(1e-8)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	################### computation of σx σp ####################
	dpF = minus(F(x, p + ϵ1), F(x, p - ϵ1)); rmul!(dpF, one(p) / T(2ϵ1))
	dJvdp = minus(apply(J(x, p + ϵ3), v), apply(J(x, p - ϵ3), v)); rmul!(dJvdp, one(p) / T(2ϵ3))
	σp = -dot(w, dJvdp) / n

	if d2F_is_known == false
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions

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
		dX, dsig, it = linearBorderedSolver(J(x, p), dpF, σx, σp, duu, dup, pbMA.linsolve)

	else
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is known analytically. Much faster than the previous case

		# we solve it ourself instead of calling linearBorderedSolver. This removes the need to pass the linear form associated to σx
		x1, _, it1 = pbMA.linsolve(J(x, p), duu)
		x2, _, it2 = pbMA.linsolve(J(x, p), dpF)
		it = (it1, it2)

		d2Fv = d2F(x, p, x1, v)
		σx1 = -dot(w, d2Fv ) / n

		copyto!(d2Fv, d2F(x, p, x2, v))
		σx2 = -dot(w, d2Fv ) / n

		dsig = (dup - σx1) / (σp - σx2)

		# dX = x1 .- dsig .* x2
		dX = copy(x1); axpy!(-dsig, x2, dX)
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
				 du.u, du.p,
				 Jfold[3]; 		# -> this is the hessian d2F
				 debug_ = debug_, d2F_is_known = foldl.d2F_is_known)

	if debug_ == false
		return BorderedArray(out[1], out[2]), out[3], out[4], out[5]
	else
		return BorderedArray(out[1], out[2]), out[3], out[4], out[5], out[6]
	end
end

################################################################################################### Newton / Continuation functions
"""
This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is the parameter associated to the Fold point
- `dF  = (x, p) -> d_xF(x, p)` associated jacobian
- `dFt = (x, p) -> transpose(d_xF(x, p))` associated jacobian, it should be implemented in an efficient manner. For matrix-free methods, `tranpose` is not readily available.
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `BorderedArray` as given by the function FoldPoint
- `eigenvec` guess for the 0 eigenvector
- `options::NewtonPar`
"""
function newtonFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; normN = norm, d2F_is_known::Bool = true ) where {T,vectype}
	# rmul!(eigenvec, 1.0 / normN(eigenvec))

	foldvariable = FoldProblemMinimallyAugmented(
		(x, p) ->  F(x, p),
		(x, p) ->  J(x, p),
		(x, p) -> Jt(x, p),
		copy(eigenvec),
		copy(eigenvec),
		options.linsolve)

	foldPb = u -> foldvariable(u)

	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, d2F))

	opt_fold = @set options.linsolve = FoldLinearSolveMinAug(d2F_is_known = d2F_is_known)

	# solve the Fold equations
	return newton(x ->  foldPb(x),
				x -> Jac_fold_MA(x, foldvariable),
				foldpointguess,
				opt_fold, normN = normN)
end


# calls when hessian is unknown, finite differences are then used
newtonFold(F, J, Jt, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype} = newtonFold(F, J, Jt, x -> x, foldpointguess, eigenvec, options; normN = normN, d2F_is_known = false)

newtonFold(F, J, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype} = newtonFold(F, J, (x, p) -> transpose(J(x, p)), foldpointguess, eigenvec, options; normN = normN)

"""
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

	`newtonFold(F, J, Jt, br::ContResult, index::Int64, options::NewtonPar)`

or

	`newtonFold(F, J, Jt, d2F, br::ContResult, index::Int64, options::NewtonPar)`

whether the Hessian d2F is known analytically or not. The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.
"""
function newtonFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]

	# solve the Fold equations
	outfold, hist, flag =  newtonFold(F, J, Jt, d2F, foldpointguess, eigenvec, options; kwargs...)

	return outfold, hist, flag
end

function newtonFold(F, J, Jt, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]

	# solve the Fold equations
	outfold, hist, flag =  newtonFold(F, J, Jt, foldpointguess, eigenvec, options ;kwargs...)

	return outfold, hist, flag
end

newtonFold(F, J, br::ContResult, ind_fold::Int64, options::NewtonPar; kwargs...) = newtonFold(F, J, (x, p) -> transpose(J(x, p)), br, ind_fold, options; kwargs...)

newtonFold(F, br::ContResult, ind_fold::Int64, options::NewtonPar; kwargs...) = newtonFold(F, (x0, p) -> finiteDifferences(x -> F(x, p), x0), br, ind_fold, options; kwargs...)


"""
Codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) ->	F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` associated jacobian
- `d2F = (x, p1, p2, v1, v2) -> d2F(x, p1, p2, v1, v2)` this is the hessian of `F` computed at `(x, p1, p2)` and evaluated at `(v1, v2)`.
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `BorderedArray` as given by the function FoldPoint
- `p2` parameter p2 for which `foldpointguess` is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`
"""
function continuationFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ; d2F_is_known = true, kwargs...) where {T,vectype}
	#TODO Bad way it creates a struct for every p2
	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb, hess) = (return (u0, pb, hess))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	foldvariable = p2 -> FoldProblemMinimallyAugmented(
		(x, p1) ->  F(x, p1, p2),
		(x, p1) ->  J(x, p1, p2),
		(x, p1) -> Jt(x, p1, p2),
		copy(eigenvec),
		copy(eigenvec),
		options_newton.linsolve)
	foldPb = (u, p2) -> foldvariable(p2)(u)
	println("--> Start Fold continuation with Hessian known? = ", d2F_is_known)

	opt_fold_cont = @set options_cont.newtonOptions.linsolve = FoldLinearSolveMinAug(d2F_is_known = d2F_is_known)

	# solve the Fold equations
	return continuation((x, p2) -> foldPb(x, p2),
		(x, p2) -> Jac_fold_MA(x, foldvariable(p2), d2F(p2)),
		foldpointguess, p2_0,
		opt_fold_cont,
		printsolution = u -> u.p,
		plotsolution = (x; kwargs...) -> (xlabel!("p2", subplot=1); ylabel!("p1", subplot=1)  ); kwargs...)
end

"""
codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) -> F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `BorderedArray` as given by the function FoldPoint
- `p2` parameter p2 for which foldpointguess is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`


!!! warning "Hessian"
	The hessian of `F` in this case is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
continuationFold(F, J, Jt, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ;kwargs...) where {T,vectype} = continuationFold(F, J, Jt, x -> x, foldpointguess, p2_0, eigenvec, options_cont ; d2F_is_known = false, kwargs...)


continuationFold(F, J, foldpointguess::BorderedArray{vectype, T}, p2_0::Real, eigenvec, options_cont::ContinuationPar ; kwargs...)  where {T,vectype} = continuationFold(F, J, (x, p1, p2) -> transpose(J(x, p1, p2)), foldpointguess, p2_0, eigenvec, options_cont ; kwargs...)

function continuationFold(F, foldpointguess::BorderedArray{vectype, T}, p2_0::Real, eigenvec, options::ContinuationPar ; kwargs...)  where {T,vectype}
	return continuationFold(F,
		(x0, p) -> finiteDifferences(x -> F(x, p), x0),
		foldpointguess, p2_0,
		eigenvec,
		options ; kwargs...)
end

"""
Simplified call for continuation of Fold point. More precisely, the call is as follows `continuationFold(F, J, Jt, d2F, br::ContResult, index::Int64, options)` where the parameters are as for `continuationFold` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

Simplified calls are also provided but at the cost of using finite differences.
"""
function continuationFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	foldpointguess = BorderedArray(br.bifpoint[ind_fold][5], br.bifpoint[ind_fold][3])
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]
	return continuationFold(F, J, Jt, d2F, foldpointguess, p2_0, eigenvec, options_cont ;d2F_is_known = true, kwargs...)
end

function continuationFold(F, J, Jt, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]
	return continuationFold(F, J, Jt, foldpointguess, p2_0, eigenvec, options_cont; kwargs...)
end

continuationFold(F, J, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, J, (x, p1, p2) -> transpose(J(x, p1, p2)), br, ind_fold, p2_0, options_cont::ContinuationPar; kwargs...)

continuationFold(F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, (x0, p1, p2) -> finiteDifferences(x -> F(x, p1, p2), x0), br, ind_fold, p2_0, options_cont::ContinuationPar; kwargs...)
