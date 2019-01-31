using KrylovKit, Parameters, RecursiveArrayTools

function FoldPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index][1] == :fold "The index provided does not refer to a Fold point"
	bifpoint = br.bifpoint[index]
	return vcat(bifpoint[5], bifpoint[3])
end

#################################################################################################### Method using Minimally Augmented formulation

struct FoldProblemMinimallyAugmented{vectype, S <: LinearSolver}
    F 					# Function F(x, p) = 0
    J 					# Jacobian of F wrt x
    Jadjoint			# Adjoint of the Jacobian of F
    a::vectype			# close to null vector of J^T
    b::vectype			# close to null vector of J
    linsolve::S
end

function (fp::FoldProblemMinimallyAugmented{vectype, S})(x::vectype, p) where {vectype, S <: LinearSolver}
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter for which the jacobian is singular
    # The equations are those of minimally augmented formulation of the turning point problem
    # The jacobian of the MA problem is solved with a minimally augmented method
    a = fp.a
    b = fp.b

    # we solve Jv + a σ1 = 0 with <b, v> = n
    n = 1.0
    v_ = fp.linsolve(fp.J(x, p), -a)[1]
    σ1 = n / dot(b, v_)
    v = σ1 * v_

    # # # we solve J'w + b σ2 = 0 with <a, w> = n
    # w_ = fp.linsolve(fp.Jadjoint(x, p), -b)[1]
    # σ2 = n / dot(a, v_)
    # w = σ2 * w_
	#
    # # the constraint is σ = <w, Jv> / n
    # σ = -dot(w, apply(fp.J(x, p), v)) / n
	# #
	# @show σ1 σ2 σ
	# # # we should have σ1 = σ2 = σ

    return fp.F(x, p), σ1
end

function (foldpb::FoldProblemMinimallyAugmented{vectype, S})(u::Vector) where {vectype, S <: LinearSolver}
	res = foldpb(u[1:end-1], u[end])
	return vcat(res[1], res[2])
end

function (foldpb::FoldProblemMinimallyAugmented{vectype, S})(x::BorderedVector{vectype, T}) where {vectype, S <: LinearSolver, T}
	res = foldpb(x.u, x.p)
	return BorderedVector(res[1], res[2])
end

# Method to invert the jacobian of the fold problem. The only parameter which affects inverting the jacobian of the fold MA problem is whether the hessian is known analytically
@with_kw struct FoldLinearSolveMinAug <: LinearSolver
	# whether the jacobian is known analytically
	d2F_is_given = false
 end

function foldMALinearSolver(x, p, pbMA::FoldProblemMinimallyAugmented,
							duu, dup, d2F;
							debug_ = false,
							d2F_is_given = false)
	# We solve Jfold⋅res = du := [duu, dup]
    # the Jacobian J is expressed at (x, p)
    # the Jacobian expression of the Fold problem is [J dpF ; σx σp] where σx :=∂_xσ
    ############### Extraction of function names #################

    F = pbMA.F
    J = pbMA.J
    Jadjoint = pbMA.Jadjoint
    a = pbMA.a
    b = pbMA.b

	n = 1.0
	# we solve Jv + a σ1 = 0 with <b, v> = n
    v_ = pbMA.linsolve(J(x, p), -a)[1]
    σ1 = n / dot(b, v_)
    v = σ1 * v_

    # we solve J'w + b σ2 = 0 with <a, w> = n
    w_ = pbMA.linsolve(Jadjoint(x, p), -b)[1]
    σ2 = n / dot(a, v_)
    w = σ2 * w_

	δ = 1e-8
    ϵ1, ϵ2, ϵ3 = δ, δ, δ
    ################### computation of σx σp ####################
    dpF = (F(x, p + ϵ1)                   - F(x, p - ϵ1)) / (2ϵ1)
    dJvdp = (apply(J(x, p + ϵ3), v) - apply(J(x, p - ϵ3), v)) / (2ϵ3)
    σp = -dot(w, dJvdp) / n

	if d2F_is_given == false
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions

		# we first need to compute the value of ∂_xσ written σx
		σx = zero(x)
		e  = zero(x)

		# this is the part which does not work if x is not AbstractArray. We use CartesianIndices to support AbstractArray as a type for the solution we are looking for
		for ii in CartesianIndices(x)
			e[ii] = 1.0
			# d2Fve := d2F(x,p)[v,e]
			d2Fve = (apply(J(x + ϵ2 * e, p), v) - apply(J(x - ϵ2 * e, p), v)) / (2ϵ2)
			σx[ii] = -dot(w, d2Fve) / n
			e[ii] = 0.0
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

		d2Fv .= d2F(x, p, x2, v)
		σx2 = -dot(w, d2Fv ) / n

		dsig = (dup - σx1) / (σp - σx2)

		dX = x1 .- dsig .* x2
	end

	if debug_
    	return dX, dsig, true, sum(it), 0., [J(x, p) dpF ; σx' σp]
	else
		return dX, dsig, true, sum(it), 0.
	end
end

# mainly for debugging purposes
function (foldl::FoldLinearSolveMinAug)(Jfold, du::vectype, debug_ = false) where {T, vectype <: AbstractVector{T}}
	out =  foldMALinearSolver(Jfold[1][1:end-1],
				 Jfold[1][end],
				 Jfold[2],
				 du[1:end-1], du[end],
				 x -> x; 		# dummy for d2F
				 debug_ = debug_, d2F_is_given = foldl.d2F_is_given)

	if debug_
		return vcat(out[1], out[2]), out[3], out[4], out[5], out[6]
	else
		return vcat(out[1], out[2]), out[3], out[4], out[5]
	end
end

function (foldl::FoldLinearSolveMinAug)(Jfold, du::BorderedVector{vectype, T}, debug_ = false) where {vectype, T}
	out =  foldMALinearSolver(Jfold[1].u,
				 Jfold[1].p,
				 Jfold[2],
				 du.u, du.p,
				 Jfold[3]; 		# this is d2F
				 debug_ = debug_, d2F_is_given = foldl.d2F_is_given)

	if debug_ == false
		return BorderedVector(out[1], out[2]), out[3], out[4], out[5]
	else
		return BorderedVector(out[1], out[2]), out[3], out[4], out[5], out[6]
	end
end

################################################################################################### Newton / Continuation functions
"""
This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is the parameter associated to the Fold point
- `dF  = (x, p) -> d_xF(x, p)` associated jacobian
- `dFt = (x, p) -> transpose(d_xF(x, p))` associated jacobian, it should be implemented in an efficient manner. For matrix-free methods, `tranpose` is not readily available.
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `Vector`
- `eigenvec` guess for the 0 eigenvector
- `options::NewtonPar`
"""
function newtonFold(F, J, Jt, d2F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype}

	foldvariable = FoldProblemMinimallyAugmented(
						(x, p) ->  F(x, p),
						(x, p) ->  J(x, p),
						(x, p) -> Jt(x, p),
						eigenvec,
						eigenvec,
						options.linsolve)

	foldPb = u -> foldvariable(u)

	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, d2F))

	opt_fold = @set options.linsolve = FoldLinearSolveMinAug(d2F_is_given = true)

	# solve the Fold equations
	return newton(x ->  foldPb(x),
						x -> Jac_fold_MA(x, foldvariable),
						foldpointguess,
						opt_fold, normN = normN)
end
 """
call when hessian is unknown, finite differences are then used
 """
function newtonFold(F, J, Jt, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype}

	foldvariable = FoldProblemMinimallyAugmented(
						(x, p) ->  F(x, p),
						(x, p) ->  J(x, p),
						(x, p) -> Jt(x, p),
						eigenvec,
						eigenvec,
						options.linsolve)

	foldPb = u -> foldvariable(u)

	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, x -> x))

	opt_fold = @set options.linsolve = FoldLinearSolveMinAug(d2F_is_given = false)

	# solve the Fold equations
	return newton(x ->  foldPb(x),
						x -> Jac_fold_MA(x, foldvariable),
						foldpointguess,
						opt_fold, normN = normN)
end

newtonFold(F, J, foldpointguess::AbstractVector, eigenvec::AbstractVector, options::NewtonPar; kwargs...) = newtonFold(F, J, (x, p)->transpose(J(x, p)), foldpointguess, eigenvec, options; kwargs...)

"""
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

	`newtonFold(F, J, Jt, br::ContResult, index::Int64, options)`

or

	`newtonFold(F, J, Jt, d2F, br::ContResult, index::Int64, options)`

when the Hessian is known. The parameters are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.
"""
function newtonFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
	foldpointguess = BorderedVector(br.bifpoint[ind_fold][5], br.bifpoint[ind_fold][3])
	bifpt = br.bifpoint[ind_fold]

	eigenvec = bifpt[end-1]

	# solve the Fold equations
	outfold, hist, flag =  newtonFold(F, J, Jt, d2F, foldpointguess, eigenvec, options; kwargs...)

	return outfold, hist, flag
end



function newtonFold(F, J, Jt, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
	foldpointguess = BorderedVector(br.bifpoint[ind_fold][5], br.bifpoint[ind_fold][3])
	bifpt = br.bifpoint[ind_fold]

	eigenvec = bifpt[end-1]

	# solve the Fold equations
	outfold, hist, flag =  newtonFold(F, J, Jt, foldpointguess, eigenvec, options ;kwargs...)

	return outfold, hist, flag
end

newtonFold(F, J, br::ContResult, ind_fold::Int64, options::NewtonPar; kwargs...) = newtonFold(F, J, (x, p) -> transpose(J(x, p)), br, ind_fold, options; kwargs...)

newtonFold(F, br::ContResult, ind_fold::Int64, options::NewtonPar; kwargs...) = newtonFold(F, (x0, p) -> finiteDifferences(x -> F(x, p), x0), br, ind_fold, options; kwargs...)


"""
codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) -> F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` associated jacobian
- `d2F = (x, p1, p2, v1, v2) -> d2F(x, p1, p2, v1, v2)` this is the hessian of `F` computed at `(x, p1, p2)` and evaluated at `(v1, v2)`.
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `Vector`
- `p2` parameter p2 for which foldpointguess is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`
"""
function continuationFold(F, J, Jt, d2F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar) where {T,vectype}
	@warn "Bad way it creates a struct for every p2"
	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb, hess) = (return (u0, pb, hess))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	foldvariable = p2 -> FoldProblemMinimallyAugmented(
						(x, p1) ->  F(x, p1, p2),
						(x, p1) ->  J(x, p1, p2),
						(x, p1) -> Jt(x, p1, p2),
						eigenvec,
						eigenvec,
						options_newton.linsolve)
	foldPb = (u, p2) -> foldvariable(p2)(u)

	opt_fold_cont = @set options_cont.newtonOptions.linsolve = FoldLinearSolveMinAug(d2F_is_given = true)

	# solve the Fold equations
	return continuation((x, p2) -> foldPb(x, p2),
						(x, p2) -> Jac_fold_MA(x, foldvariable(p2), d2F(p2)),
						foldpointguess, p2_0,
						opt_fold_cont,
						plot = true,
						printsolution = u -> u.p,
						plotsolution = (x;kwargs...)->(xlabel!("p2", subplot=1);ylabel!("p1", subplot=1)  ) )
end

"""
codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) -> F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `Vector`
- `p2` parameter p2 for which foldpointguess is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`


!!! warning "Hessian"
    The hessian of `F` in this case is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function continuationFold(F, J, Jt, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ;kwargs...) where {T,vectype}
	@warn "Bad way it creates a struct for every p2"
	# Jacobian for the Fold problem
	Jac_fold_MA(u0::Vector, p2, pb) = (return (u0, pb, x -> x))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	foldvariable = p2 -> FoldProblemMinimallyAugmented(
						(x, p1) ->  F(x, p1, p2),
						(x, p1) ->  J(x, p1, p2),
						(x, p1) -> Jt(x, p1, p2),
						eigenvec,
						eigenvec,
						options_newton.linsolve)
	foldPb = (u, p2) -> foldvariable(p2)(u)

	opt_fold_cont = @set options_cont.newtonOptions.linsolve = FoldLinearSolveMinAug(d2F_is_given = false)

	# solve the Fold equations
	return continuation((x, p2) -> foldPb(x, p2),
						(x, p2) -> Jac_fold_MA(x, p2, foldvariable(p2)),
						foldpointguess, p2_0,
						opt_fold_cont,
						plot = true,
						printsolution = u -> u[end],
						plotsolution = (x;kwargs...)->(xlabel!("p2", subplot=1);ylabel!("p1", subplot=1)  );kwargs... )
end

continuationFold(F, J, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ; kwargs...)  where {T,vectype} = continuationFold(F, J, (x, p1, p2)->transpose(J(x, p1, p2)), foldpointguess, p2_0, eigenvec, options_cont ; kwargs...)

function continuationFold(F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, p2_0::Real, eigenvec::AbstractVector, options::ContinuationPar ; kwargs...)  where {T,vectype}
	return continuationFold(F,
							(x0, p) -> finiteDifferences(x -> F(x, p), x0),
							foldpointguess, p2_0,
							eigenvec,
							options ; kwargs...)
end

"""
Simplified call for continuation of Fold point. More precisely, the call is as follows `continuationFold(F, J, Jt, br::ContResult, index::Int64, options)` where the parameters are as for `continuationFold` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.
"""
function continuationFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	foldpointguess = BorderedVector(br.bifpoint[ind_fold][5], br.bifpoint[ind_fold][3])
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]
	return continuationFold(F, J, Jt, d2F, foldpointguess, p2_0, eigenvec, options_cont ;kwargs...)
end

function continuationFold(F, J, Jt, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]
	return continuationFold(F, J, Jt, foldpointguess, p2_0, eigenvec, options_cont ;kwargs...)
end

continuationFold(F, J, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, J, (x, p1, p2)->transpose(J(x, p1, p2)), br, ind_fold, p2_0, options_cont::ContinuationPar ; kwargs...)

continuationFold(F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, (x0, p1, p2) -> finiteDifferences(x -> F(x, p1, p2), x0), br, ind_fold, p2_0, options_cont::ContinuationPar ; kwargs...)
