using KrylovKit, Parameters, RecursiveArrayTools

function FoldPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index][1] == :fold "The index provided does not refer to a Fold point"
	bifpoint = br.bifpoint[index]
	eigRes   = br.eig
	return vcat(bifpoint[5], bifpoint[3])
end

#################################################################################################### Method using Minimally Augmented formulation

struct FoldProblemMinimallyAugmented{vectype, S <: LinearSolver}
    F::Function 		# Function F(x, p) = 0
    J::Function 		# Jacobian of F wrt x
    Jadjoint::Function	# Adjoint of the Jacobian of F
    a::vectype			# close to null vector of J^T
    b::vectype			# close to null vector of J
    linsolve::S
end

function (fp::FoldProblemMinimallyAugmented{vectype, S})(x, p) where {vectype, S <: LinearSolver}
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p parameter for which the jacobian is singular
    # These are minimally augmented turning point equations
    # The jacobian will be solved using a minimally augmented method
    a = fp.a
    b = fp.b

    # we solve Jv + a σ1 = 0 with <b, v> = n
    n = 1.0
    v_ = fp.linsolve(fp.J(x, p), -a )[1]
    σ1 = n / dot(b, v_)
    v = σ1 * v_

    # # # we solve J'w + b σ2 = 0 with <a, w> = n
    # w_ = fp.linsolve(fp.Jadjoint(x, p), -b )[1]
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

function (fp::FoldProblemMinimallyAugmented{vectype, S})(u::Vector) where {vectype, S <: LinearSolver}
	res = fp(u[1:end-1], u[end])
	return vcat(res[1], res[2])
end

function (fp::FoldProblemMinimallyAugmented{vectype, S})(x::BorderedVector{vectype, T}) where {vectype, S <: LinearSolver, T}
	res = fp(x.u, x.p)
	return BorderedVector(res[1], res[2])
end

# Method to solve the associated linear system
struct FoldLinearSolveMinAug <: LinearSolver end

function (foldl::FoldLinearSolveMinAug)(x, p, pbMA::FoldProblemMinimallyAugmented, du, debug_ = false)
	# We solve Jfold⋅res = du where Jfold = d_xF(x,p)
    # the Jacobian J is expressed at (x, p)
    # the Jacobian expression of the Fold problem is [J dpF ; σx σp]
    ############### Extraction of function names #################

    F = pbMA.F
    J = pbMA.J
    Jadjoint = pbMA.Jadjoint
    a = pbMA.a
    b = pbMA.b

	# we solve Jv + a σ1 = 0 with <b, v> = n
    n = 1.0
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

    # the case of sigma_x is a bit more involved
    σx = zero(x)
    e  = zero(x)

    # this is the part which does not work if x is not AbstractVector. Maybe use CartesianIndices
    for ii in CartesianIndices(x)
        e .= 0 .* e
        e[ii] = 1.0
        d2Fve = (apply(J(x + ϵ2 * e, p), v) - apply(J(x - ϵ2 * e, p), v)) / (2ϵ2)
        σx[ii] = -dot(w, d2Fve) / n
    end

    ########## Resolution of the bordered linear system ########
    dX, dsig, it = linearBorderedSolver(J(x, p), dpF, σx, σp, du[1:end-1], du[end], pbMA.linsolve)

	if debug_
    	return vcat(dX, dsig), true, sum(it), σx, [J(x, p) dpF ; σx' σp]
	else
		return vcat(dX, dsig), true, sum(it), σx
	end

end

function (foldl::FoldLinearSolveMinAug)(Jfold::Tuple{Vector, FoldProblemMinimallyAugmented}, du::Vector{T}, debug_ = false) where {T}
	return foldl(Jfold[1][1:end-1],
				 Jfold[1][end],
				 Jfold[2],
				 du, debug_)
end

################################################################################################### Newton / Continuation functions
"""
This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `(x, p)-> F(x, p)` where `p` is the parameter associated to the Fold point
- `J = (x, p)-> d_xF(x, p)` associated jacobian
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian, it should be implenented otherwise
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `Vector`
- `eigenvec` guess for the 0 eigenvector
- `options::NewtonPar`
"""
function newtonFold(F::Function, J, Jt, foldpointguess::AbstractVector, eigenvec::AbstractVector, options::NewtonPar)

	foldvariable = FoldProblemMinimallyAugmented(
						(x, p) ->  F(x, p),
						(x, p) ->  J(x, p),
						(x, p) -> Jt(x, p),
						eigenvec,
						eigenvec,
						options.linsolve)

	foldPb = u -> foldvariable(u)

	# Jacobian for the Fold problem
	Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb))

	opt_fold = @set options.linsolve = FoldLinearSolveMinAug()

	# solve the Fold equations
	return newton(x ->  foldPb(x),
						x -> Jac_fold_MA(x, foldvariable),
						foldpointguess,
						opt_fold)
end

newtonFold(F::Function, J, foldpointguess::AbstractVector, eigenvec::AbstractVector, options::NewtonPar) = newtonFold(F, J, (x, p)->transpose(J(x, p)), foldpointguess, eigenvec, options)

"""
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows `newtonFold(F, J, Jt, br::ContResult, index::Int64, options)` where the parameters are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.
"""
function newtonFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, options::NewtonPar)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]

	eigenvec = bifpt[end-1]

	# solve the Fold equations
	outfold, hist, flag =  newtonFold(F, J, Jt, foldpointguess, eigenvec, options)

	return outfold, hist, flag
end

newtonFold(F::Function, J, br::ContResult, ind_fold::Int64, options::NewtonPar) = newtonFold(F, J, (x, p)->transpose(J(x, p)), br, ind_fold, options)

newtonFold(F::Function, br::ContResult, ind_fold::Int64, options::NewtonPar) = newtonFold(F, (x0, p) -> finiteDifferences(x -> F(x, p), x0), br, ind_fold, options)

"""
codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p1, p2) -> F(x, p1, p2)` where `p` is the parameter associated to the Fold point
- `J = (x, p1, p2) -> d_xF(x, p1, p2)` associated jacobian
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `Vector`
- `p2` parameter p2 for which foldpointguess is a good guess
- `eigenvec` guess for the 0 eigenvector at p1_0
- `options::NewtonPar`
"""
function continuationFold(F::Function, J, Jt, foldpointguess::AbstractVector, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ; kwargs...)
	@warn "Bad way it creates a struct for every p2"
	# Jacobian for the Fold problem
	Jac_fold_MA(u0::Vector, p2, pb) = (return (u0, pb))

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

	opt_fold_cont = @set options_cont.newtonOptions.linsolve = FoldLinearSolveMinAug()

	# solve the Fold equations
	return continuation((x, p2) -> foldPb(x, p2),
						(x, p2) -> Jac_fold_MA(x, p2, foldvariable(p2)),
						foldpointguess, p2_0,
						opt_fold_cont,
						plot = true,
						printsolution = u -> u[end],
						plotsolution = (x;kwargs...)->(xlabel!("p2", subplot=1);ylabel!("p1", subplot=1)  ) ; kwargs...)
end

continuationFold(F::Function, J, foldpointguess::AbstractVector, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, J, (x, p1, p2)->transpose(J(x, p1, p2)), foldpointguess, p2_0, eigenvec, options_cont ; kwargs...)

function continuationFold(F::Function, foldpointguess::AbstractVector, p2_0::Real, eigenvec::AbstractVector, options::ContinuationPar ; kwargs...)
	return continuationFold(F::Function,
							(x0, p) -> finiteDifferences(x -> F(x, p), x0),
							foldpointguess, p2_0,
							eigenvec,
							options ; kwargs...)
end

"""
Simplified call for continuation of Fold point. More precisely, the call is as follows `continuationFold(F, J, Jt, br::ContResult, index::Int64, options)` where the parameters are as for `continuationFold` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.
"""
function continuationFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.bifpoint[ind_fold]
	eigenvec = bifpt[end-1]
	return continuationFold(F, J, Jt, foldpointguess, p2_0, eigenvec, options_cont ; kwargs...)
end

continuationFold(F::Function, J, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, J, (x, p1, p2)->transpose(J(x, p1, p2)), br, ind_fold, p2_0, options_cont::ContinuationPar ; kwargs...)

continuationFold(F::Function, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationFold(F, (x0, p1, p2) -> finiteDifferences(x -> F(x, p1, p2), x0), br, ind_fold, p2_0, options_cont::ContinuationPar ; kwargs...)
