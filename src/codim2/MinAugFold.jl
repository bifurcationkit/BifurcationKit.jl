"""
For an initial guess from the index of a Fold bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonFold`.
"""
function FoldPoint(br::AbstractBranchResult, index::Int)
	bptype = br.specialpoint[index].type
	@assert bptype == :bp || bptype == :nd || bptype == :fold "This should be a Fold / BP point"
	specialpoint = br.specialpoint[index]
	return BorderedArray(_copy(specialpoint.x), specialpoint.param)
end
####################################################################################################
@inline getVec(x, ::FoldProblemMinimallyAugmented) = extractVecBLS(x)
@inline getP(x, ::FoldProblemMinimallyAugmented) = extractParBLS(x)

function (fp::FoldProblemMinimallyAugmented)(x, p::T, params) where T
	# These are the equations of the minimally augmented (MA) formulation of the Fold bifurcation point
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter value `<: Real` at which the jacobian is singular
	# The jacobian of the MA problem is solved with a BLS method
	a = fp.a
	b = fp.b
	# update parameter
	par = set(params, getLens(fp), p)
	# ┌      ┐┌  ┐   ┌ ┐
	# │ J  a ││v │ = │0│
	# │ b  0 ││σ1│   │1│
	# └      ┘└  ┘   └ ┘
	# In the notations of Govaerts 2000, a = w, b = v
	# Thus, b should be a null vector of J
	#       a should be a null vector of J'
	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
	n = T(1)
	J = jacobian(fp.prob_vf, x, par)
	σ1 = fp.linbdsolver(J, a, b, T(0), fp.zero, n)[2]
	return residual(fp.prob_vf, x, par), σ1
end

# this function encodes the functional
function (foldpb::FoldProblemMinimallyAugmented)(x::BorderedArray, params)
	res = foldpb(x.u, x.p, params)
	return BorderedArray(res[1], res[2])
end

@views function (foldpb::FoldProblemMinimallyAugmented)(x::AbstractVector, params)
	res = foldpb(x[1:end-1], x[end], params)
	return vcat(res[1], res[2])
end

# Struct to invert the jacobian of the fold MA problem.
struct FoldLinearSolverMinAug <: AbstractLinearSolver; end

function foldMALinearSolver(x, p::T, pb::FoldProblemMinimallyAugmented, par,
							rhsu, rhsp;
							debugArray = nothing) where T
	################################################################################################
	# debugArray is used as a temp to be filled with values used for debugging. If debugArray = nothing, then no debugging mode is entered. If it is AbstractArray, then it is used
	################################################################################################
	# Recall that the functional we want to solve is [F(x,p), σ(x,p)] where σ(x,p) is computed in the function above.
	# The Jacobian J of the vector field is expressed at (x, p)
	# We solve here Jfold⋅res = rhs := [rhsu, rhsp]
	# The Jacobian expression of the Fold problem is
	#           ┌         ┐
	#  Jfold =  │  J  dpF │
	#           │ σx   σp │
	#           └         ┘
	# where σx := ∂_xσ and σp := ∂_pσ
	# We recall the expression of
	#			σx = -< w, d2F(x,p)[v, x2]>
	# where (w, σ2) is solution of J'w + b σ2 = 0 with <a, w> = n
	########################## Extraction of function names ########################################
	a = pb.a
	b = pb.b

	# parameter axis
	lens = getLens(pb)
	# update parameter
	par0 = set(par, lens, p)

	# we define the following jacobian. It is used at least 3 times below. This avoids doing 3 times the (possibly) costly building of J(x, p)
	J_at_xp = jacobian(pb.prob_vf, x, par0)

	# we do the following in order to avoid computing J_at_xp twice in case pb.Jadjoint is not provided
	if isSymmetric(pb.prob_vf)
		JAd_at_xp = J_at_xp
	else
		JAd_at_xp = hasAdjoint(pb) ? jad(pb.prob_vf, x, par0) : transpose(J_at_xp)
	end

	# normalization
	n = T(1)

	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J\a>
	v, σ1, cv, itv = pb.linbdsolver(J_at_xp, a, b, T(0), pb.zero, n)
	~cv && @debug "Linear solver for J did not converge."

	# we solve J'w + b σ2 = 0 with <a, w> = n
	# the solution is w = -σ2 J'\b with σ2 = -n/<a, J'\b>
		w, σ2, _, itw = pb.linbdsolver(JAd_at_xp, b, a, T(0), pb.zero, n)

	δ = getDelta(pb.prob_vf)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	################### computation of σx σp ####################
	################### and inversion of Jfold ####################
	dpF = minus(residual(pb.prob_vf, x, set(par, lens, p + ϵ1)),
				residual(pb.prob_vf, x, set(par, lens, p - ϵ1))); rmul!(dpF, T(1 / (2ϵ1)))
	dJvdp = minus(apply(jacobian(pb.prob_vf, x, set(par, lens, p + ϵ3)), v),
				  apply(jacobian(pb.prob_vf, x, set(par, lens, p - ϵ3)), v));
	rmul!(dJvdp, T(1/(2ϵ3)))
	σp = -dot(w, dJvdp) / n

	if hasHessian(pb) == false || ~pb.usehessian
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically.
		# apply Jacobian adjoint
		u1 = applyJacobian(pb.prob_vf, x + ϵ2 * v, par0, w, true)
		u2 = apply(JAd_at_xp, w)
		σx = minus(u2, u1); rmul!(σx, 1 / ϵ2)
		########## Resolution of the bordered linear system ########
		# we invert Jfold
		dX, dsig, flag, it = pb.linbdsolver(J_at_xp, dpF, σx, σp, rhsu, rhsp)
		~flag && @debug "Linear solver for J did not converge."
	else
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is known analytically.
		# we solve it here instead of calling linearBorderedSolver because this removes the need to pass the linear form associated to σx
		# !!! Carefull, this method makes the linear system singular
		x1, x2, cv, it = pb.linsolver(J_at_xp, rhsu, dpF)
		~cv && @debug "Linear solver for J did not converge."

		d2Fv = d2F(pb.prob_vf, x, par0, x1, v)
		σx1 = -dot(w, d2Fv ) / n

		copyto!(d2Fv, d2F(pb.prob_vf, x, par0, x2, v))
		σx2 = -dot(w, d2Fv ) / n

		dsig = (rhsp - σx1) / (σp - σx2)

		# dX = x1 .- dsig .* x2
		dX = _copy(x1); axpy!(-dsig, x2, dX)
	end

	if debugArray isa AbstractArray
		debugArray .= [jacobian(pb.prob_vf, x, par0) dpF ; σx' σp]
	end

	return dX, dsig, true, sum(it) + sum(itv) + sum(itw)
end

function (foldl::FoldLinearSolverMinAug)(Jfold, du::BorderedArray{vectype, T}; debugArray = nothing, kwargs...) where {vectype, T}
	# kwargs is used by AbstractLinearSolver
	out =  foldMALinearSolver((Jfold.x).u,
				 (Jfold.x).p,
				 Jfold.fldpb,
				 Jfold.params,
				 du.u, du.p;
				 debugArray = debugArray)
	# this type annotation enforces type stability
	return BorderedArray{vectype, T}(out[1], out[2]), out[3], out[4]
end
###################################################################################################
@inline hasAdjoint(foldpb::FoldMAProblem) = hasAdjoint(foldpb.prob)
@inline isSymmetric(foldpb::FoldMAProblem) = isSymmetric(foldpb.prob)
residual(foldpb::FoldMAProblem, x, p) = foldpb.prob(x, p)
jacobian(foldpb::FoldMAProblem{Tprob, Nothing, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = (x = x, params = p, fldpb = foldpb.prob)
jacobian(foldpb::FoldMAProblem{Tprob, AutoDiff, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = ForwardDiff.jacobian(z -> foldpb.prob(z, p), x)
jad(foldpb::FoldMAProblem, args...) = jad(foldpb.prob, args...)
################################################################################################### Newton / Continuation functions
"""
$(SIGNATURES)

This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `BorderedArray` as returned by the function `FoldPoint`
- `par` parameters used for the vector field
- `eigenvec` guess for the 0 eigenvector
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

	newtonFold(br::AbstractBranchResult, ind_fold::Int; options = br.contparams.newtonOptions, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newtonFold(prob::AbstractBifurcationProblem,
				foldpointguess, par,
				eigenvec, eigenvec_ad,
				options::NewtonPar;
				normN = norm,
				bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
				usehessian = true,
				kwargs...)

	foldproblem = FoldProblemMinimallyAugmented(
		prob,
		_copy(eigenvec),
		_copy(eigenvec_ad),
		options.linsolver,
		# do not change linear solver if user provides it
		@set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver);
		usehessian = usehessian)

	prob_f = FoldMAProblem(foldproblem, nothing, foldpointguess, par, nothing, prob.plotSolution, prob.recordFromSolution)

	# options for the Newton Solver
	opt_fold = @set options.linsolver = FoldLinearSolverMinAug()

	# solve the Fold equations
	return newton(prob_f, opt_fold; normN = normN, kwargs...)
end

function newtonFold(br::AbstractBranchResult, ind_fold::Int;
				prob = br.prob,
				normN = norm,
				options = br.contparams.newtonOptions,
				nev = br.contparams.nev,
				startWithEigen = false,
				bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
				kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.specialpoint[ind_fold]
	eigenvec = bifpt.τ.u; rmul!(eigenvec, 1/normN(eigenvec))
	eigenvec_ad = _copy(eigenvec)

	if startWithEigen
		λ = zero(getvectoreltype(br))
		p = bifpt.param
		parbif = setParam(br, p)

		# jacobian at bifurcation point
		L = jacobian(prob, bifpt.x, parbif)

		# computation of zero eigenvector
		ζstar, = getAdjointBasis(L, λ, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = false)
		eigenvec .= real.(ζstar)

		# computation of adjoint eigenvector
		_Jt = ~hasAdjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)
		ζstar, = getAdjointBasis(_Jt, λ, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = false)
		eigenvec_ad .= real.(ζstar)
		rmul!(eigenvec_ad, 1/normN(eigenvec_ad))
	end

	# solve the Fold equations
	return newtonFold(prob, foldpointguess, getParams(br), eigenvec, eigenvec_ad, options; normN = normN, bdlinsolver = bdlinsolver, kwargs...)
end

"""
$(SIGNATURES)

Codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `BorderedArray` as returned by the function `FoldPoint`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `eigenvec` guess for the 0 eigenvector at p1_0
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options_cont` arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a, b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `computeEigenElements = false` whether to compute eigenelements. If `options_cont.detecttEvent>0`, it allows the detection of ZH points.
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call

	continuationFold(br::AbstractBranchResult, ind_fold::Int64, lens2::Lens, options_cont::ContinuationPar ; kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Fold point in `br` that you want to continue.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! tip "Detection of Bogdanov-Takens and Cusp bifurcations"
    In order to trigger the detection, pass `detectEvent = 1,2` in `options_cont`.
"""
function continuationFold(prob, alg::AbstractContinuationAlgorithm,
				foldpointguess::BorderedArray{vectype, T}, par,
				lens1::Lens, lens2::Lens,
				eigenvec, eigenvec_ad,
				options_cont::ContinuationPar ;
				normC = norm,
				updateMinAugEveryStep = 0,
				bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
				jacobian_ma::Symbol = :autodiff,
			 	computeEigenElements = false,
				usehessian = true,
				kwargs...) where {T, vectype}
	@assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
	@assert lens1 == getLens(prob)

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	foldPb = FoldProblemMinimallyAugmented(
			prob,
			_copy(eigenvec),
			_copy(eigenvec_ad),
			options_newton.linsolver,
			# do not change linear solver if user provides it
			@set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
			usehessian = usehessian)

	# Jacobian for the Fold problem
	if jacobian_ma == :autodiff
		foldpointguess = vcat(foldpointguess.u, foldpointguess.p)
		prob_f = FoldMAProblem(foldPb, AutoDiff(), foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
		opt_fold_cont = @set options_cont.newtonOptions.linsolver = DefaultLS()
	else
		prob_f = FoldMAProblem(foldPb, nothing, foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
		opt_fold_cont = @set options_cont.newtonOptions.linsolver = FoldLinearSolverMinAug()
	end

	# this functions allows to tackle the case where the two parameters have the same name
	lenses = getLensSymbol(lens1, lens2)

	# global variables to save call back
	foldPb.BT = one(T)
	foldPb.CP = one(T)
	foldPb.ZH = 1

	# this function is used as a Finalizer
	# it is called to update the Minimally Augmented problem
	# by updating the vectors a, b
	function updateMinAugFold(z, tau, step, contResult; kUP...)
		# we first check that the continuation step was successful
		# if not, we do not update the problem with bad information!
		success = get(kUP, :state, nothing).converged
		(~modCounter(step, updateMinAugEveryStep) || success == false) && return true

		x = getVec(z.u)	# fold point
		p1 = getP(z.u)	# first parameter
		p2 = z.p	# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		a = foldPb.a
		b = foldPb.b

		# expression of the jacobian
		J_at_xp = jacobian(foldPb.prob_vf, x, newpar)

		# compute new b
		newb = foldPb.linbdsolver(J_at_xp, a, b, T(0), foldPb.zero, T(1))[1]

		# compute new a
		if isSymmetric(foldPb)
			JAd_at_xp = J_at_xp
		else
			JAd_at_xp = hasAdjoint(foldPb) ? jad(foldPb.prob_vf, x, newpar) : transpose(J_at_xp)
		end
		newa = foldPb.linbdsolver(JAd_at_xp, b, a, T(0), foldPb.zero, T(1))[1]

		copyto!(foldPb.a, newa); rmul!(foldPb.a, 1/normC(newa))
		# do not normalize with dot(newb, foldPb.a), it prevents from BT detection
		copyto!(foldPb.b, newb); rmul!(foldPb.b, 1/normC(newb))

		# call the user-passed finalizer
		finaliseUser = get(kwargs, :finaliseSolution, nothing)
		if isnothing(finaliseUser) == false
			return finaliseUser(z, tau, step, contResult; prob = foldPb, kUP...)
		end
		return true
	end

	function testForBT_CP(iter, state)
		z = getx(state)
		x = getVec(z)		# fold point
		p1 = getP(z)		# first parameter
		p2 = getp(state)	# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		probfold = iter.prob.prob

		a = probfold.a
		b = probfold.b

		# expression of the jacobian
		J_at_xp = jacobian(probfold.prob_vf, x, newpar)

		# compute new b
		ζ = probfold.linbdsolver(J_at_xp, a, b, T(0), probfold.zero, T(1))[1]
		ζ ./= norm(ζ)

		# compute new a
		JAd_at_xp = hasAdjoint(probfold) ? jad(probfold, x, newpar) : transpose(J_at_xp)
		ζstar = probfold.linbdsolver(JAd_at_xp, b, a, T(0), probfold.zero, T(1))[1]
		ζstar ./= norm(ζstar)

		probfold.BT = dot(ζstar, ζ)
		probfold.CP = getP(state.τ)

		return probfold.BT, probfold.CP
	end

	function testForZH(iter, state)
		if isnothing(state.eigvals)
			iter.prob.prob.ZH = 1
		else
			ϵ = iter.contParams.tolStability
			ρ = minimum(abs ∘ real, state.eigvals)
			iter.prob.prob.ZH = mapreduce(x -> ((real(x) > ρ) & (imag(x) > ϵ)), +, state.eigvals)
		end
		return iter.prob.prob.ZH
	end

	# the following allows to append information specific to the codim 2 continuation to the user data
	_printsol = get(kwargs, :recordFromSolution, nothing)
	_printsol2 = isnothing(_printsol) ?
		(u, p; kw...) -> (; zip(lenses, (getP(u), p))..., BT = foldPb.BT, CP = foldPb.CP, ZH = foldPb.ZH, namedprintsol(recordFromSolution(prob)(getVec(u), p; kw...))...) :
		(u, p; kw...) -> (; namedprintsol(_printsol(getVec(u), p; kw...))..., zip(lenses, (getP(u, foldPb), p))..., BT = foldPb.BT, CP = foldPb.CP, ZH = foldPb.ZH,)

	# eigen solver
	eigsolver = FoldEigsolver(getsolver(opt_fold_cont.newtonOptions.eigsolver))

	prob_f = reMake(prob_f, recordFromSolution = _printsol2)

	# solve the Fold equations
	br = continuation(
		prob_f, alg,
		(@set opt_fold_cont.newtonOptions.eigsolver = eigsolver);
		linearAlgo = BorderingBLS(solver = opt_fold_cont.newtonOptions.linsolver, checkPrecision = false),
		kwargs...,
		kind = FoldCont(),
		normC = normC,
		finaliseSolution = updateMinAugFold,
		event = PairOfEvents(ContinuousEvent(2, testForBT_CP, computeEigenElements, ("bt", "cusp"), 0), DiscreteEvent(1, testForZH, false, ("zh",)))
		)
		@assert ~isnothing(br) "Empty branch!"
	return correctBifurcation(br)
end

function continuationFold(prob,
				br::AbstractBranchResult, ind_fold::Int,
				lens2::Lens,
				options_cont::ContinuationPar = br.contparams ;
				alg = br.alg,
				normC = norm,
				nev = br.contparams.nev,
				startWithEigen = false,
				kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.specialpoint[ind_fold]
	eigenvec = bifpt.τ.u; rmul!(eigenvec, 1/norm(eigenvec))
	eigenvec_ad = _copy(eigenvec)

	p = bifpt.param
	parbif = setParam(br, p)

	if startWithEigen
		# jacobian at bifurcation point
		L = jacobian(prob, bifpt.x, parbif)

		# computation of adjoint eigenvalue
		eigenvec .= real.(	geteigenvector(options_cont.newtonOptions.eigsolver ,br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
		rmul!(eigenvec, 1/normC(eigenvec))

		# jacobian adjoint at bifurcation point
		_Jt = hasAdjoint(prob) ? jad(prob, bifpt.x, parbif) : transpose(L)

		ζstar, λstar = getAdjointBasis(_Jt, 0, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = options_cont.newtonOptions.verbose)
		eigenvec_ad = real.(ζstar)
		rmul!(eigenvec_ad, 1/dot(eigenvec, eigenvec_ad))
	end

	return continuationFold(prob, alg,
			foldpointguess, parbif,
			getLens(br), lens2,
			eigenvec, eigenvec_ad,
			options_cont ;
			normC = normC,
			kwargs...)
end

# structure to compute eigen-elements along branch of Fold points
struct FoldEigsolver{S} <: AbstractCodim2EigenSolver
	eigsolver::S
end

function (eig::FoldEigsolver)(Jma, nev; kwargs...)
	n = min(nev, length(Jma.x.u))
	J = jacobian(Jma.fldpb.prob_vf, getVec(Jma.x), set(Jma.params, getLens(Jma.fldpb), getP(Jma.x)))
	eigenelts = eig.eigsolver(J, n; kwargs...)
	return eigenelts
end

@views function (eig::FoldEigsolver)(Jma::AbstractMatrix, nev; kwargs...)
	eigenelts = eig.eigsolver(Jma[1:end-1,1:end-1], nev; kwargs...)
	return eigenelts
end

geteigenvector(eig::FoldEigsolver, vectors, i::Int) = geteigenvector(eig.eigsolver, vectors, i)
