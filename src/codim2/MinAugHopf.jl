"""
For an initial guess from the index of a Hopf bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonHopf`.
"""
function HopfPoint(br::AbstractBranchResult, index::Int)
	@assert br.specialpoint[index].type == :hopf "The provided index does not refer to a Hopf point"
	specialpoint = br.specialpoint[index]			# Hopf point
	eigRes = br.eig									# eigenvector at the Hopf point
	p = specialpoint.param							# parameter value at the Hopf point
	ω = imag(eigRes[specialpoint.idx].eigenvals[specialpoint.ind_ev])	# frequency at the Hopf point
	return BorderedArray(specialpoint.x, [p, ω] )
end
####################################################################################################
@inline getVec(x, ::HopfProblemMinimallyAugmented) = extractVecBLS(x, 2)
@inline getP(x, ::HopfProblemMinimallyAugmented) = extractParBLS(x, 2)

# this function encodes the functional
function (hp::HopfProblemMinimallyAugmented)(x, p::T, ω::T, params) where T
	# These are the equations of the minimally augmented (MA) formulation of the Hopf bifurcation point
	# input:
	# - x guess for the point at which the jacobian has a purely imaginary eigenvalue
	# - p guess for the parameter for which the jacobian has a purely imaginary eigenvalue
	# The jacobian of the MA problem is solved with a BLS method
	a = hp.a
	b = hp.b
	# update parameter
	par = set(params, getLens(hp), p)
	# ┌         ┐┌  ┐   ┌ ┐
	# │ J-iω  a ││v │ = │0│
	# │  b    0 ││σ1│   │1│
	# └         ┘└  ┘   └ ┘
	# In the notations of Govaerts 2000, a = w, b = v
	# Thus, b should be a null vector of J - iω
	#       a should be a null vector of J'+ iω
	# we solve (J - iω)⋅v + a σ1 = 0 with <b, v> = n
	n = T(1)
	# note that the shift argument only affect J in this call:
	σ1 = hp.linbdsolver(jacobian(hp.prob_vf, x, par), a, b, T(0), hp.zero, n; shift = Complex{T}(0, -ω))[2]

	# we solve (J+iω)'w + b σ2 = 0 with <a, w> = n
	# we find sigma2 = conj(sigma1)
	# w, σ2, _ = fp.linbdsolver(fp.Jadjoint(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n)

	# the constraint is σ = <w, Jv> / n
	# σ = -dot(w, apply(fp.J(x, p) + Complex(0, ω) * I, v)) / n
	# we should have σ = σ1

	return residual(hp.prob_vf, x, par), real(σ1), imag(σ1)
end

# this function encodes the functional
function (hopfpb::HopfProblemMinimallyAugmented)(x::BorderedArray, params)
	res = hopfpb(x.u, x.p[1], x.p[2], params)
	return BorderedArray(res[1], [res[2], res[3]])
end

@views function (hopfpb::HopfProblemMinimallyAugmented)(x::AbstractVector, params)
	res = hopfpb(x[1:end-2], x[end-1], x[end], params)
	return vcat(res[1], res[2], res[3])
end

# Struct to invert the jacobian of the Hopf MA problem.
struct HopfLinearSolverMinAug <: AbstractLinearSolver; end

"""
This function solves the linear problem associated with a linearization of the minimally augmented formulation of the Hopf bifurcation point. The keyword `debugArray` is used to debug the routine by returning several key quantities.
"""
function hopfMALinearSolver(x, p::T, ω::T, pb::HopfProblemMinimallyAugmented, par,
	 						duu, dup, duω;
							debugArray = nothing) where T
	################################################################################################
	# debugArray is used as a temp to be filled with values used for debugging. If debugArray = nothing, then no debugging mode is entered. If it is AbstractVector, then it is used
	################################################################################################
	# N = length(du) - 2
	# The Jacobian J of the vector field is expressed at (x, p)
	# the jacobian expression of the hopf problem Jhopf is
	#           ┌             ┐
	#  Jhopf =  │  J  dpF   0 │
	#           │ σx   σp  σω │
	#           └             ┘
	########## Resolution of the bordered linear system ########
	# J * dX	  + dpF * dp		   = du => dX = x1 - dp * x2
	# The second equation
	#	<σx, dX> +  σp * dp + σω * dω = du[end-1:end]
	# thus becomes
	#   (σp - <σx, x2>) * dp + σω * dω = du[end-1:end] - <σx, x1>
	# This 2 x 2 system is then solved to get (dp, dω)
	############### Extraction of function names #################
	a = pb.a
	b = pb.b

	# parameter axis
	lens = getLens(pb)
	# update parameter
	par0 = set(par, lens, p)

	# we define the following jacobian. It is used at least 3 times below. This avoid doing 3 times the possibly costly building of J(x, p)
	J_at_xp = jacobian(pb.prob_vf, x, par0)

	# we do the following to avoid computing J_at_xp twice in case pb.Jadjoint is not provided
	JAd_at_xp = hasAdjoint(pb) ? jad(pb.prob_vf, x, par0) : transpose(J_at_xp)

	# normalization
	n = T(1)

	# we solve (J-iω)v + a σ1 = 0 with <b, v> = n
	v, σ1, cv, itv = pb.linbdsolver(J_at_xp, a, b, T(0), pb.zero, n; shift = Complex{T}(0, -ω))
	~cv && @debug "Linear solver for (J-iω) did not converge."

	# we solve (J+iω)'w + b σ1 = 0 with <a, w> = n
	w, σ2, cv, itw = pb.linbdsolverAdjoint(JAd_at_xp, b, a, T(0), pb.zero, n; shift = Complex{T}(0, ω))
	~cv && @debug "Linear solver for (J+iω)' did not converge."

	δ = getDelta(pb.prob_vf)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	################### computation of σx σp ####################
	################### and inversion of Jhopf ####################
	dpF   = (residual(pb.prob_vf, x, set(par, lens, p + ϵ1)) -
				residual(pb.prob_vf, x, set(par, lens, p - ϵ1))) / T(2ϵ1)
	dJvdp = (apply(jacobian(pb.prob_vf, x, set(par, lens, p + ϵ3)), v) -
				apply(jacobian(pb.prob_vf, x, set(par, lens, p - ϵ3)), v)) / T(2ϵ3)
	σp = -dot(w, dJvdp) / n

	# case of sigma_omega
	# σω = dot(w, Complex{T}(0, 1) * v) / n
	σω = Complex{T}(0, 1) * dot(w, v) / n

	# we solve J⋅x1 = duu and J⋅x2 = dpF
	x1, x2, cv, (it1, it2) = pb.linsolver(J_at_xp, duu, dpF)
	~cv && @debug "Linear solver for J did not converge."

	# the case of ∂_xσ is a bit more involved
	# we first need to compute the value of ∂_xσ written σx
	# σx = zeros(Complex{T}, length(x))
	σx = similar(x, Complex{T})

	if hasHessian(pb) == false
		cw = conj(w)
		vr = real(v); vi = imag(v)
		u1r = applyJacobian(pb.prob_vf, x + ϵ2 * vr, par0, cw, true)
		u1i = applyJacobian(pb.prob_vf, x + ϵ2 * vi, par0, cw, true)
		u2 = apply(JAd_at_xp,  cw)
		σxv2r = @. -(u1r - u2) / ϵ2
		σxv2i = @. -(u1i - u2) / ϵ2
		σx = @. σxv2r + Complex{T}(0, 1) * σxv2i

		σxx1 = dot(σx, x1)
		σxx2 = dot(σx, x2)
	else
		d2Fv = d2Fc(pb.prob_vf, x, par0, v, x1)
		σxx1 = -conj(dot(w, d2Fv) / n)
		d2Fv = d2Fc(pb.prob_vf, x, par0, v, x2)
		σxx2 = -conj(dot(w, d2Fv) / n)
	end
	# we need to be carefull here because the dot produces conjugates. Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
	dp, dω = [real(σp - σxx2) real(σω);
			  imag(σp + σxx2) imag(σω) ] \
			  [dup - real(σxx1), duω + imag(σxx1)]

	if debugArray isa AbstractVector
		debugArray .= vcat(σp, σω, σx)
	end
	return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
end

function (hopfl::HopfLinearSolverMinAug)(Jhopf, du::BorderedArray{vectype, T}; debugArray = nothing, kwargs...)  where {vectype, T}
	# kwargs is used by AbstractLinearSolver
	out = hopfMALinearSolver((Jhopf.x).u,
				(Jhopf.x).p[1],
				(Jhopf.x).p[2],
				Jhopf.hopfpb,
				Jhopf.params,
				du.u, du.p[1], du.p[2];
				debugArray = debugArray)
	# this type annotation enforces type stability
	BorderedArray{vectype, T}(out[1], [out[2], out[3]]), out[4], out[5]
end
###################################################################################################
# define a problem <: AbstractBifurcationProblem
@inline hasAdjoint(hopfpb::HopfMAProblem) = hasAdjoint(hopfpb.prob)
@inline isSymmetric(hopfpb::HopfMAProblem) = isSymmetric(hopfpb.prob)
residual(hopfpb::HopfMAProblem, x, p) = hopfpb.prob(x, p)
# jacobian(hopfpb::HopfMAProblem, x, p) = hopfpb.jacobian(x, p)
jacobian(hopfpb::HopfMAProblem{Tprob, Nothing, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = (x = x, params = p, hopfpb = hopfpb.prob)
jacobian(hopfpb::HopfMAProblem{Tprob, AutoDiff, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = ForwardDiff.jacobian(z -> hopfpb.prob(z, p), x)
################################################################################################### Newton / Continuation functions
"""
$(SIGNATURES)

This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationProblem` where `p` is a set of parameters.
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should a `BorderedArray` as returned by the function `HopfPoint`.
- `par` parameters used for the vector field
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call:
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows

	newtonHopf(br::AbstractBranchResult, ind_hopf::Int; normN = norm, options = br.contparams.newtonOptions, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newtonHopf(prob,
			hopfpointguess::BorderedArray,
			par,
			eigenvec, eigenvec_ad,
			options::NewtonPar;
			normN = norm,
			bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
			kwargs...)
	# we first need to update d2F and d3F for them to accept complex arguments

	hopfproblem = HopfProblemMinimallyAugmented(
		prob,
		_copy(eigenvec_ad),	# this is pb.a ≈ null space of (J - iω I)^*
		_copy(eigenvec), 	# this is pb.b ≈ null space of  J - iω I
		options.linsolver,
		# do not change linear solver if user provides it
		@set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver))

	prob_h = HopfMAProblem(hopfproblem, nothing, hopfpointguess, par, nothing, prob.plotSolution, prob.recordFromSolution)

	# options for the Newton Solver
	opt_hopf = @set options.linsolver = HopfLinearSolverMinAug()

	# solve the hopf equations
	return newton(prob_h, opt_hopf, normN = normN, kwargs...)
end

function newtonHopf(br::AbstractBranchResult, ind_hopf::Int;
			prob = br.prob,
			normN = norm,
			options = br.contparams.newtonOptions,
			verbose = true,
			nev = br.contparams.nev,
			startWithEigen = false,
			kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	ω = hopfpointguess.p[2]
	bifpt = br.specialpoint[ind_hopf]
	options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	@assert bifpt.idx == bifpt.step + 1 "Error, the bifurcation index does not refer to the correct step"
	ζ = geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	ζ ./= normN(ζ)
	ζad = LinearAlgebra.conj.(ζ)

	if startWithEigen
		# computation of adjoint eigenvalue. Recall that b should be a null vector of J-iω
		λ = Complex(0, ω)
		p = bifpt.param
		parbif = setParam(br, p)

		# jacobian at bifurcation point
		L = jacobian(prob, bifpt.x, parbif)

		# computation of adjoint eigenvector
		_Jt = ~hasAdjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)

		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = false)
		ζad .= ζstar ./ dot(ζstar, ζ)
	end

	# solve the hopf equations
	return newtonHopf(prob, hopfpointguess, getParams(br), ζ, ζad, options; normN = normN, kwargs...)
end

"""
$(SIGNATURES)

codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationProblem`
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedArray`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options_cont` keywords arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:

- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a,b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `computeEigenElements = false` whether to compute eigenelements. If `options_cont.detecttEvent>0`, it allows the detection of ZH, HH points.
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call:

	continuationHopf(br::AbstractBranchResult, ind_hopf::Int, lens2::Lens, options_cont::ContinuationPar ;  kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` that you want to refine.

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "Detection of Bogdanov-Takens and Bautin bifurcations"
    In order to trigger the detection, pass `detectEvent = 1,2` in `options_cont`. Note that you need to provide `d3F` in `prob`.
"""
function continuationHopf(prob_vf, alg::AbstractContinuationAlgorithm,
				hopfpointguess::BorderedArray{vectype, Tb}, par,
				lens1::Lens, lens2::Lens,
				eigenvec, eigenvec_ad,
				options_cont::ContinuationPar ;
				updateMinAugEveryStep = 0,
				normC = norm,
				bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
				jacobian_ma::Symbol = :autodiff,
				computeEigenElements = false,
				kwargs...) where {Tb, vectype}
	@assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
	@assert lens1 == getLens(prob_vf)

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions
	threshBT = 100options_newton.tol

	hopfPb = HopfProblemMinimallyAugmented(
		prob_vf,
		_copy(eigenvec_ad),	# this is a ≈ null space of (J - iω I)^*
		_copy(eigenvec), 	# this is b ≈ null space of  J - iω I
		options_newton.linsolver,
		# do not change linear solver if user provides it
		@set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver))

	# Jacobian for the Hopf problem
	if jacobian_ma == :autodiff
		hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
		prob_h = HopfMAProblem(hopfPb, AutoDiff(), hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
		opt_hopf_cont = @set options_cont.newtonOptions.linsolver = DefaultLS()
	else
		prob_h = HopfMAProblem(hopfPb, nothing, hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
		opt_hopf_cont = @set options_cont.newtonOptions.linsolver = HopfLinearSolverMinAug()
	end

	# this functions allows to tackle the case where the two parameters have the same name
	lenses = getLensSymbol(lens1, lens2)

	# current lyapunov coefficient
	eTb = eltype(Tb)
	hopfPb.l1 = Complex{eTb}(0, 0)
	hopfPb.BT = one(eTb)
	hopfPb.GH = one(eTb)

	# this function is used as a Finalizer
	# it is called to update the Minimally Augmented problem
	# by updating the vectors a, b
	function updateMinAugHopf(z, tau, step, contResult; kUP...)
		# we first check that the continuation step was successful
		# if not, we do not update the problem with bad information!
		success = get(kUP, :state, nothing).converged
		(~modCounter(step, updateMinAugEveryStep) || success == false) && return true
		x = getVec(z.u, hopfPb)	# hopf point
		p1, ω = getP(z.u, hopfPb)
		p2 = z.p		# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		a = hopfPb.a
		b = hopfPb.b

		# expression of the jacobian
		J_at_xp = jacobian(hopfPb.prob_vf, x, newpar)

		# compute new b
		T = typeof(p1)
		local n = T(1)
		newb = hopfPb.linbdsolver(J_at_xp, a, b, T(0), hopfPb.zero, n; shift = Complex(0, -ω))[1]

		# compute new a
		JAd_at_xp = hasAdjoint(hopfPb) ? jad(hopfPb.prob_vf, x, newpar) : adjoint(J_at_xp)
		newa = hopfPb.linbdsolver(JAd_at_xp, b, a, T(0), hopfPb.zero, n; shift = Complex(0, ω))[1]

		hopfPb.a .= newa ./ normC(newa)
		# do not normalize with dot(newb, hopfPb.a), it prevents BT detection
		hopfPb.b .= newb ./ normC(newb)

		# we stop continuation at Bogdanov-Takens points

		# CA NE DEVRAIT PAS ETRE ISSNOT?
		isbt = isnothing(contResult) ? true : isnothing(findfirst(x -> x.type in (:bt, :ghbt, :btgh), contResult.specialpoint))

		# if the frequency is null, this is not a Hopf point, we halt the process
		if abs(ω) < threshBT
			@warn "[Codim 2 Hopf - Finalizer] The Hopf curve seems to be close to a BT point: ω ≈ $ω. Stopping computations at ($p1, $p2). If the BT point is not detected, try lowering Newton tolerance or dsmax."
		end

		# call the user-passed finalizer
		finaliseUser = get(kwargs, :finaliseSolution, nothing)
		resFinal = isnothing(finaliseUser) ? true : finaliseUser(z, tau, step, contResult; prob = hopfPb, kUP...)

		return abs(ω) >= threshBT && isbt && resFinal
	end

	function testBT_GH(iter, state)
		z = getx(state)
		x = getVec(z, hopfPb)		# hopf point
		p1, ω = getP(z, hopfPb)		# first parameter
		p2 = getp(state)			# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		probhopf = iter.prob.prob

		a = probhopf.a
		b = probhopf.b

		# expression of the jacobian
		J_at_xp = jacobian(probhopf.prob_vf, x, newpar)

		# compute new b
		T = typeof(p1)
		n = T(1)
		ζ = probhopf.linbdsolver(J_at_xp, a, b, T(0), probhopf.zero, n; shift = Complex(0, -ω))[1]
		ζ ./= normC(ζ)

		# compute new a
		JAd_at_xp = hasAdjoint(probhopf) ? jad(probhopf.prob_vf, x, newpar) : transpose(J_at_xp)
		ζstar = probhopf.linbdsolver(JAd_at_xp, b, a, T(0), hopfPb.zero, n; shift = Complex(0, ω))[1]
		# test function for Bogdanov-Takens
		probhopf.BT = ω
		BT2 = real( dot(ζstar ./ normC(ζstar), ζ) )
		ζstar ./= dot(ζ, ζstar)

		hp = Hopf(x, p1, ω, newpar, lens1, ζ, ζstar, (a = Complex{T}(0,0), b = Complex{T}(0,0)), :hopf)
		hopfNormalForm(prob_vf, hp, options_newton.linsolver, verbose = false)

		# lyapunov coefficient
		probhopf.l1 = hp.nf.b
		# test for Bautin bifurcation.
		# If GH is too large, we take the previous value to avoid spurious detection
		# GH will be large close to BR points
		probhopf.GH = abs(real(hp.nf.b)) < 1e5 ? real(hp.nf.b) : state.eventValue[2][2]
		return probhopf.BT, probhopf.GH
	end

	# the following allows to append information specific to the codim 2 continuation to the user data
	_printsol = get(kwargs, :recordFromSolution, nothing)
	_printsol2 = isnothing(_printsol) ?
		(u, p; kw...) -> (; namedprintsol(recordFromSolution(prob_vf)(getVec(u, hopfPb), p; kw...))..., zip(lenses, (getP(u, hopfPb)[1], p))..., ω = getP(u, hopfPb)[2], l1 = hopfPb.l1, BT = hopfPb.BT, GH = hopfPb.GH) :
		(u, p; kw...) -> (; namedprintsol(_printsol(getVec(u, hopfPb), p; kw...))..., zip(lenses, (getP(u, hopfPb)[1], p))..., ω = getP(u, hopfPb)[2], l1 = hopfPb.l1, BT = hopfPb.BT, GH = hopfPb.GH)

	prob_h = reMake(prob_h, recordFromSolution = _printsol2)

	# eigen solver
	eigsolver = HopfEig(getsolver(opt_hopf_cont.newtonOptions.eigsolver))

	# event for detecting codim 2 points
	if computeEigenElements
		event = PairOfEvents(ContinuousEvent(2, testBT_GH, computeEigenElements, ("bt", "gh"), threshBT), BifDetectEvent)
		# careful here, we need to adjust the tolerance for stability to avoid
		# spurious ZH or HH bifurcations
		@set! opt_hopf_cont.tolStability = 10opt_hopf_cont.newtonOptions.tol
	else
		event = ContinuousEvent(2, testBT_GH, false, ("bt", "gh"), threshBT)
	end

	prob_h = reMake(prob_h, recordFromSolution = _printsol2)

	# solve the hopf equations
	br = continuation(
		prob_h, alg,
		(@set opt_hopf_cont.newtonOptions.eigsolver = eigsolver);
		kwargs...,
		kind = HopfCont(),
		linearAlgo = BorderingBLS(solver = opt_hopf_cont.newtonOptions.linsolver, checkPrecision = false),
		normC = normC,
		finaliseSolution = updateMinAugHopf,
		event = event
	)
	@assert ~isnothing(br) "Empty branch!"
	return correctBifurcation(br)
end

function continuationHopf(prob,
						br::AbstractBranchResult, ind_hopf::Int64,
						lens2::Lens,
						options_cont::ContinuationPar = br.contparams;
						alg = br.alg,
						startWithEigen = false,
						normC = norm,
						kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	ω = hopfpointguess.p[2]
	bifpt = br.specialpoint[ind_hopf]

	@assert ~isnothing(br.eig) "The branch contains no eigen elements. This is strange because a Hopf point was detected. Please open an issue on the website."

	@assert ~isnothing(br.eig[1].eigenvecs) "The branch contains no eigenvectors for the Hopf point. Please provide one."

	ζ = geteigenvector(options_cont.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	ζ ./= normC(ζ)
	ζad = conj.(ζ)

	p = bifpt.param
	parbif = setParam(br, p)

	if startWithEigen
		# computation of adjoint eigenvalue
		λ = Complex(0, ω)
		# jacobian at bifurcation point
		L = jacobian(prob, bifpt.x, parbif)

		# jacobian adjoint at bifurcation point
		_Jt = ~hasAdjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)

		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), br.contparams.newtonOptions.eigsolver; nev = br.contparams.nev, verbose = false)
		ζad .= ζstar ./ dot(ζstar, ζ)
	end

	return continuationHopf(br.prob, alg,
					hopfpointguess, parbif,
					getLens(br), lens2,
					ζ, ζad,
					options_cont ;
					normC = normC,
					kwargs...)
end

# structure to compute the eigenvalues along the Hopf branch
struct HopfEig{S} <: AbstractCodim2EigenSolver
	eigsolver::S
end

function (eig::HopfEig)(Jma, nev; kwargs...)
	n = min(nev, length(Jma.x.u))

	x = Jma.x.u		# hopf point
	p1, ω = Jma.x.p	# first parameter
	newpar = set(Jma.params, getLens(Jma.hopfpb), p1)

	J = jacobian(Jma.hopfpb.prob_vf, x, newpar)

	eigenelts = eig.eigsolver(J, n; kwargs...)
	return eigenelts
end

@views function (eig::HopfEig)(Jma::AbstractMatrix, nev; kwargs...)
	eigenelts = eig.eigsolver(Jma[1:end-2,1:end-2], nev; kwargs...)
	return eigenelts
end

geteigenvector(eig::HopfEig, vectors, i::Int) = geteigenvector(eig.eigsolver, vectors, i)
