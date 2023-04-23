"""
$(SIGNATURES)

For an initial guess from the index of a NS bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonFold`.
"""
function NSPoint(br::AbstractBranchResult, index::Int)
	bptype = br.specialpoint[index].type
	@assert bptype == :ns "This should be a NS point"
	specialpoint = br.specialpoint[index]
	ω = imag(br.eig[specialpoint.idx].eigenvals[specialpoint.ind_ev])
	return BorderedArray(_copy(specialpoint.x), [specialpoint.param, ω])
end
####################################################################################################
@inline getVec(x, ::NeimarkSackerProblemMinimallyAugmented) = extractVecBLS(x, 2)
@inline getP(x, ::NeimarkSackerProblemMinimallyAugmented) = extractParBLS(x, 2)

nstest(JacNS, v, w, J22, _zero, n; lsbd = MatrixBLS()) = lsbd(JacNS, v, w, J22, _zero, n)

# this function encodes the functional
function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x, p::T, ω::T, params) where T
	# These are the equations of the minimally augmented (MA) formulation of the Period-Doubling bifurcation point
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter value `<: Real` at which the jacobian is singular
	# The jacobian of the MA problem is solved with a BLS method
	a = 𝐍𝐒.a
	b = 𝐍𝐒.b
	# update parameter
	par = set(params, getLens(𝐍𝐒), p)
	# ┌      ┐┌  ┐   ┌ ┐
	# │ J  a ││v │ = │0│
	# │ b  0 ││σ1│   │1│
	# └      ┘└  ┘   └ ┘
	# In the notations of Govaerts 2000, a = w, b = v
	# Thus, b should be a null vector of J
	#       a should be a null vector of J'
	# we solve Jv + a σ1 = 0 with <b, v> = 1
	# the solution is v = -σ1 J\a with σ1 = -1/<b, J^{-1}a>
	J = jacobianNeimarkSacker(𝐍𝐒.prob_vf, x, par, ω)
	σ1 = nstest(J, a, b, T(0), 𝐍𝐒.zero, T(1); lsbd = 𝐍𝐒.linbdsolver)[2]
	return residual(𝐍𝐒.prob_vf, x, par), real(σ1), imag(σ1)
end

# this function encodes the functional
function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x::BorderedArray, params)
	res = 𝐍𝐒(x.u, x.p[1], x.p[2], params)
	return BorderedArray(res[1], [res[2], res[3]])
end

@views function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x::AbstractVector, params)
	res = 𝐍𝐒(x[1:end-2], x[end-1], x[end], params)
	return vcat(res[1], res[2], res[3])
end

###################################################################################################
# Struct to invert the jacobian of the pd MA problem.
struct NSLinearSolverMinAug <: AbstractLinearSolver; end

function NSMALinearSolver(x, p::T, ω::T, 𝐍𝐒::NeimarkSackerProblemMinimallyAugmented, par,
							duu, dup, duω;
							debugArray = nothing) where T
	################################################################################################
	# debugArray is used as a temp to be filled with values used for debugging. If debugArray = nothing, then no debugging mode is entered. If it is AbstractArray, then it is populated
	################################################################################################
	# Recall that the functional we want to solve is [F(x,p), σ(x,p)]
	# where σ(x,p) is computed in the above functions and F is the periodic orbit
	# functional. We recall that N⋅[v, σ] ≡ [0, 1]
	# The Jacobian Jpd of the functional is expressed at (x, p)
	# We solve here Jpd⋅res = rhs := [rhsu, rhsp, rhsω]
	# The Jacobian expression of the NS problem is
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
	########################## Extraction of function names ########################################
	a = 𝐍𝐒.a
	b = 𝐍𝐒.b

	# get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
	POWrap = 𝐍𝐒.prob_vf

	# parameter axis
	lens = getLens(𝐍𝐒)
	# update parameter
	par0 = set(par, lens, p)

	# we define the following jacobian. It is used at least 3 times below. This avoids doing 3 times the (possibly) costly building of J(x, p)
	JNS = jacobianNeimarkSacker(POWrap, x, par0, ω) # jacobian with period doubling boundary condition

	# we do the following in order to avoid computing the jacobian twice in case 𝐍𝐒.Jadjoint is not provided
	JNS★ = hasAdjoint(𝐍𝐒) ? jacobianAdjointNeimarkSacker(POWrap, x, par0, ω) : adjoint(JNS)

	# we solve N[v, σ1] = [0, 1]
	v, σ1, cv, itv = nstest(JNS, a, b, T(0), 𝐍𝐒.zero, T(1); lsbd = 𝐍𝐒.linbdsolver)
	~cv && @debug "Linear solver for N did not converge."

	# # we solve Nᵗ[w, σ2] = [0, 1]
	w, σ2, cv, itw = nstest(JNS★, b, a, T(0), 𝐍𝐒.zero, T(1); lsbd = 𝐍𝐒.linbdsolver)
	~cv && @debug "Linear solver for Nᵗ did not converge."

	δ = getDelta(POWrap)
	ϵ1, ϵ2, ϵ3 = T(δ), (T(δ)), (T(δ))
	################### computation of σx σp ####################
	################### and inversion of Jpd ####################
	dₚF = minus(residual(POWrap, x, set(par, lens, p + ϵ1)),
				residual(POWrap, x, set(par, lens, p - ϵ1))); rmul!(dₚF, T(1 / (2ϵ1)))
	dJvdp = minus(apply(jacobianNeimarkSacker(POWrap, x, set(par, lens, p + ϵ3), ω), v),
				  apply(jacobianNeimarkSacker(POWrap, x, set(par, lens, p - ϵ3), ω), v));
	rmul!(dJvdp, T(1/(2ϵ3)))
	σₚ = -dot(w, dJvdp)

	# case of ∂σ_ω
	σω = -(dot(w, apply(jacobianNeimarkSacker(POWrap, x, par, ω+ϵ2), v)) - 
			dot(w, apply(jacobianNeimarkSacker(POWrap, x, par, ω), v)) )/ϵ2

	if hasHessian(𝐍𝐒) == false || 𝐍𝐒.usehessian == false
		cw = conj(w)
		vr = real(v); vi = imag(v)
		u1r = jacobianNeimarkSacker(POWrap, x .+ ϵ2 .* vcat(vr,0), par0, ω).jacpb' * cw
		u1i = jacobianNeimarkSacker(POWrap, x .+ ϵ2 .* vcat(vi,0), par0, ω).jacpb' * cw
		u2 = JNS★.jacpb * cw
		σxv2r = @. -(u1r - u2) / ϵ2 # careful, this is a complex vector
		σxv2i = @. -(u1i - u2) / ϵ2
		σx = @. σxv2r + Complex{T}(0, 1) * σxv2i

		dJvdt = minus(apply(jacobianNeimarkSacker(POWrap, x .+ ϵ2 .* vcat(0*vr,1),par0, ω), v),
				  apply(jacobianNeimarkSacker(POWrap, x .- ϵ2 .* vcat(0*vr,1),par0, ω), v));
		rmul!(dJvdt, T(1/(2ϵ3)))
		σt = -dot(w, dJvdt) 

		_Jpo = jacobian(POWrap, x, par0).jacpb
		x1, x2, cv, (it1, it2) = 𝐍𝐒.linsolver(_Jpo, duu, dₚF)

		σxx1 = dot(vcat(σx,σt), x1)
		σxx2 = dot(vcat(σx,σt), x2)

		dp, dω = [real(σₚ - σxx2) real(σω);
			  imag(σₚ + σxx2) imag(σω) ] \
			  [dup - real(σxx1), duω + imag(σxx1)]

		# Jns = hcat(_Jpo, dₚF, zero(dₚF))
		# Jns = vcat(Jns, vcat(real(σx), real(σt), real(σₚ), real(σω))')
		# Jns = vcat(Jns, vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')

		# sol = Jns \ vcat(duu,dup,duω)
		# return sol[1:end-2], sol[end-1],sol[end],true,2

		# Jfd = ForwardDiff.jacobian(z->𝐍𝐒(z,par0),vcat(x,p,ω))

		# display(Jfd[end, 1:19]')	
		# display(vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')

		# @debug "" norm(Jns-Jfd, Inf) dp dω

		# Jns .= Jfd 
		
		return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
	else
		@assert 1==0 "WIP"
	end

	if debugArray isa AbstractArray
		debugArray .= [jacobian(POWrap, x, par0) dₚF ; σₓ' σₚ]
	end

	return dX, dsig, true, sum(it) + sum(itv) + sum(itw)
end

function (pdls::NSLinearSolverMinAug)(Jns, rhs::BorderedArray{vectype, T}; debugArray = nothing, kwargs...) where {vectype, T}
	# kwargs is used by AbstractLinearSolver
	out = NSMALinearSolver((Jns.x).u,
				(Jns.x).p[1],
				(Jns.x).p[2],
				Jns.nspb,
				Jns.params,
				rhs.u, rhs.p[1], rhs.p[2];
				debugArray = debugArray)
	# this type annotation enforces type stability
	return BorderedArray{vectype, T}(out[1], [out[2], out[3]]), out[4], out[5]
end
###################################################################################################
residual(nspb::NSMAProblem, x, p) = nspb.prob(x, p)

jacobian(nspb::NSMAProblem{Tprob, Nothing, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = (x = x, params = p, nspb = nspb.prob)

jacobian(nspb::NSMAProblem{Tprob, AutoDiff, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{Lens, Nothing}, Tplot, Trecord} = ForwardDiff.jacobian(z -> nspb.prob(z, p), x)
###################################################################################################
function continuationNS(prob, alg::AbstractContinuationAlgorithm,
				nspointguess::BorderedArray{vectype, 𝒯b}, par,
				lens1::Lens, lens2::Lens,
				eigenvec, eigenvec_ad,
				options_cont::ContinuationPar ;
				normC = norm,
				updateMinAugEveryStep = 0,
				bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
				jacobian_ma::Symbol = :autodiff,
			 	computeEigenElements = false,
				kind = NSCont(),
				usehessian = true,
				kwargs...) where {𝒯b, vectype}
	@assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
	@assert lens1 == getLens(prob)

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	𝐍𝐒 = NeimarkSackerProblemMinimallyAugmented(
			prob,
			_copy(eigenvec),
			_copy(eigenvec_ad),
			options_newton.linsolver,
			# do not change linear solver if user provides it
			@set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
			usehessian = usehessian)

	@assert jacobian_ma in (:autodiff, :finiteDifferences, :minaug, :finiteDifferencesMF)

	# Jacobian for the NS problem
	if jacobian_ma == :autodiff
		nspointguess = vcat(nspointguess.u, nspointguess.p...)
		prob_ns = NSMAProblem(𝐍𝐒, AutoDiff(), nspointguess, par, lens2, plotDefault, prob.recordFromSolution)
		opt_ns_cont = @set options_cont.newtonOptions.linsolver = DefaultLS()
	else
		prob_ns = NSMAProblem(𝐍𝐒, nothing, nspointguess, par, lens2, plotDefault, prob.recordFromSolution)
		opt_ns_cont = @set options_cont.newtonOptions.linsolver = NSLinearSolverMinAug()
	end

	# this functions allows to tackle the case where the two parameters have the same name
	lenses = getLensSymbol(lens1, lens2)

	# current lyapunov coefficient
	𝒯 = eltype(𝒯b)
	𝐍𝐒.l1 = Complex{𝒯}(0, 0)

	function testCH(iter, state)
		z = getx(state)
		x = getVec(z, 𝐍𝐒)		# NS point
		p1, ω = getP(z, 𝐍𝐒)	# first parameter
		p2 = getp(state)	   # second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		prob_ns = iter.prob.prob
		pbwrap = prob_ns.prob_vf

		#############
		ns0 = NeimarkSacker(copy(x), p1, ω, newpar, lens1, nothing, nothing, nothing, :none)
		if pbwrap.prob isa ShootingProblem
			sh = pbwrap.prob
			ns = neimarksackerNormalForm(pbwrap, ns0, (1, 1), NewtonPar(options_newton, verbose = false,))
			prob_ns.l1 = ns.nf.nf.b
			prob_ns.l1 = abs(real(ns.nf.nf.b)) < 1e5 ? real(ns.nf.nf.b) : state.eventValue[2][2]
			#############
		end

		return real(prob_ns.l1)
	end

	# the following allows to append information specific to the codim 2 continuation to the user data
	_printsol = get(kwargs, :recordFromSolution, nothing)
	_printsol2 = isnothing(_printsol) ?
		(u, p; kw...) -> (; zip(lenses, (getP(u, 𝐍𝐒)[1], p))..., ωₙₛ = getP(u, 𝐍𝐒)[2], CH = 𝐍𝐒.l1,  namedprintsol(recordFromSolution(prob)(getVec(u, 𝐍𝐒), p; kw...))...) :
		(u, p; kw...) -> (; namedprintsol(_printsol(getVec(u, 𝐍𝐒), p; kw...))..., zip(lenses, (getP(u, 𝐍𝐒)[1], p))..., ωₙₛ = getP(u, 𝐍𝐒)[2], CH = 𝐍𝐒.l1, )

	# eigen solver
	eigsolver = HopfEig(getsolver(opt_ns_cont.newtonOptions.eigsolver), prob_ns)

	prob_ns = reMake(prob_ns, recordFromSolution = _printsol2)

	# define event for detecting bifurcations. Coupled it with user passed events
	# event for detecting codim 2 points
	event_user = get(kwargs, :event, nothing)
	if isnothing(event_user)
		event = ContinuousEvent(1, testCH, computeEigenElements, ("ch",), 0)
	else
		event = PairOfEvents(ContinuousEvent(1, testCH, computeEigenElements, ("ch",), 0), event_user)
	end

	# solve the P equations
	br_ns_po = continuation(
		prob_ns, alg,
		(@set opt_ns_cont.newtonOptions.eigsolver = eigsolver);
		linearAlgo = BorderingBLS(solver = opt_ns_cont.newtonOptions.linsolver, checkPrecision = false),
		kwargs...,
		kind = kind,
		event = event,
		normC = normC,
		)
	(br_ns_po)
end