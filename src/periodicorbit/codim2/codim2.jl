function d2PO(f, x, dx1, dx2)
   return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> f(x .+ t1 .* dx1 .+ t2 .* dx2,), 0.), 0.)
end

struct FloquetWrapperBLS{T} <: AbstractBorderedLinearSolver
	solver::T # use solver as a field is good for BLS
end
(ls::FloquetWrapperBLS)(J, args...; k...) = ls.solver(J, args...; k...)
(ls::FloquetWrapperBLS)(J::FloquetWrapper, args...; k...) = ls.solver(J.jacpb, args...; k...)
Base.transpose(J::FloquetWrapper) = transpose(J.jacpb)

for op in (:NeimarkSackerProblemMinimallyAugmented,
			:PeriodDoublingProblemMinimallyAugmented)
	@eval begin
		"""
		$(TYPEDEF)

		Structure to encode functional based on a Minimally Augmented formulation.

		# Fields

		$(FIELDS)
		"""
		mutable struct $op{Tprob <: AbstractBifurcationProblem, vectype, T <: Real, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver, Tmass} <: AbstractProblemMinimallyAugmented
			"Functional F(x, p) - vector field - with all derivatives"
			prob_vf::Tprob
			"close to null vector of Jᵗ"
			a::vectype
			"close to null vector of J"
			b::vectype
			"vector zero, to avoid allocating it many times"
			zero::vectype
			"Lyapunov coefficient"
			l1::Complex{T}
			"Cusp test value"
			CP::T
			"Bogdanov-Takens test value"
			FOLDNS::T
			"Generalised period douling test value"
			GPD::T
			"Fold-NS test values"
			FLIPNS::Int
			"linear solver. Used to invert the jacobian of MA functional"
			linsolver::S
			"linear solver for the jacobian adjoint"
			linsolverAdjoint::Sa
			"bordered linear solver"
			linbdsolver::Sbd
			"linear bordered solver for the jacobian adjoint"
			linbdsolverAdjoint::Sbda
			"wether to use the hessian of prob_vf"
			usehessian::Bool
			"wether to use a mass matrix M for studying M⋅∂tu = F(u), default = I"
			massmatrix::Tmass
		end

		@inline getDelta(pb::$op) = getDelta(pb.prob_vf)
		@inline hasHessian(pb::$op) = hasHessian(pb.prob_vf)
		@inline isSymmetric(pb::$op) = isSymmetric(pb.prob_vf)
		@inline hasAdjoint(pb::$op) = hasAdjoint(pb.prob_vf)
		@inline hasAdjointMF(pb::$op) = hasAdjointMF(pb.prob_vf)
		@inline isInplace(pb::$op) = isInplace(pb.prob_vf)
		@inline getLens(pb::$op) = getLens(pb.prob_vf)
		jad(pb::$op, args...) = jad(pb.prob_vf, args...)

		# constructors
		function $op(prob, a, b, linsolve::AbstractLinearSolver, linbdsolver = MatrixBLS(); usehessian = true, massmatrix = LinearAlgebra.I)
			# determine scalar type associated to vectors a and b
			α = norm(a) # this is valid, see https://jutho.github.io/KrylovKit.jl/stable/#Package-features-and-alternatives-1
			Ty = eltype(α)
			return $op(prob, a, b, 0*a,
						complex(zero(Ty)),  # l1
						real(one(Ty)),		# cp
						real(one(Ty)),		# fold-ns
						real(one(Ty)),		# gpd
						1,					# flip-ns
						linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix)
		end

		# empty constructor, mainly used for dispatch
		function $op(prob ;linsolve = DefaultLS(), linbdsolver = MatrixBLS(), usehessian = true, massmatrix = LinearAlgebra.I)
			a = b = 0.
			α = norm(a) 
			Ty = eltype(α)
			return $op(prob, a, b, 0*a,
						complex(zero(Ty)),  # l1
						real(one(Ty)),		# cp
						real(one(Ty)),		# fold-ns
						real(one(Ty)),		# gpd
						1,					# flip-ns
						linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix)
		end
	end
end

function correctBifurcation(contres::ContResult{<: Union{FoldPeriodicOrbitCont, PDPeriodicOrbitCont, NSPeriodicOrbitCont}})
	if contres.prob.prob isa FoldProblemMinimallyAugmented
		conversion = Dict(:bp => :R1, :hopf => :foldNS, :fold => :cusp, :nd => :nd, :pd => :foldpd)
	elseif contres.prob.prob isa PeriodDoublingProblemMinimallyAugmented
		conversion = Dict(:bp => :foldFlip, :hopf => :pdNS, :pd => :R2,)
	elseif contres.prob.prob isa NeimarkSackerProblemMinimallyAugmented
		conversion = Dict(:bp => :foldNS, :hopf => :nsns, :pd => :pdNS,)
	else
		throw("Error! this should not occur. Please open an issue on the website of BifurcationKit.jl")
	end
	for (ind, bp) in pairs(contres.specialpoint)
		if bp.type in keys(conversion)
			@set! contres.specialpoint[ind].type = conversion[bp.type]
		end
	end
	return contres
end
####################################################################################################
function continuation(br::AbstractResult{Tkind, Tprob}, ind_bif::Int,
			options_cont::ContinuationPar,
			probPO::AbstractPeriodicOrbitProblem;
			detectCodim2Bifurcation::Int = 0,
			kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
	# compute the normal form of the bifurcation point
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	verbose && (println("──▶ Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif))

	nf = getNormalForm(getProb(br), br, ind_bif; detailed = true)

	# options to detect codim2 bifurcations
	_contParams = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)
	@set! _contParams.newtonOptions.eigsolver = getsolver(_contParams.newtonOptions.eigsolver)
	return _continuation(nf, br, _contParams, probPO; kwargs...)
end

function _continuation(gh::Bautin, br::AbstractResult{Tkind, Tprob},
			_contParams::ContinuationPar,
			probPO::AbstractPeriodicOrbitProblem;
			alg = br.alg,
			linearAlgo = nothing,
			δp = nothing, ampfactor::Real = 1,
			nev = _contParams.nev,
			detectCodim2Bifurcation::Int = 0,
			Teigvec = getvectortype(br),
			scaleζ = norm,
			# startWithEigen = false,
			Jᵗ = nothing,
			bdlinsolver::AbstractBorderedLinearSolver = getProb(br).prob.linbdsolver,
			kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp |> abs
	𝒯 = typeof(ds)
	pred = predictor(gh, Val(:FoldPeriodicOrbitCont), ds; verbose = verbose, ampfactor = 𝒯(ampfactor))
	pred0 = predictor(gh, Val(:FoldPeriodicOrbitCont), 0; verbose = verbose, ampfactor = 𝒯(ampfactor))

	M = getMeshSize(probPO)
	ϕ = 0
	orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]]

	# extract the vector field and use it possibly to affect the PO functional
	lens1, lens2 = gh.lens
	newparams = set(gh.params, lens1, pred.params[1])
	newparams = set(newparams, lens2, pred.params[2])

	prob_ma = getProb(br).prob
	prob_vf = reMake(prob_ma.prob_vf, params = newparams)

	# build the variable to hold the functional for computing PO based on finite differences
	probPO, orbitguess = reMake(probPO, prob_vf, gh, gh.ζ, orbitguess_a, abs(2pi/pred.ω); orbit = pred.orbit)

	verbose && printstyled(color = :green, "━"^61*
			"\n┌─ Start branching from Bautin bif. point to folds of periodic orbits.",
			"\n├─── Bautin params = ", pred0.params,
			"\n├─── new params  p = ", pred.params, ", p - p0 = ", pred.params - pred0.params,
			"\n├─── period      T = ", 2pi / pred.ω, " (from T = $(2pi / pred0.ω))",
			"\n├─ Method = \n", probPO, "\n")

	if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
		_contParams = @set _contParams.newtonOptions.linsolver.N = length(orbitguess)
	end

	# change the user provided functions by passing probPO in its parameters
	_finsol = modifyPO_2ParamsFinalise(probPO, kwargs, FoldProblemMinimallyAugmented(probPO))
	_recordsol = modifyPORecord(probPO, kwargs, getParams(probPO), getLens(probPO))
	_plotsol = modifyPOPlot(probPO, kwargs)

	jac = buildJacobian(probPO, orbitguess, getParams(probPO); δ = getDelta(prob_vf))
	pbwrap = wrap(probPO, jac, orbitguess, getParams(probPO), getLens(probPO), _plotsol, _recordsol)

	# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
	options = _contParams.newtonOptions
	_linearAlgo = isnothing(linearAlgo) ?  MatrixBLS() : linearAlgo
	linearAlgo = @set _linearAlgo.solver = FloquetWrapperLS(_linearAlgo.solver)
	alg = update(alg, _contParams, linearAlgo)

	contParams = (@set _contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver));

	# set second derivative
	probshFold = BifurcationProblem((x, p) -> residual(pbwrap, x, p), orbitguess, getParams(pbwrap), getLens(pbwrap);
				J = (x, p) -> jacobian(pbwrap, x, p),
				Jᵗ = Jᵗ,
				d2F = (x, p, dx1, dx2) -> d2PO(z -> probPO(z, p), x, dx1, dx2),
				recordFromSolution = _recordsol,
				plotSolution = _plotsol,
				)

	# create fold point guess
	foldpointguess = BorderedArray(orbitguess, get(newparams, lens1))
	
	# get the approximate null vectors
	jacpo = jacobian(probshFold, orbitguess, getParams(probshFold)).jacpb
	ls = DefaultLS()
	nj = length(orbitguess)
	p = rand(nj); q = rand(nj)
	rhs = zero(orbitguess); #rhs[end] = 1
	q, = bdlinsolver(jacpo, p, q, 0, rhs, 1); q ./= norm(q) #≈ ker(J)
	p, = bdlinsolver(jacpo', q, p, 0, rhs, 1); p ./= norm(p)

	q, = bdlinsolver(jacpo, p, q, 0, rhs, 1); q ./= norm(q) #≈ ker(J)
	p, = bdlinsolver(jacpo', q, p, 0, rhs, 1); p ./= norm(p)

	@assert sum(isnan, q) == 0 "Please report this error to the website."

	# perform continuation
	branch = continuationFold(probshFold, alg,
		foldpointguess, getParams(probshFold),
		lens1, lens2,
		p, q,
		# q, p,
		contParams;
		kind = FoldPeriodicOrbitCont(),
		kwargs...,
		bdlinsolver = FloquetWrapperBLS(bdlinsolver),
		# linearAlgo = linearAlgo,
		finaliseSolution = _finsol
	)
	return Branch(branch, gh)
end

wrap(prob::PeriodicOrbitOCollProblem, args...) = WrapPOColl(prob, args...)
wrap(prob::ShootingProblem, args...) = WrapPOSh(prob, args...)
wrap(prob::PeriodicOrbitTrapProblem, args...) = WrapPOTrap(prob, args...)

function _continuation(hh::HopfHopf, br::AbstractResult{Tkind, Tprob},
			_contParams::ContinuationPar,
			probPO::AbstractPeriodicOrbitProblem;
			whichns::Int = 1,
			alg = br.alg,
			linearAlgo = nothing,
			δp = nothing, ampfactor::Real = 1,
			nev = _contParams.nev,
			detectCodim2Bifurcation::Int = 0,
			Teigvec = getvectortype(br),
			scaleζ = norm,
			Jᵗ = nothing,
			eigsolver = FloquetQaD(getsolver(_contParams.newtonOptions.eigsolver)),
			bdlinsolver::AbstractBorderedLinearSolver = getProb(br).prob.linbdsolver,
			kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
	@assert whichns in (1, 2) "This parameter must belong to {1,2}."		
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	
	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp |> abs
	𝒯 = typeof(ds)
	pred = predictor(hh, Val(:NS), ds; verbose = verbose, ampfactor = 𝒯(ampfactor))
	pred0 = predictor(hh, Val(:NS), 0; verbose = verbose, ampfactor = 𝒯(ampfactor))

	_orbit = whichns == 1 ? pred.ns1 : pred.ns2
	period = whichns == 1 ? pred.T1 : pred.T2
	period0 = whichns == 1 ? pred0.T1 : pred0.T2

	M = getMeshSize(probPO)
	ϕ = 0
	orbitguess_a = [_orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]]

	# extract the vector field and use it possibly to affect the PO functional
	lens1, lens2 = hh.lens
	_params = whichns == 1 ? pred.params1 : pred.params2
	newparams = set(hh.params, lens1, _params[1])
	newparams = set(newparams, lens2, _params[2])

	prob_ma = getProb(br).prob
	prob_vf = reMake(prob_ma.prob_vf, params = newparams)

	@assert lens1 == getLens(prob_vf) "Please open an issue on the website of BifurcationKit"

	# build the variable to hold the functional for computing PO based on finite differences
	probPO, orbitguess = reMake(probPO, prob_vf, hh, hh.ζ.q1, orbitguess_a, period; orbit = _orbit)

	verbose && printstyled(color = :green, "━"^61*
		"\n┌─ Start branching from Hopf-Hopf bif. point to curve of Neimark-Sacker bifurcations of periodic orbits.",
		"\n├─── Hopf-Hopf params = ", pred0.params1,
		"\n├─── new params     p = ", _params, ", p - p0 = ", _params - pred0.params1,
		"\n├─── period         T = ", period, " (from T = $(period0))",
		"\n├─ Method = \n", probPO, "\n")

	if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
		_contParams = @set _contParams.newtonOptions.linsolver.N = length(orbitguess)
	end

	contParams = computeEigenElements(_contParams) ? (@set _contParams.newtonOptions.eigsolver = eigsolver) : _contParams

	# change the user provided functions by passing probPO in its parameters
	_finsol = modifyPO_2ParamsFinalise(probPO, kwargs, NeimarkSackerProblemMinimallyAugmented(probPO))
	_recordsol = modifyPORecord(probPO, kwargs, getParams(probPO), getLens(probPO))
	_plotsol = modifyPOPlot(probPO, kwargs)

	jac = buildJacobian(probPO, orbitguess, getParams(probPO); δ = getDelta(prob_vf))
	pbwrap = wrap(probPO, jac, orbitguess, getParams(probPO), getLens(probPO), _plotsol, _recordsol)

	# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
	options = _contParams.newtonOptions
	_linearAlgo = isnothing(linearAlgo) ?  MatrixBLS() : linearAlgo
	linearAlgo = @set _linearAlgo.solver = FloquetWrapperLS(_linearAlgo.solver)
	alg = update(alg, _contParams, linearAlgo)

	contParams = (@set contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver));

	# create fold point guess
	ωₙₛ = whichns == 1 ? pred.k1 : pred.k2
	nspointguess = BorderedArray(_copy(orbitguess), [get(newparams, lens1), ωₙₛ])
	
	# get the approximate null vectors
	if pbwrap isa WrapPOColl
		@debug "Collocation, get borders"
		jac = jacobian(pbwrap, orbitguess, getParams(pbwrap))
		J = Complex.(copy(jac.jacpb))
		nj = size(J, 1)
		J[end, :] .= rand(nj) #must be close to eigensapce
		J[:, end] .= rand(nj)
		J[end, end] = 0
		# enforce NS boundary condition
		N, m, Ntst = size(probPO)
		J[end-N:end-1, end-N:end-1] .= UniformScaling(cis(ωₙₛ))(N)

		rhs = zeros(nj); rhs[end] = 1
		q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) #≈ ker(J)
		p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

		@set! contParams.newtonOptions.eigsolver = FloquetColl()
	else
		@debug "Shooting, get borders"
		J = jacobianNeimarkSacker(pbwrap, orbitguess, getParams(pbwrap), ωₙₛ)
		nj = length(orbitguess)-1
		q, = bdlinsolver(J, Complex.(rand(nj)), Complex.(rand(nj)), 0, Complex.(zeros(nj)), 1)
		q ./= norm(q)
		p = conj(q)
	end

	@assert sum(isnan, q) == 0 "Please report this error to the website."

	# perform continuation
	branch = continuationNS(pbwrap, alg,
			nspointguess, getParams(pbwrap),
			lens1, lens2,
			p, q,
			# q, p,
			contParams;
			kind = NSPeriodicOrbitCont(),
			kwargs...,
			# linearAlgo = linearAlgo,
			plotSolution = _plotsol,
			bdlinsolver = FloquetWrapperBLS(bdlinsolver),
			finaliseSolution = _finsol
	)
	return Branch(branch, hh)
end