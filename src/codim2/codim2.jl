abstract type AbstractProblemMinimallyAugmented end
abstract type AbstractCodim2EigenSolver <: AbstractEigenSolver end

# @inline getLens(pb::AbstractProblemMinimallyAugmented) = pb.lens
getsolver(eig::AbstractCodim2EigenSolver) = eig.eigsolver
# @inline issymmetric(pb::AbstractProblemMinimallyAugmented) = pb.issymmetric

for op in (:FoldProblemMinimallyAugmented, :HopfProblemMinimallyAugmented)
	@eval begin
		"""
		$(TYPEDEF)

		Structure to encode Fold / Hopf functional based on a Minimally Augmented formulation.

		# Fields

		$(FIELDS)
		"""
		struct $op{Tprob <: AbstractBifurcationProblem, vectype, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver} <: AbstractProblemMinimallyAugmented
			"Functional F(x, p) - vector field - with all derivatives"
			prob_vf::Tprob
			"close to null vector of Jᵗ"
			a::vectype
			"close to null vector of J"
			b::vectype
			"vector zero, to avoid allocating it many times"
			zero::vectype
			"linear solver. Used to invert the jacobian of MA functional"
			linsolver::S
			"linear solver for the jacobian adjoint"
			linsolverAdjoint::Sa
			"bordered linear solver"
			linbdsolver::Sbd
			"linear bordered solver for the jacobian adjoint"
			linbdsolverAdjoint::Sbda
		end

		@inline hasHessian(pb::$op) = hasHessian(pb.prob_vf)
		@inline isSymmetric(pb::$op) = isSymmetric(pb.prob_vf)
		@inline hasAdjoint(pb::$op) = hasAdjoint(pb.prob_vf)
		@inline isInplace(pb::$op) = isInplace(pb.prob_vf)
		@inline getLens(pb::$op) = getLens(pb.prob_vf)

		# constructor
		$op(prob, a, b, linsolve::AbstractLinearSolver, linbdsolver = MatrixBLS()) = $op(prob, a, b, 0*a, linsolve, linsolve, linbdsolver, linbdsolver)
	end
end

function detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)
	if detectCodim2Bifurcation > 0
		if get(kwargs, :updateMinAugEveryStep, 0) == 0
			@error "You ask for detection of codim 2 bifurcations but passed the option `updateMinAugEveryStep = 0`. The bifurcation detection algorithm may not work faithfully. Please use `updateMinAugEveryStep > 0`."
		end
		return setproperties(options_cont; detectBifurcation = 0, detectEvent = detectCodim2Bifurcation, detectFold = false)
	else
		return options_cont
	end
end

"""
$(SIGNATURES)

This function turns an initial guess for a Fold/Hopf point into a solution to the Fold/Hopf problem based on a Minimally Augmented formulation.

## Arguments
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`

# Optional arguments:
- `options::NewtonPar`, default value `br.contparams.newtonOptions`
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newton(br::AbstractBranchResult, ind_bif::Int64; normN = norm, options = br.contparams.newtonOptions, startWithEigen = false, kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	if br.specialpoint[ind_bif].type == :hopf
		return newtonHopf(br, ind_bif; normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	else
		return newtonFold(br, ind_bif; normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	end
end

"""
$(SIGNATURES)

Codimension 2 continuation of Fold / Hopf points. This function turns an initial guess for a Fold/Hopf point into a curve of Fold/Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`
- `lens2` second parameter used for the continuation, the first one is the one used to compute `br`, e.g. `getLens(br)`
- `options_cont = br.contparams` arguments to be passed to the regular [continuation](@ref Library-Continuation)

# Optional arguments:
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a,b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `detectCodim2Bifurcation ∈ {0,1,2}` whether to detect Bogdanov-Takens, Bautin and Cusp. If equals `1` non precise detection is used. If equals `2`, a bisection method is used to locate the bifurcations.
- `kwargs` keywords arguments to be passed to the regular [continuation](@ref Library-Continuation)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function continuation(br::AbstractBranchResult, ind_bif::Int64,
				lens2::Lens, options_cont::ContinuationPar = br.contparams ;
				startWithEigen = false,
				detectCodim2Bifurcation::Int = 0,
				kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	# options to detect codim2 bifurcations
	computeEigenElements = options_cont.detectBifurcation > 0
	_options_cont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

	if br.specialpoint[ind_bif].type == :hopf
		return continuationHopf(br.prob, br, ind_bif, lens2, _options_cont;
			startWithEigen = startWithEigen,
			computeEigenElements = computeEigenElements,
			kwargs...)
	else
		return continuationFold(br.prob, br, ind_bif, lens2, _options_cont;
			startWithEigen = startWithEigen,
			computeEigenElements = computeEigenElements,
			kwargs...)
	end
end
####################################################################################################
# branch switching at bt
function continuation(br::AbstractResult{Tkind, Tprob}, ind_bif::Int,
			options_cont::ContinuationPar = br.contparams;
			alg = br.alg,
			δp = nothing, ampfactor::Real = 1,
			nev = options_cont.nev,
			detectCodim2Bifurcation::Int = 0,
			Teigvec = getvectortype(br),
			scaleζ = norm,
			startWithEigen = false,
			autodiff = false,
			kwargs...) where {Tkind, Tprob <: Union{FoldMAProblem, HopfMAProblem}}

		verbose = get(kwargs, :verbosity, 0) > 0 ? true : false
		verbose && println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif)

		@assert br.specialpoint[ind_bif].type in (:bt,:zh) "Only branching from Bogdanov-Takens and Zero-Hopf (for now)"

		# functional
		prob_ma = br.prob.prob
		prob_vf = prob_ma.prob_vf

		# continuation parameters
		computeEigenElements = options_cont.detectBifurcation > 0
		optionsCont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

		# scalar type
		Ty = eltype(Teigvec)

		# compute the normal form of the bifurcation point
		nf = getNormalForm(br, ind_bif; nev = nev, verbose = verbose, Teigvec = Teigvec, scaleζ = scaleζ, autodiff = autodiff)

		# compute predictor for point on new branch
		ds = isnothing(δp) ? optionsCont.ds : δp

		if prob_ma isa FoldProblemMinimallyAugmented
			# define guess for the first Hopf point on the branch
			pred = predictor(nf, Val(:HopfCurve), ds)

			# new continuation parameters
			parcont = pred.hopf(ds)

			# new full parameters
			params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

			# guess for the Hopf point
			hopfpt = BorderedArray(nf.x0 .+ pred.x0(ds), [parcont[1], pred.ω(ds)])

			# estimates for eigenvectors for ±iω
			ζ = pred.EigenVec(ds)
			ζstar = pred.EigenVecAd(ds)

			# put back original options
			@set! optionsCont.newtonOptions.eigsolver =
								getsolver(optionsCont.newtonOptions.eigsolver)
			@set! optionsCont.newtonOptions.linsolver = prob_ma.linsolver

			branch = continuationHopf(prob_vf, alg,
					hopfpt, params,
					nf.lens...,
					ζ, ζstar,
					optionsCont;
					bdlinsolver = prob_ma.linbdsolver,
					startWithEigen = startWithEigen,
					computeEigenElements = computeEigenElements,
					kwargs...
					)
			return Branch(branch, nf)

		else
			@assert prob_ma isa HopfProblemMinimallyAugmented
			pred = predictor(nf, Val(:FoldCurve), 0.)
			# new continuation parameters
			parcont = pred.fold(ds)

			# new full parameters
			params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

			# guess for the fold point
			foldpt = BorderedArray(nf.x0 .+ 0 .* pred.x0(ds), parcont[1])

			# estimates for null eigenvectors
			ζ = pred.EigenVec(ds)
			ζstar = pred.EigenVecAd(ds)

			# put back original options
			@set! optionsCont.newtonOptions.eigsolver =
								getsolver(optionsCont.newtonOptions.eigsolver)
			@set! optionsCont.newtonOptions.linsolver = prob_ma.linsolver
			# @set! optionsCont.detectBifurcation = 0
			# @set! optionsCont.detectEvent = 0

			branch = continuationFold(prob_vf, alg,
					foldpt, params,
					nf.lens...,
					ζstar, ζ,
					optionsCont;
					bdlinsolver = prob_ma.linbdsolver,
					startWithEigen = startWithEigen,
					computeEigenElements = computeEigenElements,
					kwargs...
					)
			return Branch(branch, nf)
		end
end

"""
$(SIGNATURES)

This function uses information in the branch to detect codim 2 bifurcations like BT, ZH and Cusp.
"""
function correctBifurcation(contres::ContResult)
	if contres.prob.prob isa AbstractProblemMinimallyAugmented == false
		return contres
	end
	if contres.prob.prob isa FoldProblemMinimallyAugmented
		conversion = Dict(:bp => :bt, :hopf => :zh, :fold => :cusp, :nd => :nd)
	elseif contres.prob.prob isa HopfProblemMinimallyAugmented
		conversion = Dict(:bp => :zh, :hopf => :hh, :fold => :nd, :nd => :nd, :ghbt => :bt, :btgh => :bt)
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
