import Base: iterate
####################################################################################################
# Iterator interface
"""
$(TYPEDEF)

# Useful functions
- `setParam(iter, p)` set parameter with lens `iter.prob.lens` to `p`
- `isEventActive(iter)` whether the event detection is active
- `computeEigenElements(iter)` whether to compute eigen elements
- `saveEigenvectors(iter)` whether to save eigen vectors
- `getParams(iter)` get full list of params
- `length(iter)`
- `isInDomain(iter, p)` whether `p` in is domain [pMin, pMax]. (See [`ContinuationPar`](@ref))
- `isOnBoundary(iter, p)` whether `p` in is {pMin, pMax}
"""
@with_kw_noshow struct ContIterable{Tkind <: AbstractContinuationKind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent} <: AbstractContinuationIterable{Tkind}
	kind::Tkind
	prob::Tprob
	alg::Talg
	contParams::ContinuationPar{T, S, E}
	plot::Bool = false
	event::Tevent = nothing
	normC::TnormC
	finaliseSolution::Tfinalisesolution
	callbackN::TcallbackN
	verbosity::Int64 = 2
	filename::String
end

# default finalizer
finaliseDefault(z, tau, step, contResult; k...) = true

# constructor
function ContIterable(prob::AbstractBifurcationProblem,
					alg::AbstractContinuationAlgorithm,
					contParams::ContinuationPar{T, S, E};
					kind = EquilibriumCont(),
					filename = "branch-" * string(Dates.now()),
					plot = false,
					normC = norm,
					finaliseSolution = finaliseDefault,
					callbackN = cbDefault,
					event = nothing,
					verbosity = 0, kwargs...
					) where {T <: Real, S, E}

	return ContIterable(kind = kind,
				prob = prob,
				alg = alg,
				contParams = contParams,
				plot = plot,
				normC = normC,
				finaliseSolution = finaliseSolution,
				callbackN = callbackN,
				event = event,
				verbosity = verbosity,
				filename = filename)
end

Base.eltype(it::ContIterable{Tkind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent}) where {Tkind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent} = T

setParam(it::ContIterable{Tkind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent}, p0::T) where {Tkind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent} = setParam(it.prob, p0)

# getters
@inline getLens(it::ContIterable) = getLens(it.prob)
@inline getAlg(it::ContIterable) = it.alg
@inline callback(it::ContIterable) = it.callbackN
recordFromSolution(it::ContIterable) = recordFromSolution(it.prob)
plotSolution(it::ContIterable) = plotSolution(it.prob)

getLensSymbol(it::ContIterable) = getLensSymbol(getLens(it))

# get the linear solver for Continuation
getLinsolver(iter::ContIterable) = getLinsolver(iter.alg)

@inline isEventActive(it::ContIterable) = !isnothing(it.event) && it.contParams.detectEvent > 0
@inline computeEigenElements(it::ContIterable) = computeEigenElements(it.contParams) || (isEventActive(it) && computeEigenElements(it.event))
@inline saveEigenvectors(it::ContIterable) = saveEigenvectors(it.contParams)

@inline getContParams(it::ContIterable) = it.contParams
Base.length(it::ContIterable) = it.contParams.maxSteps
@inline isInDomain(it::ContIterable, p) = it.contParams.pMin < p < it.contParams.pMax
@inline isOnBoundary(it::ContIterable, p) = (it.contParams.pMin == p) || (p == it.contParams.pMax)
# clamp p value
clampPredp(p::Number, it::AbstractContinuationIterable) = clamp(p, it.contParams.pMin, it.contParams.pMax)
####################################################################################################
"""
	state = ContState(ds = 1e-4,...)

Returns a variable containing the state of the continuation procedure. The fields are meant to change during the continuation procedure.

# Arguments
- `z_pred` current solution on the branch
- `converged` Boolean for newton correction
- `τ` tangent predictor
- `z` previous solution
- `itnewton` Number of newton iteration (in corrector)
- `step` current continuation step
- `ds` step size
- `stopcontinuation` Boolean to stop continuation

# Useful functions
- `copy(state)` returns a copy of `state`
- `copyto!(dest, state)` returns a copy of `state`
- `getSolution(state)` returns the current solution (x, p)
- `getx(state)` returns the x component of the current solution
- `getp(state)` returns the p component of the current solution
- `getpreviousp(state)` returns the p component of the previous solution
- `isStable(state)` whether the current state is stable
"""
@with_kw_noshow mutable struct ContState{Tv, T, Teigvals, Teigvec, Tcb} <: AbstractContinuationState{Tv}
	z_pred::Tv								# predictor
	τ::Tv									# tangent to the curve
	z::Tv									# current solution
	z_old::Tv								# previous solution

	converged::Bool							# boolean for newton correction
	itnewton::Int64 = 0						# number of newton iteration (in corrector)
	itlinear::Int64 = 0						# number of linear iteration (in newton corrector)

	step::Int64 = 0							# current continuation step
	ds::T									# step size

	stopcontinuation::Bool = false			# boolean to stop continuation
	stepsizecontrol::Bool = true			# perform step size adaptation

	# the following values encode the current, previous number of unstable (resp. imaginary) eigen values
	# it is initialized as -1 when unknown
	n_unstable::Tuple{Int64,Int64}  = (-1, -1)	# (current, previous)
	n_imag::Tuple{Int64,Int64} 		= (-1, -1)	# (current, previous)
	convergedEig::Bool				= true

	eigvals::Teigvals = nothing				# current eigenvalues
	eigvecs::Teigvec = nothing				# current eigenvectors

	eventValue::Tcb = nothing
end

function Base.copy(state::ContState)
	return ContState(
		z_pred 	= _copy(state.z_pred),
		τ 	= _copy(state.τ),
		z 	= _copy(state.z),
		z_old 	= _copy(state.z_old),
		converged = state.converged,
		itnewton 	= state.itnewton,
		step 		= state.step,
		ds 			= state.ds,
		stopcontinuation = state.stopcontinuation,
		stepsizecontrol  = state.stepsizecontrol,
		n_unstable 		 = state.n_unstable,
		n_imag 			 = state.n_imag,
		eventValue		 = state.eventValue,
		eigvals			 = state.eigvals,
		eigvecs			 = state.eigvecs # can be removed? to save memory?
	)
end

function Base.copyto!(dest::ContState, src::ContState)
		copyto!(dest.z_pred , src.z_pred)
		copyto!(dest.τ , src.τ)
		copyto!(dest.z , src.z)
		copyto!(dest.z_old , src.z_old)
		dest.converged 	= src.converged
		dest.itnewton 	= src.itnewton
		dest.step 		= src.step
		dest.ds 		= src.ds
		dest.stopcontinuation = src.stopcontinuation
		dest.stepsizecontrol  = src.stepsizecontrol
		dest.n_unstable 	  = src.n_unstable
		dest.n_imag 		  = src.n_imag
		dest.eventValue		  = src.eventValue
		dest.eigvals		  = src.eigvals
		dest.eigvecs		  = src.eigvecs # can be removed? to save memory?
	return dest
end

# getters
@inline converged(state::AbstractContinuationState) = state.converged
getSolution(state::AbstractContinuationState) 	= state.z
getx(state::AbstractContinuationState) 			= state.z.u
@inline getp(state::AbstractContinuationState)  = state.z.p
@inline getpreviousp(state::AbstractContinuationState) = state.z_old.p
@inline isStable(state::AbstractContinuationState) = state.n_unstable[1] == 0
@inline stepsizecontrol(state::AbstractContinuationState) = state.stepsizecontrol
####################################################################################################
# condition for halting the continuation procedure (i.e. when returning false)
@inline done(it::ContIterable, state::ContState) =
			(state.step <= it.contParams.maxSteps) &&
			(isInDomain(it, getp(state)) || state.step == 0) &&
			(state.stopcontinuation == false)

function getStateSummary(it, state)
	x = getx(state)
	p = getp(state)
	pt = recordFromSolution(it)(x, p)
	stable = computeEigenElements(it) ? isStable(state) : nothing
	return mergefromuser(pt, (param = p, itnewton = state.itnewton, itlinear = state.itlinear, ds = state.ds, n_unstable = state.n_unstable[1], n_imag = state.n_imag[1], stable = stable, step = state.step))
end

function updateStability!(state::ContState, n_unstable::Int, n_imag::Int, converged::Bool)
	state.n_unstable = (n_unstable, state.n_unstable[1])
	state.n_imag = (n_imag, state.n_imag[1])
	state.convergedEig = converged
end

function save!(br::ContResult, it::AbstractContinuationIterable, state::AbstractContinuationState)
	# update branch field
	push!(br.branch, getStateSummary(it, state))
	# save solution
	if it.contParams.saveSolEveryStep > 0 && (modCounter(state.step, it.contParams.saveSolEveryStep) || ~done(it, state))
		push!(br.sol, (x = getSolution(it.prob, _copy(getx(state))), p = getp(state), step = state.step))
	end
	# save eigen elements
	if computeEigenElements(it)
		if mod(state.step, it.contParams.saveEigEveryStep) == 0
			push!(br.eig, (eigenvals = state.eigvals, eigenvecs = state.eigvecs, converged = state.convergedEig, step = state.step))
		end
	end
end

function plotBranchCont(contres::ContResult, state::AbstractContinuationState, iter::ContIterable)
	if iter.plot && mod(state.step, getContParams(iter).plotEveryStep) == 0
		return plotBranchCont(contres, getSolution(state), getContParams(iter), plotSolution(iter))
	end
end

function ContResult(it::AbstractContinuationIterable, state::AbstractContinuationState)
	x0 = _copy(getx(state))
	p0 = getp(state)
	pt = recordFromSolution(it)(x0, p0)
	eiginfo = computeEigenElements(it) ? (state.eigvals, state.eigvecs) : nothing
	return _ContResult(it.prob, getAlg(it), pt, getStateSummary(it, state), getSolution(it.prob, x0), state.τ, eiginfo, getContParams(it), computeEigenElements(it), it.kind)
end

# function to update the state according to the event
function updateEvent!(it::ContIterable, state::ContState)
	# if the event is not active, we return false (not detected)
	if (isEventActive(it) == false) return false; end
	outcb = it.event(it, state)
	state.eventValue = (outcb, state.eventValue[1])
	# update number of positive values
	return isEventCrossed(it.event, it, state)
end
####################################################################################################
# Continuation Iterator
#
# function called at the beginning of the continuation
# used to determine first point on branch and tangent at this point
function Base.iterate(it::ContIterable; _verbosity = it.verbosity)
	# the keyword argument is to overwrite verbosity behaviour, like when locating bifurcations
	verbose = min(it.verbosity, _verbosity) > 0
	prob = it.prob
	p₀ = getParam(prob)

	T = eltype(it)

	verbose && printstyled("━"^55*"\n"*"─"^18*" ",typeof(getAlg(it)).name.name," "*"─"^18*"\n\n", bold = true, color = :red)

	# newton parameters
	@unpack pMin, pMax, maxSteps, newtonOptions, η, ds = it.contParams
	if !(pMin <= p₀ <= pMax)
		@error "Initial parameter $p₀ must be within bounds [$pMin, $pMax]"
		return nothing
	end

	# apply Newton algo to initial guess
	verbose && printstyled("━"^18*"  INITIAL GUESS   "*"━"^18, bold = true, color = :magenta)

	# we pass additional kwargs to newton so that it is sent to the newton callback
	sol₀ = newton(prob, newtonOptions; normN = it.normC, callback = callback(it), iterationC = 0, p = p₀)
	if  ~converged(sol₀)
		println("Newton failed to converge the initial guess on the branch.")
		display(sol₀.residuals)
		throw("")
	end
	verbose && (print("\n──▶ convergence of initial guess = ");printstyled("OK\n\n", color=:green))
	verbose && println("──▶ parameter = ", p₀, ", initial step")
	verbose && printstyled("\n"*"━"^18*" INITIAL TANGENT  "*"━"^18, bold = true, color = :magenta)
	sol₁ = newton(reMake(prob; params = setParam(it, p₀ + ds / η), u0 = sol₀.u),
			newtonOptions; normN = it.normC, callback = callback(it), iterationC = 0, p = p₀ + ds / η)
	@assert converged(sol₁) "Newton failed to converge. Required for the computation of the initial tangent."
	verbose && (print("\n──▶ convergence of the initial guess = ");printstyled("OK\n\n", color=:green))
	verbose && println("──▶ parameter = ", p₀ + ds/η, ", initial step (bis)")
	return iterateFromTwoPoints(it, sol₀.u, p₀, sol₁.u, p₀ + ds / η; _verbosity = _verbosity)
end

# same as previous function but when two (initial guesses) points are provided
function iterateFromTwoPoints(it::ContIterable, u₀, p₀::T, u₁, p₁::T; _verbosity = it.verbosity) where T
	ds = it.contParams.ds

	# compute eigenvalues to get the type. Necessary to give a ContResult
	if computeEigenElements(it)
		eigvals, eigvecs, cveig, = computeEigenvalues(it, nothing, u₀, getParams(it.prob), it.contParams.nev)
		if ~saveEigenvectors(it)
			eigvecs = nothing
		end
	else
		eigvals, eigvecs = nothing, nothing
	end

	# compute event value and store into state
	cbval = isEventActive(it) ? initialize(it.event, T) : nothing # event result
	state = ContState(z_pred = BorderedArray(_copy(u₀), p₀),
						τ = BorderedArray(0*u₁, zero(p₁)),
						z = BorderedArray(_copy(u₁), p₁),
						z_old = BorderedArray(_copy(u₀), p₀),
						converged = true,
						ds = it.contParams.ds,
						eigvals = eigvals,
						eigvecs = eigvecs,
						eventValue = (cbval, cbval))

	# compute the state for the continuation algorithm, at this point the tangent is set up
	initialize!(state, it)

	# update stability
	if computeEigenElements(it)
		@unpack n_unstable, n_imag = isStable(getContParams(it), eigvals)
		updateStability!(state, n_unstable, n_imag, cveig)
	end

	# we update the event function result
	updateEvent!(it, state)
	return state, state
end

function Base.iterate(it::ContIterable, state::ContState; _verbosity = it.verbosity)
	if !done(it, state) return nothing end
	# next line is to overwrite verbosity behaviour, for example when locating bifurcations
	verbosity = min(it.verbosity, _verbosity)
	verbose = verbosity > 0; verbose1 = verbosity > 1

	@unpack step, ds = state

	if verbose
		printstyled("─"^55*"\nContinuation Step $step \n", bold = true);
		@printf("Step size = %2.4e\n", ds); print("Parameter ", getLensSymbol(it))
		@printf(" = %2.4e ⟶  %2.4e [guess]\n", getp(state), clampPredp(state.z_pred.p, it))
	end

	# in PALC, z_pred contains the previous solution
	corrector!(state, it; iterationC = step, z0 = state.z)

	if converged(state)
		if verbose
			verbose1 && printstyled("──▶ Step Converged in $(state.itnewton) Nonlinear Iteration(s)\n", color = :green)
			print("Parameter ", getLensSymbol(it))
			@printf(" = %2.4e ⟶  %2.4e\n", state.z_old.p, getp(state))
		end

		# Eigen-elements computation, they are stored in state
		if computeEigenElements(it)
			# this computes eigen-elements, store them in state and update the stability indices in state
			it_eigen = computeEigenvalues!(it, state)
			verbose1 && printstyled(color=:green,"──▶ Computed ", length(state.eigvals), " eigenvalues in ", it_eigen, " iterations, #unstable = ", state.n_unstable[1], "\n")
		end
		state.step += 1
	else
		verbose && printstyled("Newton correction failed\n", color = :red)
	end

	# step size control
	# we update the parameters ds stored in state
	stepSizeControl!(state, it)

	# predictor: state.z_pred. The following method only mutates z_pred and τ
	getPredictor!(state, it)

	return state, state
end

function continuation!(it::ContIterable, state::ContState, contRes::ContResult)
	contParams = getContParams(it)
	verbose = it.verbosity > 0
	verbose1 = it.verbosity > 1

	next = (state, state)

	while ~isnothing(next)
		# get the current state
		_, state = next
		########################################################################################
		# the new solution has been successfully computed
		# we perform saving, plotting, computation of eigenvalues...
		# the case state.step = 0 was just done above
		if converged(state) && (state.step <= it.contParams.maxSteps) && (state.step > 0)
			# Detection of fold points based on parameter monotony, mutates contRes.specialpoint
			# if we detect bifurcations based on eigenvalues, we disable fold detection to avoid duplicates
			if contParams.detectFold && contParams.detectBifurcation < 2
				foldetected = locateFold!(contRes, it, state)
				if foldetected && contParams.detectLoop
					state.stopcontinuation |= detectLoop(contRes, nothing; verbose = verbose1)
				end
			end

			if contParams.detectBifurcation > 1 && detectBifucation(state)
				status::Symbol = :guess
				_T = eltype(it)
				interval::Tuple{_T, _T} = getinterval(getpreviousp(state), getp(state))
				# if the detected bifurcation point involves a parameter values with is on
				# the boundary of the parameter domain, we disable bisection because it would
				# lead to infinite looping. Indeed, clamping messes up the `ds`
				if contParams.detectBifurcation > 2 && ~isOnBoundary(it, getp(state))
					verbose1 && printstyled(color = :red, "──▶ Bifurcation detected before p = ", getp(state), "\n")
					# locate bifurcations with bisection, mutates state so that it stays very close the bifurcation point. It also updates the eigenelements at the current state. The call returns :guess or :converged
					status, interval = locateBifurcation!(it, state, it.verbosity > 2)
				end
				# we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
				if detectBifucation(state)
					_, bifpt = getBifurcationType(it, state, status, interval)
					if bifpt.type != :none; push!(contRes.specialpoint, bifpt); end
					# detect loop in the branch
					contParams.detectLoop && (state.stopcontinuation |= detectLoop(contRes, bifpt))
				end
			end

			if isEventActive(it)
				# check if an event occurred between the 2 continuation steps
				eveDetected = updateEvent!(it, state)
				verbose1 && printstyled(color = :blue, "──▶ Event values: ", state.eventValue[2], "\n"*" "^14*"──▶ ", state.eventValue[1],"\n")
				eveDetected && (verbose && printstyled(color=:red, "──▶ Event detected before p = ", getp(state), "\n"))
				# save the event if detected and / or use bisection to locate it precisely
				if eveDetected
					_T = eltype(it); status = :guess; intervalevent = (_T(0),_T(0))
					if contParams.detectEvent > 1
						# interval = getinterval(state.z_pred.p, getp(state))::Tuple{_T, _T}
						status, intervalevent = locateEvent!(it.event, it, state, it.verbosity > 2)
					end
					success, bifpt = getEventType(it.event, it, state, it.verbosity, status, intervalevent)
					state.stopcontinuation |= ~success
					if bifpt.type != :none; push!(contRes.specialpoint, bifpt); end
					# detect loop in the branch
					contParams.detectLoop && (state.stopcontinuation |= detectLoop(contRes, bifpt))
				end
			end

			# save solution to file
			contParams.saveToFile && saveToFile(it, getx(state), getp(state), state.step, contRes)

			# call user saved finaliseSolution function. If returns false, stop continuation
			# we put a OR to stop continuation if the stop was required before
			state.stopcontinuation |= ~it.finaliseSolution(getSolution(state), state.τ, state.step, contRes; state = state, iter = it)

			# save current state in the branch
			save!(contRes, it, state)

			# plot current state
			plotBranchCont(contRes, state, it)
		end
		########################################################################################
		# body
		next = iterate(it, state)
	end

	it.plot && plotBranchCont(contRes, state.z, contParams, plotSolution(it))

	# return current solution in case the corrector did not converge
	push!(contRes.specialpoint, SpecialPoint(it, state, :endpoint, :converged, (getp(state), getp(state))))
	return contRes
end

function continuation(it::ContIterable)
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The return type of this method, e.g. ContResult
	# is not known at compile time so we
	# use a function barrier to resolve it
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	# we compute the cache for the continuation, i.e. state::ContState
	# In this call, we also compute the initial point on the branch (and its stability) and the initial tangent
	states = iterate(it)
	isnothing(states) && return nothing

	# variable to hold the result from continuation, i.e. a branch
	contRes = ContResult(it, states[1])

	# perform the continuation
	return continuation!(it, states[1], contRes)
end
####################################################################################################

"""
$(SIGNATURES)

Compute the continuation curve associated to the functional `F` which is stored in the bifurcation problem `prob`. General information is available in [Continuation methods: introduction](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/IntroContinuation/).

# Arguments:
- `prob::AbstractBifurcationFunction` a `::AbstractBifurcationProblem`, typically a  [`BifurcationProblem`](@ref) which holds the vector field and its jacobian. We also refer to  [`BifFunction`](@ref) for more details.
- `alg` continuation algorithm, for example `Natural(), PALC(), Multiple(),...`. See [algos](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/Predictors/)
- `contParams` parameters for continuation. See [`ContinuationPar`](@ref)

# Optional Arguments:
- `plot = false` whether to plot the solution/branch/spectrum while computing the branch
- `bothside = true` compute the branches on the two sides of `p0`, merge them and return it.
- `finaliseSolution = (z, tau, step, contResult; kwargs...) -> true` Function called at the end of each continuation step. Can be used to alter the continuation procedure (stop it by returning `false`), saving personal data, plotting... The notations are ``z=(x, p)`` where `x` (resp. `p`) is the current solution (resp. parameter value), `tau` is the tangent at `z`, `step` is the index of the current continuation step and `ContResult` is the current branch. For advanced use, the current `state::ContState` of the continuation is passed in `kwargs`. Note that you can have a better control over the continuation procedure by using an iterator, see [Iterator Interface](@ref).
- `verbosity::Int = 0` controls the amount of information printed during the continuation process. Must belong to `{0,1,2,3}`. In case `contParams.newtonOptions.verbose = false`, the following is valid (otherwise the newton iterations are shown). Each case prints more information than the previous one:
    - case 0: print nothing
    - case 1: print basic information about the continuation: used predictor, step size and parameter values
    - case 2: print newton iterations number, stability of solution, detected bifurcations / events
    - case 3: print information during bisection to locate bifurcations / events
- `normC = norm` norm used in the Newton solves
- `filename` to save the computed branch during continuation. The identifier .jld2 will be appended to this filename. This requires `using JLD2`.
- `callbackN` callback for newton iterations. See docs for [`newton`](@ref). For example, it can be used to change preconditioners.
- `kind::AbstractContinuationKind` [Internal] flag to describe continuation kind (equilibrium, codim 2, ...). Default = `EquilibriumCont()`

# Output:
- `contres::ContResult` composite type which contains the computed branch. See [`ContResult`](@ref) for more information.

!!! tip "Continuing the branch in the opposite direction"
    Just change the sign of `ds` in `ContinuationPar`.
"""
function continuation(prob::AbstractBifurcationProblem,
						alg::AbstractContinuationAlgorithm,
						contParams::ContinuationPar;
						linearAlgo = nothing,
						bothside::Bool = false,
						kwargs...)
	# create a bordered linear solver using the newton linear solver provided by the user
	alg = update(alg, contParams, linearAlgo)

	# perform continuation
	itfwd = ContIterable(prob, alg, contParams; kwargs...)
	if bothside
		itbwd = ContIterable(deepcopy(prob), deepcopy(alg), contParams; kwargs...)
		@set! itbwd.contParams.ds = -contParams.ds

		resfwd = continuation(itfwd)
		resbwd = continuation(itbwd)
		contresult = _merge(resfwd, resbwd)

		# we have to update the branch if saved on a file
		itfwd.contParams.saveToFile && saveToFile(itfwd, contresult)

		return contresult

	else
		contresult = continuation(itfwd)
		# we have to update the branch if saved on a file,
		# basically this removes "branchfw" or "branchbw" in file and append "branch"
		itfwd.contParams.saveToFile && saveToFile(itfwd, contresult[1])
		return contresult
	end
end
