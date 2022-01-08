import Base: iterate
abstract type AbstractContinuationIterable end
abstract type AbstractContinuationState end
####################################################################################################
# Iterator interface
"""
$(TYPEDEF)

# Useful functions
- `setParam(iter, p)` set parameter with lens `iter.lens` to `p`
- `isEventActive(iter)` whether the event detection is active
- `computeEigenElements(iter)` whether to compute eigen elements
- `saveEigenvectors(iter)` whether to save eigen vectors
- `getParams(iter)` get full list of params
- `length(iter)`
- `isInDomain(iter, p)` whether `p` in is domain [pMin, pMax]. (See [`ContinuationPar`](@ref))
- `isOnBoundary(iter, p)` whether `p` in is {pMin, pMax}
"""
@with_kw_noshow struct ContIterable{TF, TJ, Tv, Tp, Tlens, T, S, E, Ttangent, Tlinear, Tplotsolution, Tprintsolution, TnormC, Tdot, Tfinalisesolution, TcallbackN, Tevent, Tfilename} <: AbstractContinuationIterable
	F::TF
	J::TJ

	x0::Tv			# initial guess
	par::Tp			# reference to parameter, so no worry if this one is big like sparse matrix
	lens::Tlens		# param axis to be considered specified by a ::Lens

	contParams::ContinuationPar{T, S, E}

	tangentAlgo::Ttangent
	linearAlgo::Tlinear

	plot::Bool = false
	plotSolution::Tplotsolution
	recordFromSolution::Tprintsolution
	event::Tevent = nothing

	normC::TnormC
	dottheta::Tdot
	finaliseSolution::Tfinalisesolution
	callbackN::TcallbackN

	verbosity::Int64 = 2

	filename::Tfilename
end

# default finalizer
finaliseDefault(z, tau, step, contResult; k...) = true

# constructor
function ContIterable(Fhandle, Jhandle,
					x0, par, lens::Lens,
					contParams::ContinuationPar{T, S, E},
					linearAlgo::AbstractBorderedLinearSolver = BorderingBLS(DefaultLS());
					filename = "branch-" * string(Dates.now()),
					tangentAlgo = SecantPred(),
					plot = false,
					plotSolution = (x, p; kwargs...) -> nothing,
					recordFromSolution = (x, p; kwargs...) -> norm(x),
					normC = norm,
					dotPALC = (x,y) -> dot(x,y) / length(x),
					finaliseSolution = finaliseDefault,
					callbackN = cbDefault,
					event = nothing,
					verbosity = 0, kwargs...
					) where {T <: Real, S, E}

	return ContIterable(F = Fhandle, J = Jhandle,
				x0 = x0, par = par, lens = lens,
				contParams = contParams,
				tangentAlgo = tangentAlgo,
				linearAlgo = linearAlgo,
				plot = plot,
				plotSolution = plotSolution,
				recordFromSolution = recordFromSolution,
				normC = normC,
				dottheta = DotTheta(dotPALC),
				finaliseSolution = finaliseSolution,
				callbackN = callbackN,
				event = event,
				verbosity = verbosity,
				filename = filename)
end

Base.eltype(it::ContIterable{TF, TJ, Tv, Tp, Tlens, T, S, E, Ttangent, Tlinear, Tplotsolution, Tprintsolution, TnormC, Tdot, Tfinalisesolution, TcallbackN, Tevent, Tfilename}) where {TF, TJ, Tv, Tp, Tlens, T, S, E, Ttangent, Tlinear, Tplotsolution, Tprintsolution, TnormC, Tdot, Tfinalisesolution, TcallbackN, Tevent, Tfilename} = T

setParam(it::ContIterable{TF, TJ, Tv, Tp, Tlens, T, S, E, Ttangent, Tlinear, Tplotsolution, Tprintsolution, TnormC, Tdot, Tfinalisesolution, TcallbackN, Tevent, Tfilename}, p0::T) where {TF, TJ, Tv, Tp, Tlens, T, S, E, Ttangent, Tlinear, Tplotsolution, Tprintsolution, TnormC, Tdot, Tfinalisesolution, TcallbackN, Tevent, Tfilename} = set(it.par, it.lens, p0)

@inline isEventActive(it::ContIterable) = !isnothing(it.event) && it.contParams.detectEvent > 0
@inline computeEigenElements(it::ContIterable) = computeEigenElements(it.contParams) || (isEventActive(it) && computeEigenElements(it.event))
@inline saveEigenvectors(it::ContIterable) = saveEigenvectors(it.contParams)

@inline getParams(it::ContIterable) = it.contParams
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
- `tau` tangent predictor
- `z_old` previous solution
- `isconverged` Boolean for newton correction
- `itnewton` Number of newton iteration (in corrector)
- `step` current continuation step
- `ds` step size
- `theta` theta parameter for constraint equation in PALC
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
@with_kw_noshow mutable struct ContState{Tv, T, Teigvals, Teigvec, Tcb} <: AbstractContinuationState
	z_pred::Tv								# predictor solution
	tau::Tv									# tangent predictor
	z_old::Tv								# current solution

	isconverged::Bool						# Boolean for newton correction
	itnewton::Int64 = 0						# Number of newton iteration (in corrector)
	itlinear::Int64 = 0						# Number of linear iteration (in newton corrector)

	step::Int64 = 0							# current continuation step
	ds::T									# step size
	theta::T								# theta parameter for constraint equation in PALC

	stopcontinuation::Bool = false			# Boolean to stop continuation
	stepsizecontrol::Bool = true			# Perform step size adaptation

	# the following values encode the current, previous number of unstable (resp. imaginary) eigen values
	# it is initialized as -1 when unknown
	n_unstable::Tuple{Int64,Int64}  = (-1, -1)	# (current, previous)
	n_imag::Tuple{Int64,Int64} 		= (-1, -1)	# (current, previous)

	eigvals::Teigvals = nothing				# current eigenvalues
	eigvecs::Teigvec = nothing				# current eigenvectors

	eventValue::Tcb = nothing
end

function Base.copy(state::ContState)
	return ContState(
		z_pred 	= _copy(state.z_pred),
		tau = _copy(state.tau),
		z_old 	= _copy(state.z_old),
		isconverged = state.isconverged,
		itnewton 	= state.itnewton,
		step 		= state.step,
		ds 			= state.ds,
		theta 		= state.theta,
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
		copyto!(dest.tau , src.tau)
		copyto!(dest.z_old , src.z_old)
		src.isconverged = src.isconverged
		src.itnewton 	= src.itnewton
		src.step 		= src.step
		src.ds 			= src.ds
		src.theta 		= src.theta
		src.stopcontinuation = src.stopcontinuation
		src.stepsizecontrol  = src.stepsizecontrol
		src.n_unstable 		 = src.n_unstable
		src.n_imag 			 = src.n_imag
		src.eventValue		 = src.eventValue
		src.eigvals			 = src.eigvals
		src.eigvecs			 = src.eigvecs # can be removed? to save memory?
	return dest
end

# getters
getSolution(state::AbstractContinuationState) 		= state.z_old
getx(state::AbstractContinuationState) = state.z_old.u
@inline getp(state::AbstractContinuationState) = state.z_old.p
@inline getpreviousp(state::AbstractContinuationState) = state.z_pred.p
@inline isStable(state::AbstractContinuationState) = state.n_unstable[1] == 0
@inline stepsizecontrol(state::AbstractContinuationState) = state.stepsizecontrol
####################################################################################################
# condition for halting the continuation procedure (i.e. when returning false)
@inline done(it::ContIterable, state::ContState) =
			(state.step <= it.contParams.maxSteps) &&
			(isInDomain(it, state.z_old.p) || state.step == 0) &&
			(state.stopcontinuation == false)

function getStateSummary(it, state)
	x = getx(state); p = getp(state)
	pt = it.recordFromSolution(x, p)
	stable = computeEigenElements(it) ? isStable(state) : nothing
	return mergefromuser(pt, (param = p, itnewton = state.itnewton, itlinear = state.itlinear, ds = state.ds, theta = state.theta, n_unstable = state.n_unstable[1], n_imag = state.n_imag[1], stable = stable, step = state.step))
end

function updateStability!(state::ContState, n_unstable::Int, n_imag::Int)
	state.n_unstable = (n_unstable, state.n_unstable[1])
	state.n_imag = (n_imag, state.n_imag[1])
end

function save!(br::ContResult, it::AbstractContinuationIterable, state::AbstractContinuationState)
	# update branch field
	push!(br.branch, getStateSummary(it, state))
	# save solution
	if it.contParams.saveSolEveryStep > 0 && (modCounter(state.step, it.contParams.saveSolEveryStep) || ~done(it, state))
		push!(br.sol, (x = copy(getx(state)), p = getp(state), step = state.step))
	end
	# save eigen elements
	if computeEigenElements(it)
		if mod(state.step, it.contParams.saveEigEveryStep) == 0
			push!(br.eig, (eigenvals = state.eigvals, eigenvec = state.eigvecs, step = state.step))
		end
	end
end

function plotBranchCont(contres::ContResult, state::AbstractContinuationState, iter::ContIterable)
	if iter.plot && mod(state.step, getParams(iter).plotEveryStep) == 0
		return plotBranchCont(contres, getSolution(state), getParams(iter), iter.plotSolution)
	end
end

function ContResult(it::AbstractContinuationIterable, state::AbstractContinuationState)
	x0 = getx(state); p0 = getp(state)
	pt = it.recordFromSolution(x0, p0)
	eiginfo = computeEigenElements(it) ? (state.eigvals, state.eigvecs) : nothing
	return _ContResult(pt, getStateSummary(it, state), x0, setParam(it, p0), it.lens, eiginfo, getParams(it), computeEigenElements(it))
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
	p0 = get(it.par, it.lens)
	T = eltype(it)

	verbose && printstyled("#"^53*"\n────────── Pseudo-Arclength Continuation ────────────\n\n", bold = true, color = :red)

	# Get parameters
	@unpack pMin, pMax, maxSteps, newtonOptions, η, ds = it.contParams
	if !(pMin <= p0 <= pMax)
		@error "Initial parameter $p0 must be within bounds [$pMin, $pMax]"
		return nothing
	end

	# Converge initial guess
	verbose && printstyled("─────────────────  INITIAL GUESS ────────────────────", bold = true, color = :magenta)

	# we pass additional kwargs to newton so that it is sent to the newton callback
	u0, _, isconverged, _, _ = newton(it.F, it.J, it.x0, it.par, newtonOptions; normN = it.normC, callback = it.callbackN, iterationC = 0, p = p0)
	@assert isconverged "Newton failed to converge initial guess on the branch."
	verbose && (print("\n--> convergence of initial guess = ");printstyled("OK\n\n", color=:green))
	verbose && println("--> parameter = ", p0, ", initial step")
	verbose && printstyled("\n───────────────── INITIAL TANGENT ───────────────────", bold = true, color = :magenta)
	u_pred, _, isconverged, _, _ = newton(it.F, it.J,
			u0, setParam(it, p0 + ds / η), newtonOptions; normN = it.normC, callback = it.callbackN, iterationC = 0, p = p0 + ds / η)
	@assert isconverged "Newton failed to converge. Required for the computation of the initial tangent."
	verbose && (print("\n--> convergence of initial guess = ");printstyled("OK\n\n", color=:green))
	verbose && println("--> parameter = ", p0 + ds/η, ", initial step (bis)")
	return iterateFromTwoPoints(it, u0, p0, u_pred, p0 + ds / η; _verbosity = _verbosity)
end

# same as the previous function but when two (initial guesses) points  are provided
function iterateFromTwoPoints(it::ContIterable, u0, p0::T, u1, p1::T; _verbosity = it.verbosity) where T
	theta = it.contParams.theta
	ds = it.contParams.ds
	# this is the last (first) point on the branch
	z_old   = BorderedArray(_copy(u0), p0)
	# this is a predictor for the next point on the branch, we could have used z_old as well
	z_pred	= BorderedArray(_copy(u1), p1)
	tau  = _copy(z_pred)

	# compute the tangent using Secant predictor
	getTangent!(tau, z_pred, z_old, it, ds, theta, SecantPred(), _verbosity)

	# compute eigenvalues to get the type. Necessary to give a ContResult
	if computeEigenElements(it)
		eigvals, eigvecs, = computeEigenvalues(it, u0, it.par, it.contParams.nev)
		if ~saveEigenvectors(it)
			eigvecs = nothing
		end
	else
		eigvals, eigvecs = nothing, nothing
	end

	# compute event value and store into state
	cbval = isEventActive(it) ? initialize(it.event, T) : nothing # event result
	state = ContState(z_pred = z_pred, tau = tau, z_old = z_old, isconverged = true, ds = it.contParams.ds, theta = it.contParams.theta, eigvals = eigvals, eigvecs = eigvecs, eventValue = (cbval, cbval))

	# update stability
	if computeEigenElements(it)
		_, n_unstable, n_imag = isStable(getParams(it), eigvals)
		updateStability!(state, n_unstable, n_imag)
	end

	# we update the event function result
	updateEvent!(it, state)
	return state, state
end

function Base.iterate(it::ContIterable, state::ContState; _verbosity = it.verbosity)
	if !done(it, state) return nothing end
	# next line is to overwrite verbosity behaviour, like when locating bifurcations
	verbosity = min(it.verbosity, _verbosity)
	verbose = verbosity > 0
	verbose1 = verbosity > 1

	@unpack step, ds, theta = state

	# Predictor: state.z_pred. The following method only mutates z_pred
	getPredictor!(state, it)
	
	if verbose
		printstyled("──"^35*"\nContinuation Step $step \n", bold= true);
		@printf("Step size = %2.4e\n", ds); print("Parameter ", getLensSymbol(it.lens))
		@printf(" = %2.4e ⟶  %2.4e [guess]\n", state.z_old.p, clampPredp(state.z_pred.p, it))
	end

	# Corrector, ie newton correction. This does not mutate the arguments
	z_newton, fval, state.isconverged, state.itnewton, state.itlinear = corrector(it,
			state.z_old, state.tau, state.z_pred,
			ds, theta,
			it.tangentAlgo, it.linearAlgo;
			normC = it.normC, callback = it.callbackN, iterationC = step, z0 = state.z_old)

	# Successful step
	if state.isconverged
		if verbose
			verbose1 && printstyled("--> Step Converged in $(state.itnewton) Nonlinear Iteration(s)\n", color=:green)
			print("Parameter ", getLensSymbol(it.lens))
			@printf(" = %2.4e ⟶  %2.4e \n", state.z_old.p, z_newton.p)
		end

		# Get tangent, it only mutates tau
		getTangent!(state.tau, z_newton, state.z_old, it,
					ds, theta, it.tangentAlgo, verbosity)

		# record previous parameter (cheap) and update current solution
		state.z_pred.p = state.z_old.p
		copyto!(state.z_old, z_newton)

		# Eigen-elements computation, they are stored in state
		if computeEigenElements(it)
			# this compute eigen-elements, store them in state and update the stab indices in state
			iteigen = computeEigenvalues!(it, state)
			verbose1 && printstyled(color=:green,"--> Computed ", length(state.eigvals), " eigenvalues in ", iteigen, " iterations, #unstable = ", state.n_unstable[1],"\n")
		end
		state.step += 1
	else
		verbose && printstyled("Newton correction failed\n", color=:red)
		verbose && (println("--> Newton Residuals history = ");display(fval))
	end

	# Step size control
	if ~state.stopcontinuation && stepsizecontrol(state)
		# we update the PALC parameters ds and theta, they are in the state variable
		state.ds, state.theta, state.stopcontinuation = stepSizeControl(ds, theta, it.contParams, state.isconverged, state.itnewton, state.tau, it.tangentAlgo, verbosity)
	end

	return state, state
end

function continuation!(it::ContIterable, state::ContState, contRes::ContResult)
	contParams = getParams(it)
	verbose = it.verbosity > 0
	verbose1 = it.verbosity > 1

	next = (state, state)

	while ~isnothing(next)
		# we get the current state
		_, state = next
		########################################################################################
		# the new solution has been successfully computed
		# we perform saving, plotting, computation of eigenvalues...
		# the case state.step = 0 was just done above
		if state.isconverged && (state.step <= it.contParams.maxSteps) && (state.step > 0)
			# Detection of fold points based on parameter monotony, mutates contRes.specialpoint
			# if we detect bifurcations based on eigenvalues, we disable fold detection to avoid duplicates
			if contParams.detectFold && contParams.detectBifurcation < 2
				foldetected = locateFold!(contRes, it, state)
			end

			if contParams.detectBifurcation > 1 && detectBifucation(state)
				status::Symbol = :guess
				_T = eltype(it)
				interval::Tuple{_T, _T} = getinterval(state.z_pred.p, getp(state))
				# if the detected bifurcation point involves a parameter values with is on
				# the boundary of the parameter domain, we disable bisection because it would
				# lead to infinite looping. Indeed, clamping messes up the `ds`
				if contParams.detectBifurcation > 2 && ~isOnBoundary(it, getp(state))
					verbose1 && printstyled(color=:red, "--> Bifurcation detected before p = ", getp(state), "\n")
					# locate bifurcations with bisection, mutates state so that it stays very close to the bifurcation point. It also updates the eigenelements at the current state. The call returns :guess or :converged
					status, interval = locateBifurcation!(it, state, it.verbosity > 2)
				end
				# we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
				if detectBifucation(state)
					_, bifpt = getBifurcationType(contParams, state, it.normC, it.recordFromSolution, it.verbosity, status, interval)
					if bifpt.type != :none; push!(contRes.specialpoint, bifpt); end
					# detect loop in the branch
					# contParams.detectLoop && (state.stopcontinuation = detectLoop(contRes, bifpt))
				end
			end

			if isEventActive(it)
				# check if an event occured between the 2 continuation steps
				eveDetected = updateEvent!(it, state)
				verbose1 && printstyled(color = :blue, "--> Event values: ", state.eventValue[2], " --> ", state.eventValue[1],"\n")
				eveDetected && (verbose && printstyled(color=:red, "--> Event detected before p = ", getp(state), "\n"))
				# save the event if detected and / or use bisection to locate it precisely
				if eveDetected
					_T = eltype(it); status = :guess; intervalevent = (_T(0),_T(0))
					if contParams.detectEvent > 1
						# interval = getinterval(state.z_pred.p, getp(state))::Tuple{_T, _T}
						status, intervalevent = locateEvent!(it.event, it, state, it.verbosity > 2)
					end
					success, bifpt = getEventType(it.event, it, state, it.verbosity, status, intervalevent)
					state.stopcontinuation |= success
					if bifpt.type != :none; push!(contRes.specialpoint, bifpt); end
				end
			end

			# Saving Solution to File
			contParams.saveToFile && saveToFile(it, getx(state), getp(state), state.step, contRes)

			# Call user saved finaliseSolution function. If returns false, stop continuation
			# we put a OR to stop continuation if the stop was required before
			state.stopcontinuation |= ~it.finaliseSolution(state.z_old, state.tau, state.step, contRes; state = state, iter = it)

			# Save current state in the branch
			save!(contRes, it, state)

			# Plotting
			plotBranchCont(contRes, state, it)
		end
		########################################################################################
		# body
		next = iterate(it, state)
	end

	it.plot && plotBranchCont(contRes, state.z_old, contParams, it.plotSolution)

	# return current solution in case the corrector did not converge
	return contRes, state.z_old, state.tau
end

function continuation(it::ContIterable)
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The return type of this method, e.g. ContResult
	# is not known at compile time so we
	# need a function barrier to resolve it
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	# we compute the cache for the continuation, i.e. state::ContState
	# In this call, we also compute the initial point on the branch (and its stability) and the initial tangent
	states = iterate(it)
	isnothing(states) && return nothing, nothing, nothing

	# variable to hold the result from continuation, i.e. a branch
	contRes = ContResult(it, states[1])

	# perform the continuation
	return continuation!(it, states[1], contRes)
end

function continuation(Fhandle, Jhandle, x0, par, lens::Lens, contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; bothside = false, kwargs...)
	it = ContIterable(Fhandle, Jhandle, x0, par, lens, contParams, linearAlgo; kwargs...)
	if bothside
		res1 = continuation(it)
		@set! it.contParams.ds = -contParams.ds
		res2 = continuation(it)
		contresult = _merge(res1[1],res2[1])
		return contresult, res1[2], res1[3]

	else
		return continuation(it)
	end
end

####################################################################################################

"""
	continuation(F, J, x0, par, lens::Lens, contParams::ContinuationPar; plot = false, normC = norm, dotPALC = (x,y) -> dot(x,y) / length(x), recordFromSolution = norm, plotSolution = (x, p; kwargs...)->nothing, finaliseSolution = (z, tau, step, contResult; kwargs...) -> true, callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) -> true, linearAlgo = BorderingBLS(), tangentAlgo = SecantPred(), verbosity = 0)

Compute the continuation curve associated to the functional `F` and its jacobian `J`.

# Arguments:
- `F` is a function with input arguments `(x, p)`, where `p` is the set of parameters passed to `F`, and returning a vector `r` which represents the functional. For type stability, the types of `x` and `r` should match. In particular, it is not **inplace**,
- `J` is the jacobian of `F` at `(x, p)`. It can assume three forms.
    1. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, you can use the default arguments of `contParams::ContinuationPar` for `continuation`.
    2. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returning `dr` of the same type as `dx`. In our notation, `dr = J * dx`. In this case, you cannot use the default parameters of `contParams::ContinuationPar` for `continuation` and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`,
    3. Or `J` is a function and `J(x, p)` returns a variable `j` which can assume any type. Then, you must implement a linear solver `ls` as a composite type, subtype of `AbstractLinearSolver` which is called like `ls(j, rhs)` and which returns the solution of the jacobian linear system. See for example `examples/SH2d-fronts-cuda.jl`. This linear solver is passed to `NewtonPar(linsolver = ls)` which itself is passed to `ContinuationPar`. Similarly, you have to implement an eigensolver `eig` as a composite type, subtype of `AbstractEigenSolver`.
- `x0` initial guess,
- `par` initial set of parameters,
- `lens::Lens` specifies which parameter axis among `par` is used for continuation. For example, if `par = (α = 1.0, β = 1)`, we can perform continuation w.r.t. `α` by using `lens = (@lens _.α)`. If you have an array `par = [ 1.0, 2.0]` and want to perform continuation w.r.t. the first variable, you can use `lens = (@lens _[1])`. For more information, we refer to `SetField.jl`.
- `contParams` parameters for continuation. See [`ContinuationPar`](@ref) for more information about the options

# Optional Arguments:
- `plot = false` whether to plot the solution while computing
- `recordFromSolution = (x, p) -> norm(x)` function used record a few indicators about the solution. It could be `norm` or `(x, p) -> x[1]`. This is also useful when saving several huge vectors is not possible for memory reasons (for example on GPU...). This function can return pretty much everything but you should keep it small. For example, you can do `(x, p) -> (x1 = x[1], x2 = x[2], nrm = norm(x))` or simply `(x, p) -> (sum(x), 1)`. This will be stored in `contres.branch` (see below). Finally, the first component is used to plot in the continuation curve.
- `plotSolution = (x, p; kwargs...) -> nothing` function implementing the plot of the solution. For example, you can pass something like `(x, p; kwargs...) -> plot(x; kwargs...)`.
- `finaliseSolution = (z, tau, step, contResult; kwargs...) -> true` Function called at the end of each continuation step. Can be used to alter the continuation procedure (stop it by returning `false`), saving personal data, plotting... The notations are ``z=(x, p)``, `tau` is the tangent at `z` (see below), `step` is the index of the current continuation step and `ContResult` is the current branch. For advanced use, the current `state::ContState` of the continuation is passed in `kwargs`. Note that you can have a better control over the continuation procedure by using an iterator, see [Iterator Interface](@ref).
- `callbackN` callback for newton iterations. See docs for [`newton`](@ref). Can be used to change preconditioners
- `tangentAlgo = SecantPred()` controls the algorithm used to predict the tangents along the curve of solutions or the corrector. Can be `NaturalPred`, `SecantPred` or `BorderedPred`. See below for more information.
- `linearAlgo = BorderingBLS()`. Used to control the way the extended linear system associated to the continuation problem is solved. Can be `MatrixBLS`, `BorderingBLS` or `MatrixFreeBLS`.
- `verbosity::Int = 0` controls the amount of information printed during the continuation process. Must belong to `{0,1,2,3}`. In case `contParams.newtonOptions.verbose = false`, the following is valid (Otherwise the newton iterations are shown). Each case prints more information then the previous one:
    - case 0: print nothing
    - case 1: print basic information about the continuation: used predictor, step size and parameter values
    - case 2: print newton iterations number, stability of solution, detected bifurcations / events
    - case 3: print information during bisection to detect bifurcation / events
- `normC = norm` norm used in the different Newton solves
- `dotPALC = (x, y) -> dot(x, y) / length(x)`, dot product used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see below). This argument can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...)
- `filename` name of a file to save the computed branch during continuation. The identifier .jld2 will be appended to this filename
- `bothside=true` compute the branches on the two sides of `p0`, merge them and return it.

# Outputs:
- `contres::ContResult` composite type which contains the computed branch. See [`ContResult`](@ref) for more information.
- `u::BorderedArray` the last solution computed on the branch

!!! tip "Controlling the argument `linearAlgo`"
    In this simplified interface to `continuation`, the argument `linearAlgo` is internally overwritten to provide a valid argument to the algorithm. If you do not want this to happen, call directly `continuation(F, J, x0, par, lens, contParams, linearAlgo; kwargs...)`.

!!! tip "Continuing the branch in the opposite direction"
    Just change the sign of `ds` in `ContinuationPar`.

# Simplified call:
You can also use the following call for which the jacobian **matrix** (beware of large systems of equations!) is computed internally using Finite Differences

	continuation(Fhandle, x0, par, lens, contParams::ContinuationPar; kwargs...)
"""
function continuation(Fhandle, Jhandle, x0, par, lens::Lens, contParams::ContinuationPar;
					linearAlgo = nothing, kwargs...)
	# Create a bordered linear solver using the newton linear solver provided by the user
	if isnothing(linearAlgo)
		_linearAlgo = BorderingBLS(contParams.newtonOptions.linsolver)
	else
		# no linear solver has been specified
		if isnothing(linearAlgo.solver)
			_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
		else
			_linearAlgo = linearAlgo
		end
	end

	return continuation(Fhandle, Jhandle, x0, par, lens, contParams, _linearAlgo; kwargs...)
end

continuation(Fhandle, x0, par, lens::Lens, contParams::ContinuationPar; kwargs...) = continuation(Fhandle, (x, p) -> finiteDifferences(u -> Fhandle(u, p), x), x0, par, lens, contParams; kwargs...)
