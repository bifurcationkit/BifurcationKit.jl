####################################################################################################
# propagate getter and functions on ContIterable
getIterCont(it::AbstractContinuationIterable) = it
getIterCont(it) = it.itr

getParams(it) = getParams(it.itr)
done(it, s) = done(it.itr, s)
plotBranchCont(contres::ContResult, state::AbstractContinuationState, it) = plotBranchCont(contres, state, getIterCont(it))
save!(br, it, state) = save!(br, it.itr, state)
####TODO TOTO
# faire getter for maxSteps
####################################################################################################
struct HaltingIterable{I, F}
	itr::I
	fun::F
end

function iterate(iter::HaltingIterable)
	next = iterate(iter.itr)
	return dispatch(iter, next)
end

function iterate(iter::HaltingIterable, (instruction, state))
	if instruction == :halt return nothing end
	next = iterate(iter.itr, state)
	return dispatch(iter, next)
end

function dispatch(iter::HaltingIterable, next)
	if next === nothing return nothing end
	return next[1], (iter.fun(iter, next[1]) ? :halt : :continue, next[2])
end

halt(iter::I, fun::F) where {I, F} = HaltingIterable{I, F}(iter, fun)
####################################################################################################
struct TeeIterable{I, F}
	itr::I
	fun::F
end

function iterate(iter::TeeIterable, args...)
	next = iterate(iter.itr, args...)
	if next !== nothing iter.fun(next[1]) end
	return next
end

tee(iter::I, fun::F) where {I, F} = TeeIterable{I, F}(iter, fun)
####################################################################################################
@with_kw_noshow mutable struct ContState2{Tv, T, Teigvals, Teigvec, Tcb} <: AbstractContinuationState
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

	n_unstable::Tuple{Int64,Int64}  = (-1, -1)	# (current, previous)
	n_imag::Tuple{Int64,Int64} 		= (-1, -1)	# (current, previous)

	eigvals::Teigvals = nothing				# current eigenvalues
	eigvecs::Teigvec = nothing				# current eigenvectors

	event::Tcb = nothing
end

function Base.copy(state::ContState2)
	return ContState2(
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
		event		 	 = state.event,
		eigvals			 = state.eigvals
	)
end

function copy2(state::ContState)
	return ContState2(
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
		event		 	 = state.event,
		eigvals			 = state.eigvals
	)
end

# condition for halting the continuation procedure (i.e. when returning false)
@inline done(it::AbstractContinuationIterable, state::ContState2) =
			(state.step <= it.contParams.maxSteps) &&
			((it.contParams.pMin < state.z_old.p < it.contParams.pMax) || state.step == 0) &&
			(state.stopcontinuation == false)
####################################################################################################
function iterate(it::AbstractContinuationIterable, state::ContState2; _verbosity = it.verbosity)
	@warn "Iterate qcq"
	# if !done(it, state) return nothing end
	# next line is to overwrite verbosity behaviour, like when locating bifurcations
	verbosity = min(it.verbosity, _verbosity) > 0
	verbose = verbosity > 0

	@unpack step, ds, theta = state

	# Predictor: state.z_pred. The following method only mutates z_pred
	getPredictor!(state, it)
	verbose && print("#"^35*"\nStart of Continuation Step $step:\nParameter $(getLensParam(it.lens))");
	verbose && @printf(" = %2.4e ⟶  %2.4e [guess]\n", state.z_old.p, state.z_pred.p)
	verbose && @printf("Step size = %2.4e\n", ds)

	# Corrector, ie newton correction. This does not mutate the arguments
	z_newton, fval, state.isconverged, state.itnewton, state.itlinear = corrector(it,
			state.z_old, state.tau, state.z_pred,
			ds, theta,
			it.tangentAlgo, it.linearAlgo;
			normC = it.normC, callback = it.callbackN, iterationC = step, z0 = state.z_old)

	# Successful step
	if state.isconverged
		verbose && printstyled("--> Step Converged in $(state.itnewton) Nonlinear Iterations\n", color=:green)

		# Get tangent, it only mutates tau
		getTangent!(state.tau, z_newton, state.z_old, it,
					ds, theta, it.tangentAlgo, verbosity)

		# record previous parameter (cheap) and update current solution
		state.z_pred.p = state.z_old.p
		copyto!(state.z_old, z_newton)
	else
		verbose && printstyled("Newton correction failed\n", color=:red)
		verbose && (println("--> Newton Residuals history = ");display(fval))
	end

	# Step size control
	if ~state.stopcontinuation && stepsizecontrol(state)
		# we update the PALC paramters ds and theta, they are in the state variable
		state.ds, state.theta, state.stopcontinuation = stepSizeControl(ds, theta, it.contParams, state.isconverged, state.itnewton, state.tau, it.tangentAlgo, verbosity)
	end

	state.step += 1
	return state, state
end
####################################################################################################
# construction of iterator
function continuation2(Fhandle, Jhandle, x0, par, lens::Lens, contParams::ContinuationPar;
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
	return continuation2(Fhandle, Jhandle, x0, par, lens, contParams, _linearAlgo; kwargs...)
end

function continuation2(it)
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The result type of this method
	# is not known at compile time so we
	# need a function barrier to resolve it
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	# we compute the cache for the continuation, i.e. state::ContState
	# In this call, we also compute the initial point on the branch (and its stability) and the initial tangent
	if 1==0
		_state, rest = Base.Iterators.peel(it)
		isnothing(_state) && return nothing, nothing, nothing
		@show _state
		contRes = ContResult(getIterCont(it), _state)
		return continuation2!(rest, copy2(_state), contRes)
	else
		_states = iterate(it)
		_state = copy2(_states[1])
		states = (_state, copy(_state))
		isnothing(states) && return nothing, nothing, nothing

		# variable to hold the result from continuation, i.e. a branch
		contRes = ContResult(getIterCont(it), states[1])

		# perform the continuation
		return continuation2!(it, (_states[2][1], _state), contRes)
	end
end

function continuation2(Fhandle, Jhandle, x0, par, lens::Lens, contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; bothside = false, kwargs...)
	it = ContIterable(Fhandle, Jhandle, x0, par, lens, contParams, linearAlgo; kwargs...)
	# add the stoping criterion
	it = halt(it, (i,s) -> ~done(i,s))
	return continuation2(it)
end

function continuation2!(iterCont, _state, contRes::ContResult)
	maxSteps = getParams(getIterCont(iterCont)).maxSteps
	it2 = Base.Iterators.rest(iterCont, _state)
	# limit steps number
	it2 = Base.Iterators.take(it2, maxSteps+1)
	for state in it2
		plotBranchCont(contRes, state, iterCont)
		state.step <= maxSteps && save!(contRes, iterCont, state)
	end
	return contRes, 1
end
