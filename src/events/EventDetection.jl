# these functions are indicators, an event occurs between their value change
nbSigns(x, ::AbstractContinuousEvent) = mapreduce(x -> x > 0, +, x)
nbSigns(x, ::AbstractDiscreteEvent) = x

function nbSigns(x, event::PairOfEvents)
	xc = x[1:event.eventC.nb]
	xd = x[event.eventC.nb+1:end]
	res = (nbSigns(xc, event.eventC)..., nbSigns(xd, event.eventD)...)
	return res
end

function nbSigns(x, event::SetOfEvents)
	nb = 0
	nC = length(event.eventC)
	nD = length(event.eventD)
	nCb = length(x)
	@inbounds for i in 1:nC
		nb += nbSigns(x[i], event.eventC[i])
	end
	@inbounds for i in nC+1:nCb
		nb += nbSigns(x[i], event.eventC[i-nC])
	end
	return  nb
end
####################################################################################################
# Function to locate precisely an Event using a bisection algorithm. We make sure that, at the end of the algorithm, the state is just after the event (in the s coordinate).
# I put the event in first argument even if it is in `iter` in order to allow for easier dispatch
function locateEvent!(event::AbstractEvent, iter, _state, verbose::Bool = true)
	@assert isnothing(_state.eventValue) == false "Empty event value, this should not be happening. Please open an issue."

	# type of scalars in iter
	_T = eltype(iter)
	if abs(_state.ds) < iter.contParams.dsmin; return :none, (_T(0), _T(0)); end

	# get continuation parameters
	contParams = iter.contParams

	n2, n1 = nbSigns(_state.eventValue[1], event), nbSigns(_state.eventValue[2], event)
	verbose && println("----> Entering [Event], indicator of 2 last events = ", (n2, n1))
	verbose && println("----> [Bisection] initial ds = ", _state.ds)

	# we create a new state copy for stepping through the continuation routine
	after = copy(_state)	# after the bifurcation point
	state = copy(_state)	# current state of the bisection
	before = copy(_state)	# before the bifurcation point

	# we reverse some indicators for `before`. It is OK, it will never be used other than for getp(before)
	before.n_unstable = (before.n_unstable[2], before.n_unstable[1])
	before.n_imag = (before.n_imag[2], before.n_imag[1])
	before.eventValue = (before.eventValue[2], before.eventValue[1])
	before.z_pred.p, before.z_old.p = before.z_old.p, before.z_pred.p

	# the bifurcation point is before the current state
	# so we want to first iterate backward
	# we turn off stepsizecontrol because it would not be a
	# bisection otherwise
	state.ds *= -1
	state.step = 0
	state.stepsizecontrol = false

	next = (state, state)

	# record sequence of event indicators
	nsigns = [n2]

	# interval which contains the event
	interval = getinterval(getp(state), state.z_pred.p)
	# index of active index in the bisection interval, allows to track interval
	indinterval = interval[1] == getp(state) ? 1 : 2

	verbose && println("----> [Bisection] state.ds = ", state.ds)

	# we put this to be able to reference it at the end of this function
	# we don't know its type yet
	eiginfo = nothing

	# we compute the number of changes in event indicator
	n_inversion = 0
	status = :guess

	eventlocated = false

	# for a polynomial tangent predictor, we disable the update of the predictor parameters
	# TODO Find better way to do this
	if iter.tangentAlgo isa PolynomialPred; iter.tangentAlgo.update = false; end

	verbose && printstyled(color=:green, "--> eve (initial) ",
		_state.eventValue[2], " --> ",  _state.eventValue[1], "\n")
	(verbose && ~isnothing(_state.eigvals)) && printstyled(color=:green, "\n--> eigvals = ", _state.eigvals, "\n")

	# emulate a do-while
	while true
		if ~state.isconverged
			@error "----> Newton failed when locating bifurcation using bisection method!"
			break
		 end

		# if PALC stops, break the bisection
		if isnothing(next)
			break
		end

		# perform one continuation step
		(_, state) = next

		# the eigenelements have been computed/stored in state during the call iterate(iter, state)
		updateEvent!(iter, state)
		push!(nsigns, nbSigns(state.eventValue[1], event))
		verbose && printstyled(color=:green, "----> eve (current) ",
			state.eventValue[2], " --> ", state.eventValue[1], "\n")
		(verbose && ~isnothing(state.eigvals)) && printstyled(color=:green, "\n----> eigvals = ", state.eigvals, "\n")


		if nsigns[end] == nsigns[end-1]
			# event still after current state, keep going
			state.ds /= 2
		else
			# we passed the event, reverse continuation
			state.ds /= -2
			n_inversion += 1
			indinterval = (indinterval == 2) ? 1 : 2
		end

		if iseven(n_inversion)
			copyto!(after, state)
		else
			copyto!(before, state)
		end

		# update the interval containing the event
		state.step > 0 && (@set! interval[indinterval] = getp(state))

		# we call the finalizer
		state.stopcontinuation = ~iter.finaliseSolution(state.z_old, state.tau, state.step, nothing; bisection = true, state = state)

		if verbose
			printstyled(color=:blue, "----> ", state.step,
				" - [Bisection] (n1, n_current, n2) = ", (n1, nsigns[end], n2),
				"\n\t\t\tds = ", state.ds, ", p = ", getp(state), ", #reverse = ", n_inversion,
				"\n----> event ∈ ", getinterval(interval...),
				", precision = ", @sprintf("%.3E", interval[2] - interval[1]), "\n")
		end

		eventlocated = (isEventCrossed(event, iter, state) &&
				abs(interval[2] - interval[1]) < contParams.tolParamBisectionEvent)

		# condition for breaking the while loop
		if (isnothing(next) == false &&
				abs(state.ds) >= contParams.dsminBisection &&
				state.step < contParams.maxBisectionSteps &&
				n_inversion < contParams.nInversion &&
				eventlocated == false) == false
			break
		end

		next = iterate(iter, state; _verbosity = 0)
	end

	verbose && printstyled(color=:red, "----> Found at p = ", getp(state), " ∈ $interval, \n\t\t\t  δn = ", abs.(2 .* nsigns[end] .- n1 .- n2), ", from p = ", getp(_state), "\n")

	if iter.tangentAlgo isa PolynomialPred
		iter.tangentAlgo.update = true
	end

	######## update current state ########
	# So far we have (possibly) performed an even number of event crossings.
	# We started at the right of the event point. The current state is thus at the
	# right of the event point if iseven(n_inversion) == true. Otherwise, the event
	# point is deemed undetected.
	# In the case n_inversion = 0, we are still on the right of the event point
	if iseven(n_inversion)
		status = n_inversion >= contParams.nInversion ? :converged : :guess
		copyto!(_state.z_pred, state.z_pred)
		copyto!(_state.z_old,  state.z_old)
		copyto!(_state.tau, state.tau)
		# if there is no inversion, the eventValue will possibly be constant like (0, 0). Hence

		if computeEigenElements(iter.event)
			# save eigen-elements
			_state.eigvals = state.eigvals
			if saveEigenvectors(contParams)
				_state.eigvecs = state.eigvecs
			end
			# to prevent spurious event detection, update the following numbers carefully
			_state.n_unstable = (state.n_unstable[1], before.n_unstable[1])
			_state.n_imag = (state.n_imag[1], before.n_imag[1])
		end

		_state.eventValue = (state.eventValue[1], before.eventValue[1])
		interval = (getp(state), getp(before))
	else
		status = :guessL
		copyto!(_state.z_pred, after.z_pred)
		copyto!(_state.z_old,  after.z_old)
		copyto!(_state.tau, after.tau)

		if computeEigenElements(iter.event)
			# save eigen-elements
			_state.eigvals = after.eigvals
			if contParams.saveEigenvectors
				_state.eigvecs = after.eigvecs
			end
			# to prevent spurious event detection, update the following numbers carefully
			_state.n_unstable = (after.n_unstable[1], state.n_unstable[1])
			_state.n_imag = (after.n_imag[1], state.n_imag[1])
		end

		_state.eventValue = (after.eventValue[1], state.eventValue[1])
		interval = (getp(state), getp(after))
	end
	verbose && println("----> Leaving [Loc-Bif]")
	return status, getinterval(interval...)
end
####################################################################################################
####################################################################################################
# because of the way the results are recorded, with state corresponding to the (continuation) step = 0 saved in br.branch[1], it means that br.eig[k] corresponds to state.step = k-1. Thus, the eigen-elements (and other information) corresponding to the current event point are saved in br.eig[step+1]
EventSpecialPoint(state::ContState, Utype::Symbol, status::Symbol, printsolution, normC, interval) = SpecialPoint(state, Utype, status, printsolution, normC, interval; idx = state.step + 1)

# I put the callback in first argument even if it is in iter in order to allow for dispatch
# function to tell the event type based  on the coordinates of the zero
function getEventType(event::AbstractEvent, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}, ind = :) where T
	# record information about the event point
	userpoint = EventSpecialPoint(state, :user, status, iter.recordFromSolution, iter.normC, interval)
	(verbosity > 0) && printstyled(color=:red, "!! User point at p ≈ $(getp(state)) \n")
	return true, userpoint
end
####################################################################################################
function getEventType(event::AbstractContinuousEvent, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}, ind = :; typeE = "userC") where T
	event_index_C = Int32[]
	if state.eventValue[1] isa Real
		if (state.eventValue[1] * state.eventValue[2] < 0)
			push!(event_index_C,1)
		end
	elseif state.eventValue[1][ind] isa Real
		if state.eventValue[1][ind] * state.eventValue[2][ind] < 0
			push!(event_index_C,1)
		end
	else
		for ii in eachindex(state.eventValue[1][ind])
			if state.eventValue[1][ind][ii] * state.eventValue[2][ind][ii] < 0
				push!(event_index_C,ii)
				typeE = typeE * "-$ii"
			end
		end
	end
	@assert isempty(event_index_C) == false "Strange, no event was found whereas it was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n We have eventValue = $(state.eventValue)"
	if hasCustomLabels(event)
		typeE = labels(event, event_index_C)
	end
	# record information about the event point
	userpoint = EventSpecialPoint(state, Symbol(typeE), status, iter.recordFromSolution, iter.normC, interval)
	(verbosity > 0) && printstyled(color=:red, "!! Continuous user point at p ≈ $(getp(state)) \n")
	return true, userpoint
end
####################################################################################################
function getEventType(event::AbstractDiscreteEvent, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}, ind = :; typeE = "userD") where T
	event_index_D = Int32[]
	if state.eventValue[1] isa Real && (abs(state.eventValue[1] - state.eventValue[2]) > 0)
		push!(event_index_D,1)
	elseif state.eventValue[1][ind] isa Real && (abs(state.eventValue[1][ind] - state.eventValue[2][ind]) > 0)
		push!(event_index_D,1)
	else
		for ii in eachindex(state.eventValue[1][ind])
			if abs(state.eventValue[1][ind][ii] - state.eventValue[2][ind][ii]) > 0
				push!(event_index_D,ii)
				typeE = typeE * "-$ii"
			end
		end
	end
	@assert isempty(event_index_D) == false "Strange, no event was found whereas it was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n We have eventValue = $(state.eventValue)"
	if hasCustomLabels(event)
		typeE = labels(event, event_index_D)
	end
	# record information about the ev point
	userpoint = EventSpecialPoint(state, Symbol(typeE), status, iter.recordFromSolution, iter.normC, interval)
	(verbosity > 0) && printstyled(color=:red, "!! Discrete user point at p ≈ $(getp(state)) \n")
	return true, userpoint
end
####################################################################################################
function getEventType(event::PairOfEvents, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}) where T
	nC = length(event.eventC)
	n = length(event)

	(isEventCrossed(event.eventC, iter, state, 1:nC) && isEventCrossed(event.eventD, iter, state, nC+1:n)) && @warn "More than one Event was detected. We call the continuous event to save data in the branch."
	if isEventCrossed(event.eventC, iter, state, 1:nC)
		return getEventType(event.eventC, iter, state, verbosity, status, interval, 1:nC; typeE = "userC")
	elseif isEventCrossed(event.eventD, iter, state, nC+1:n)
		return getEventType(event.eventD, iter, state, verbosity, status, interval, nC+1:n; typeE = "userD")
	else
		throw("Error, no event was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. Indeed, this should not happen.")
	end
end

####################################################################################################
function getEventType(event::SetOfEvents, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}) where T
	# find the active events
	event_index_C = Int32[]
	event_index_D = Int32[]
	for (ind, eve) in enumerate(event.eventC)
		if isEventCrossed(eve, iter, state, ind)
			push!(event_index_C, ind)
		end
	end

	nC = length(event.eventC)
	for (ind, eve) in enumerate(event.eventD)
		if isEventCrossed(eve, iter, state, nC + ind)
			push!(event_index_D, ind)
		end
	end

	length(event_index_C) + length(event_index_D) > 1 && @warn "More than one event in `SetOfEvents` was detected. We take the first in the list to save data in the branch."

	if isempty(event_index_C) == false
		indC = event_index_C[1]
		return getEventType(event.eventC[indC], iter, state, verbosity, status, interval, indC; typeE = "userC$indC")
	elseif isempty(event_index_D) == false
		indD = event_index_D[1]
		return getEventType(event.eventD[indD], iter, state, verbosity, status, interval, indD+nC; typeE = "userD$indD")
	else
		@assert 1==0 "Error, no event was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. Indeed, this should not happen."
	end
end
