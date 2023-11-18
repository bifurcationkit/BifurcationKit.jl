# these functions are indicators, an event occurs when they change value
nb_signs(x, ::AbstractContinuousEvent) = mapreduce(x -> x > 0, +, x)
nb_signs(x, ::AbstractDiscreteEvent) = x

function nb_signs(x, event::PairOfEvents)
    xc = x[1:event.eventC.nb]
    xd = x[event.eventC.nb+1:end]
    res = (nb_signs(xc, event.eventC)..., nb_signs(xd, event.eventD)...)
    return res
end

function nb_signs(x, event::SetOfEvents)
    nb = 0
    nC = length(event.eventC)
    nD = length(event.eventD)
    nCb = length(x)
    @inbounds for i in 1:nC
        nb += nb_signs(x[i], event.eventC[i])
    end
    @inbounds for i in nC+1:nCb
        nb += nb_signs(x[i], event.eventC[i-nC])
    end
    return nb
end
####################################################################################################
# Function to locate precisely an Event using a bisection algorithm. We make sure that, at the end of the algorithm, the state is just after the event (in the s coordinate).
# I put the event in first argument even if it is in `iter` in order to allow for easier dispatch
function locate_event!(event::AbstractEvent, iter, _state, verbose::Bool = true)
    @assert isnothing(_state.eventValue) == false "Empty event value, this should not be happening. Please open an issue."

    # type of scalars in iter
    _T = eltype(iter)

    # we test if the current state is an event, ie satisfies the constraint
    # up to a given tolerance. Very important to detect BT
    if isonevent(event, _state.eventValue[1])
        return :converged, getinterval(getp(_state), getpreviousp(_state))
    end

    if abs(_state.ds) < iter.contparams.dsmin; return :none, (_T(0), _T(0)); end

    # get continuation parameters
    contParams = iter.contparams

    n2, n1 = nb_signs(_state.eventValue[1], event), nb_signs(_state.eventValue[2], event)
    verbose && println("────> Entering [Event], indicator of 2 last events = ", (n2, n1))
    verbose && println("────> [Bisection] initial ds = ", _state.ds)

    # we create a new state copy for stepping through the continuation routine
    after = copy(_state)  # after the bifurcation point
    state = copy(_state)  # current state of the bisection
    before = copy(_state) # before the bifurcation point

    # we reverse some indicators for `before`. It is OK, it will never be used other than for getp(before)
    before.n_unstable = (before.n_unstable[2], before.n_unstable[1])
    before.n_imag = (before.n_imag[2], before.n_imag[1])
    before.eventValue = (before.eventValue[2], before.eventValue[1])
    before.z_old.p, before.z.p = before.z.p, before.z_old.p

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
    interval = getinterval(getp(state), getpreviousp(state))
    # index of active index in the bisection interval, allows to track interval
    indinterval = interval[1] == getp(state) ? 1 : 2

    verbose && println("────> [Bisection] state.ds = ", state.ds)

    # we put this to be able to reference it at the end of this function
    # we don't know its type yet
    eiginfo = nothing

    # we compute the number of changes in event indicator
    n_inversion = 0
    status = :guess

    eventlocated::Bool = false

    # for a polynomial tangent predictor, we disable the update of the predictor parameters
    internal_adaptation!(iter.alg, false)

    verbose && printstyled(color=:green, "──> eve (initial) ",
        _state.eventValue[2], " ──> ",  _state.eventValue[1], "\n")
    if verbose && ~isnothing(_state.eigvals)
        printstyled(color=:green, "\n──> eigvals = \n")
        print_ev(_state.eigvals, :green)
        # calcul des VP et determinant
    end

    # emulate a do-while
    while true
        if ~converged(state)
            @error "Newton failed to fully locate bifurcation point using bisection parameters!"
            break
         end

        # if PALC stops, break the bisection
        if isnothing(next)
            break
        end

        # perform one continuation step
        (_, state) = next

        # the eigenelements have been computed/stored in state during the call iterate(iter, state)
        update_event!(iter, state)
        push!(nsigns, nb_signs(state.eventValue[1], event))
        verbose && printstyled(color=:green, "\n────> eve (current) ",
            state.eventValue[2], " ──> ", state.eventValue[1], "\n")

        if verbose && ~isnothing(state.eigvals)
            printstyled(color=:blue, "────> eigvals = \n")
            print_ev(state.eigvals, :blue)
        end

        if nsigns[end] == nsigns[end-1]
            # event still after current state, keep going
            state.ds /= 2
        else
            # we passed the event, reverse continuation
            state.ds /= -2
            n_inversion += 1
            indinterval = (indinterval == 2) ? 1 : 2
        end
        update_predictor!(state, iter)

        if iseven(n_inversion)
            copyto!(after, state)
        else
            copyto!(before, state)
        end

        # update the interval containing the event
        state.step > 0 && (@set! interval[indinterval] = getp(state))

        # we call the finalizer
        state.stopcontinuation = ~iter.finalise_solution(state.z, state.τ, state.step, nothing; bisection = true, state = state)

        if verbose
            printstyled(color=:blue, bold = true, "────> ", state.step,
                " - [Bisection] (n1, n_current, n2) = ", (n1, nsigns[end], n2),
                "\n\t\t\tds = ", state.ds, ", p = ", getp(state), ", #reverse = ", n_inversion,
                "\n────> event ∈ ", getinterval(interval...),
                ", precision = ", @sprintf("%.3E", interval[2] - interval[1]), "\n")
        end

        # this test contains the verification that the current state is an
        # event up to a given tolerance. Very important to detect BT
        eventlocated = (is_event_crossed(event, iter, state) &&
                abs(interval[2] - interval[1]) < contParams.tol_param_bisection_event)

        # condition for breaking the while loop
        if (isnothing(next) == false &&
                abs(state.ds) >= contParams.dsmin_bisection &&
                state.step < contParams.max_bisection_steps &&
                n_inversion < contParams.n_inversion &&
                eventlocated == false) == false
            break
        end

        next = iterate(iter, state; _verbosity = 0)
    end

    if verbose
        printstyled(color=:red, "────> Found at p = ", getp(state), " ∈ $interval, \n\t\t\t  δn = ", abs.(2 .* nsigns[end] .- n1 .- n2), ", from p = ", getp(_state), "\n")
        printstyled(color=:blue, "─"^40*"\n────> Stopping reason:\n──────> isnothing(next)           = ", isnothing(next),
                "\n──────> |ds| < dsmin_bisection     = ", abs(state.ds) < contParams.dsmin_bisection,
                "\n──────> step >= max_bisection_steps = ", state.step >= contParams.max_bisection_steps,
                "\n──────> n_inversion >= n_inversion = ", n_inversion >= contParams.n_inversion,
                "\n──────> eventlocated              = ", eventlocated == true, "\n")

    end

    internal_adaptation!(iter.alg, true)

    ######## update current state ########
    # So far we have (possibly) performed an even number of event crossings.
    # We started at the right of the event point. The current state is thus at the
    # right of the event point if iseven(n_inversion) == true. Otherwise, the event
    # point is deemed undetected.
    # In the case n_inversion = 0, we are still on the right of the event point
    if iseven(n_inversion)
        status = n_inversion >= contParams.n_inversion ? :converged : :guess
        copyto!(_state.z_pred, state.z_pred)
        copyto!(_state.z_old,  state.z_old)
        copyto!(_state.z,  state.z)
        copyto!(_state.τ, state.τ)
        # if there is no inversion, the eventValue will possibly be constant like (0, 0). Hence

        if compute_eigenelements(iter.event)
            # save eigen-elements
            _state.eigvals = state.eigvals
            if save_eigenvectors(contParams)
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
        copyto!(_state.z_old, after.z_old)
        copyto!(_state.z,  after.z)
        copyto!(_state.τ, after.τ)

        if compute_eigenelements(iter.event)
            # save eigen-elements
            _state.eigvals = after.eigvals
            if contParams.save_eigenvectors
                _state.eigvecs = after.eigvecs
            end
            # to prevent spurious event detection, update the following numbers carefully
            _state.n_unstable = (after.n_unstable[1], state.n_unstable[1])
            _state.n_imag = (after.n_imag[1], state.n_imag[1])
        end

        _state.eventValue = (after.eventValue[1], state.eventValue[1])
        interval = (getp(state), getp(after))
    end
    # update the predictor before leaving
    update_predictor!(_state, iter)
    verbose && println("────> Leaving [Loc-Bif]")
    return status, getinterval(interval...)
end
####################################################################################################
####################################################################################################
# because of the way the results are recorded, with state corresponding to the (continuation) step = 0 saved in br.branch[1], it means that br.eig[k] corresponds to state.step = k-1. Thus, the eigen-elements (and other information) corresponding to the current event point are saved in br.eig[step+1]
EventSpecialPoint(it::ContIterable, state::ContState, Utype::Symbol, status::Symbol, interval) = SpecialPoint(it, state, Utype, status, interval; idx = state.step + 1)

# I put the callback in first argument even if it is in iter in order to allow for dispatch
# function to tell the event type based  on the coordinates of the zero
function get_event_type(event::AbstractEvent, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}, ind = :) where T
    # record information about the event point
    userpoint = EventSpecialPoint(state, :user, status, record_from_solution(iter), iter.normC, interval)
    (verbosity > 0) && printstyled(color=:red, "!! User point at p ≈ $(getp(state)) \n")
    return true, userpoint
end
####################################################################################################
function get_event_type(event::AbstractContinuousEvent, 
                        iter::AbstractContinuationIterable, 
                        state, 
                        verbosity, 
                        status::Symbol, 
                        interval::Tuple{T, T}, 
                        ind = :; 
                        typeE = "userC") where T
    event_index_C = Int32[]
    if state.eventValue[1] isa Real
        if test_event(event, state.eventValue[1], state.eventValue[2])
            push!(event_index_C, 1)
        end
    elseif state.eventValue[1][ind] isa Real
        if test_event(event, state.eventValue[1][ind], state.eventValue[2][ind])
            push!(event_index_C, 1)
        end
    else
        for ii in eachindex(state.eventValue[1][ind])
            if test_event(event, state.eventValue[1][ind][ii], state.eventValue[2][ind][ii])
                push!(event_index_C, ii)
                typeE = typeE * "-$ii"
            end
        end
    end
    if isempty(event_index_C) == true
        @error "Error, no event was characterized whereas one was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n The events are eventValue = $(state.eventValue)"
        # we halt continuation as it will mess up the detection of events
        state.stopcontinuation = true
        return false, EventSpecialPoint(state, Symbol(typeE), status, record_from_solution(iter), iter.normC, interval)
    end

    if has_custom_labels(event)
        typeE = labels(event, event_index_C)
    end
    # record information about the event point
    userpoint = EventSpecialPoint(iter, state, Symbol(typeE), status, interval)
    (verbosity > 0) && printstyled(color=:red, "!! Continuous user point at p ≈ $(getp(state)) \n")
    return true, userpoint
end
####################################################################################################
function get_event_type(event::AbstractDiscreteEvent, 
                        iter::AbstractContinuationIterable, 
                        state, 
                        verbosity, 
                        status::Symbol, 
                        interval::Tuple{T, T}, 
                        ind = :; 
                        typeE = "userD") where T
    event_index_D = Int32[]
    if state.eventValue[1] isa Real && (abs(state.eventValue[1] - state.eventValue[2]) > 0)
        push!(event_index_D, 1)
    elseif state.eventValue[1][ind] isa Real && (abs(state.eventValue[1][ind] - state.eventValue[2][ind]) > 0)
        push!(event_index_D, 1)
    else
        for ii in eachindex(state.eventValue[1][ind])
            if abs(state.eventValue[1][ind][ii] - state.eventValue[2][ind][ii]) > 0
                push!(event_index_D, ii)
                typeE = typeE * "-$ii"
            end
        end
    end
    if isempty(event_index_D) == true
        @error "Error, no event was characterized whereas one was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n The events are eventValue = $(state.eventValue)"
        # we halt continuation as it will mess up the detection of events
        state.stopcontinuation = true
        return false, EventSpecialPoint(state, Symbol(typeE), status, record_from_solution(iter), iter.normC, interval)
    end
    if has_custom_labels(event)
        typeE = labels(event, event_index_D)
    end
    # record information about the ev point
    userpoint = EventSpecialPoint(iter, state, Symbol(typeE), status, interval)
    (verbosity > 0) && printstyled(color=:red, "!! Discrete user point at p ≈ $(getp(state)) \n")
    return true, userpoint
end
####################################################################################################
function get_event_type(event::PairOfEvents, 
                        iter::AbstractContinuationIterable, 
                        state, 
                        verbosity, 
                        status::Symbol, 
                        interval::Tuple{T, T}) where T
    nC = length(event.eventC)
    n = length(event)

    if (is_event_crossed(event.eventC, iter, state, 1:nC) && is_event_crossed(event.eventD, iter, state, nC+1:n))
        evc = get_event_type(event.eventC, iter, state, verbosity, status, interval, 1:nC; typeE = "userC")
        evd = get_event_type(event.eventD, iter, state, verbosity, status, interval, nC+1:n; typeE = "userD")
        @warn "More than one Event was detected $(evc[2].type)-$(evd[2].type). We call the continuous event to save data in the branch."
        @set! evc[2].type = Symbol(evc[2].type, evd[2].type)
        return evc
    end
    if is_event_crossed(event.eventC, iter, state, 1:nC)
        return get_event_type(event.eventC, iter, state, verbosity, status, interval, 1:nC; typeE = "userC")
    elseif is_event_crossed(event.eventD, iter, state, nC+1:n)
        return get_event_type(event.eventD, iter, state, verbosity, status, interval, nC+1:n; typeE = "userD")
    else
        @error "Error, no event was characterized whereas one was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n The events are eventValue = $(state.eventValue)"
        # we halt continuation as it will mess up the detection of events
        state.stopcontinuation = true
        return false, EventSpecialPoint(state, :PairOfEvents, status, record_from_solution(iter), iter.normC, interval)
    end
end

####################################################################################################
function get_event_type(event::SetOfEvents, 
                        iter::AbstractContinuationIterable, 
                        state, 
                        verbosity, 
                        status::Symbol, 
                        interval::Tuple{T, T}) where T
    # find the active events
    event_index_C = Int32[]
    event_index_D = Int32[]
    for (ind, eve) in enumerate(event.eventC)
        if is_event_crossed(eve, iter, state, ind)
            push!(event_index_C, ind)
        end
    end

    nC = length(event.eventC)
    for (ind, eve) in enumerate(event.eventD)
        if is_event_crossed(eve, iter, state, nC + ind)
            push!(event_index_D, ind)
        end
    end

    length(event_index_C) + length(event_index_D) > 1 && @warn "More than one event in `SetOfEvents` was detected. We take the first in the list to save data in the branch."

    if isempty(event_index_C) == false
        indC = event_index_C[1]
        return get_event_type(event.eventC[indC], iter, state, verbosity, status, interval, indC; typeE = "userC$indC")
    elseif isempty(event_index_D) == false
        indD = event_index_D[1]
        return get_event_type(event.eventD[indD], iter, state, verbosity, status, interval, indD+nC; typeE = "userD$indD")
    else
        @error "Error, no event was characterized whereas one was detected. Please open an issue at https://github.com/rveltz/BifurcationKit.jl/issues. \n The events are eventValue = $(state.eventValue)"
        # we halt continuation as it will mess up the detection of events
        state.stopcontinuation = true
        return false, EventSpecialPoint(state, :SetOfEvents, status, record_from_solution(iter), iter.normC, interval)
    end
end
