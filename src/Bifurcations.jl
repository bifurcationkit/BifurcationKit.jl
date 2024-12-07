####################################################################################################
"""
This function checks whether the solution with eigenvalues `eigvalues` is stable. It also computes the number of unstable eigenvalues with nonzero imaginary part
"""
function is_stable(contparams::ContinuationPar, eigvalues)::NamedTuple{(:isstable, :n_unstable, :n_imag), Tuple{Bool, Int64, Int64}}
    # the return type definition above is to remove type instability in continuation
    # numerical precision for deciding if an eigenvalue is above a threshold
    precision = contparams.tol_stability

    # update number of unstable eigenvalues
    n_unstable = mapreduce(x -> real(x) > precision, +, eigvalues)

    # update number of unstable eigenvalues with nonzero imaginary part
    n_imag = mapreduce(x -> (abs(imag(x)) > precision) * (real(x) > precision), +, eigvalues)

    isstable = n_unstable == 0
    return (;isstable, n_unstable, n_imag)
end
is_stable(::ContinuationPar, ::Nothing) = (isstable = true , n_unstable = 0, n_imag = 0)

# we detect a bifurcation by a change in the number of unstable eigenvalues
function detect_bifurcation(state::ContState)
    n1, n2 = state.n_unstable
    # deals with missing value encoded by n_unstable = -1
    if n1 == -1 || n2 == -1; return false; end
    # detect a bifurcation if the numbers do not match
    return n1 !== n2
end
####################################################################################################

# Test function for Fold bifurcation
@inline detect_fold(p1, p2, p3) = (p3 - p2) * (p2 - p1) < 0

# if fold point is located, store it in contres
function locate_fold!(contres::ContResult, iter::ContIterable, state::ContState)
    branch = contres.branch
    n_br = length(branch)
    lazy_params = LazyRows(branch)

    # Fold point detection based on continuation parameter monotony
    if iter.contparams.detect_fold && length(branch) > 2 && detect_fold(lazy_params[n_br-2].param, lazy_params[n_br-1].param, lazy_params[n_br].param)
        if iter.verbosity > 0
            printstyled(color=:red, "──> Fold bifurcation point in ", getinterval(branch[end-1].param, branch[end].param), "\n")
        end
        npar = length(branch[1]) - 8
        push!(contres.specialpoint, SpecialPoint(
            type = :fold,
            idx = length(branch) - 1,
            param = getp(state),
            norm = iter.normC(getx(state)),
            printsol = NamedTuple{keys(branch[end-1])[begin:npar]}(values(branch[end-1])[begin:npar]),
            x = save_solution(iter.prob, _copy(getx(state)), setparam(iter.prob, getp(state))),
            τ = copy(state.τ),
            ind_ev = 0,
            # it means the fold occurs between step - 2 and step:
            step = length(branch) - 1,
            status = :guess,
            δ = (0, 0),
            precision = -1.,
            interval = (lazy_params[n_br-1].param, lazy_params[n_br-1].param)))
        return true
    else
        return false
    end
end
####################################################################################################
"""
Function for coarse detection of bifurcation points.
"""
function get_bifurcation_type(it::ContIterable, state, status::Symbol, interval::Tuple{T, T}, eig::AbstractEigenSolver) where T
    # this boolean ensures that edge cases are handled
    known::Bool = false

    # get current/previous number of unstable eigenvalues and
    n_unstable, n_unstable_prev = state.n_unstable
    # unstable eigenvalues with nonzero imaginary part
    n_imag, n_imag_prev = state.n_imag

    # computation of the index of the bifurcating eigenvalue
    ind_ev = n_unstable < n_unstable_prev ? n_unstable_prev : n_unstable
    # bifurcation type
    tp = :none

    δn_unstable = abs(n_unstable - n_unstable_prev)
    δn_imag     = abs(n_imag - n_imag_prev)

    # codim 1 bifurcation point detection based on eigenvalues distribution
    if δn_unstable == 1
    # In this case, only a single eigenvalue crossed the imaginary axis
    # Either it is a Branch Point
        if δn_imag == 0
            tp = :bp
        elseif δn_imag == 1
    # Either it is a Hopf bifurcation for a Complex valued system
            tp = eig isa AbstractFloquetSolver ? :pd : :hopf
        else # I dont know what bifurcation this is
            tp = :nd
        end
        known = true
    elseif δn_unstable == 2
        if δn_imag == 2
            tp = eig isa AbstractFloquetSolver ? :ns : :hopf
        else
            tp = :nd
        end
        known = true
    elseif δn_unstable > 2
        tp = :nd
        known = true
    end

    if δn_unstable < δn_imag
        @warn "Error in eigenvalues computation. It seems that an eigenvalue is missing, probably conj(λ) for some already computed eigenvalue λ. This makes the identification (but not the detection) of bifurcation points erroneous. You should increase the number of requested eigenvalues `nev`."
        tp = :nd
        known = true
    end

    # rule out initial condition where we populate n_unstable = (-1,-1) and n_imag = (-1,-1)
    if prod(state.n_unstable) < 0 || prod(state.n_imag) < 0
        tp = :nd
        known = true
    end

    if known
        # record information about the bifurcation point
        # because of the way the results are recorded, with state corresponding to the (continuation) step = 0 saved in br.branch[1], it means that br.eig[k] corresponds to state.step = k-1. Thus, the eigen-elements (and other information) corresponding to the current bifurcation point are saved in br.eig[step+1]
        specialpoint = SpecialPoint(it, state, tp, status, interval;
            δ = (n_unstable - n_unstable_prev, n_imag - n_imag_prev),
            idx = state.step + 1,
            ind_ev = ind_ev)
        (it.verbosity>0) && printstyled(color=:red, "──> ", tp, " Bifurcation point at p ≈ ", getp(state), ", δn_unstable = ", δn_unstable,",  δn_imag = ", δn_imag, "\n")
    else
        throw("We could not detect/identify the bifurcation point. (δn_unstable, δn_imag) = ($δn_unstable, $δn_imag)")
    end
    return known, specialpoint
end

# this allows for dispatch on eigensolver type
get_bifurcation_type(it::ContIterable, state, status::Symbol, interval::Tuple{T, T}) where T = get_bifurcation_type(it, state, status, interval, it.contparams.newton_options.eigsolver)

"""
Function to locate precisely bifurcation points using a bisection algorithm. We make sure that at the end of the algorithm, the state is just after the bifurcation point (in the s coordinate).
"""
function locate_bifurcation!(iter::ContIterable, _state::ContState, verbose::Bool = true)
    ~detect_bifurcation(_state) && error("No bifurcation detected for the state")
    verbose && println("┌─── Entering [Locate-Bifurcation], state.n_unstable = ", _state.n_unstable)

    # type of scalars in iter
    _T = eltype(iter)

    # number of unstable eigenvalues after, before the bifurcation point
    n2, n1 = _state.n_unstable
    if n1 == -1 || n2 == -1 return :none, (_T(0), _T(0)) end

    contParams = iter.contparams
    
    if abs(_state.ds) < contParams.dsmin; return :none, (_T(0), _T(0)); end
    verbose && println("├─── [Bisection] initial ds = ", _state.ds)

    # we create a new state for stepping through the continuation routine
    after = copy(_state)  # after the bifurcation point
    state = copy(_state)  # current state of the bisection
    before = copy(_state) # before the bifurcation point
    state.in_bisection = true # we signal to the state the we are entering bisection

    # we reverse some indicators for `before`. It is OK, it will never be used other than for getp(before)
    before.n_unstable = (before.n_unstable[2], before.n_unstable[1])
    before.n_imag = (before.n_imag[2], before.n_imag[1])
    before.z_old.p, before.z.p = before.z.p, before.z_old.p

    # the bifurcation point is before the current state so we want to first iterate backward with
    # half step size. We turn off stepsizecontrol because it would not make a bisection otherwise
    state.ds *= -1
    state.step = 0
    state.stepsizecontrol = false

    # this variable is used for the iterator
    next = (state, state)

    # record sequence of unstable eigenvalue numbers and parameters
    nunstbls = [n2]
    nimags   = [state.n_imag[1]]

    # interval which contains the bifurcation point
    interval = getinterval(getp(state), getpreviousp(state))

    # index of active index in the bisection interval, allows to track interval
    indinterval = interval[1] == getp(state) ? 1 : 2

    verbose && println("├─── [Bisection] state.ds = ", state.ds)

    # we put this to be able to reference it at the end of this function
    # we don't know its type yet
    eiginfo = nothing

    # we compute the number of changes in n_unstable
    n_inversion = 0
    status = :guess

    biflocated = false

    # for a polynomial tangent predictor, we disable the update of the predictor parameters
    internal_adaptation!(iter.alg, false)

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
        push!(nunstbls, state.n_unstable[1])
        push!(nimags, state.n_imag[1])

        if nunstbls[end] == nunstbls[end-1]
            # bifurcation point still after current state, keep going
            state.ds /= 2
        else
            # we passed the bifurcation point, reverse continuation
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

        state.step > 0 && (interval = @set interval[indinterval] = getp(state))

        # we call the finalizer
        # iter.finalise_solution(state.z_old, state.tau, state.step, nothing; )

        if verbose
            ct0 = rightmost(state.eigvals)
            printstyled(color=:blue,
                "├─── $(state.step) - [Bisection] (n1, n_current, n2) = ", (n1, nunstbls[end], n2),
                ", ds = ", state.ds, " p = ", getp(state), ", #reverse = ", n_inversion,
                "\n├─── bifurcation ∈ ", getinterval(interval...),
                ", precision = ", @sprintf("%.3E", interval[2] - interval[1]),
                "\n├─── ", length(ct0)," Eigenvalues closest to ℜ = 0:\n")
            verbose && Base.display(sort(ct0[begin:min(5, length(ct0))], by = real))
        end

        biflocated = abs(real.(rightmost(state.eigvals))[1]) < contParams.tol_bisection_eigenvalue

        if (isnothing(next) == false &&
                abs(state.ds) >= contParams.dsmin_bisection &&
                state.step < contParams.max_bisection_steps &&
                n_inversion < contParams.n_inversion &&
                biflocated == false) == false
            break
        end

        next = iterate(iter, state; _verbosity = 0)
    end

    verbose && printstyled(color=:red, "────> Found at p = ", getp(state), ", δn = ", abs(2nunstbls[end]-n1-n2),", δim = ",abs(2nimags[end]-sum(state.n_imag))," from p = ",getp(_state),"\n")

    if verbose
        printstyled(color=:red, "────> Found at p = ", getp(state), " ∈ $interval, \n\t\t\t  δn = ", abs(2nunstbls[end]-n1-n2), ", δim = ",abs(2nimags[end]-sum(state.n_imag))," from p = ",getp(_state),"\n")
        printstyled(color=:blue, "─"^40*
                "\n┌─── Stopping reason:\n├───── isnothing(next)             = ", isnothing(next),
                "\n├───── |ds| < dsmin_bisection      = ", abs(state.ds) < contParams.dsmin_bisection,
                "\n├───── step >= max_bisection_steps = ", state.step >= contParams.max_bisection_steps,
                "\n├───── n_inversion >= n_inversion  = ", n_inversion >= contParams.n_inversion,
                "\n└───── biflocated                  = ", biflocated == true, "\n")

    end
    internal_adaptation!(iter.alg, true)

    ######## update current state ########
    # So far we have (possibly) performed an even number of bifurcation crossings
    # we started at the right of the bifurcation point. The current state is thus at the
    # right of the bifurcation point if iseven(n_inversion) == true. Otherwise, the bifurcation
    # point is still deemed undetected
    if iseven(n_inversion)
        status = n_inversion >= contParams.n_inversion ? :converged : :guess
        copyto!(_state.z_old, state.z_old)
        copyto!(_state.z_pred, state.z_pred)
        copyto!(_state.z, state.z)
        copyto!(_state.τ, state.τ)

        _state.eigvals = state.eigvals
        if save_eigenvectors(contParams)
            _state.eigvecs = state.eigvecs
        end

        # to prevent spurious bifurcation detection, update the following numbers carefully
        # since the current state is after the bifurcation point, we just save
        # the current state of n_unstable and n_imag
        _state.n_unstable = (state.n_unstable[1], before.n_unstable[1])
        _state.n_imag = (state.n_imag[1], before.n_imag[1])

        # previous_p = n_inversion == 0 ? before.z_old.p : getp(before)
        interval = (getp(state), getp(before))
    else
        status = :guessL
        copyto!(_state.z_old, after.z_old)
        copyto!(_state.z_pred, after.z_pred)
        copyto!(_state.z,  after.z)
        copyto!(_state.τ, after.τ)

        _state.eigvals = after.eigvals
        if contParams.save_eigenvectors
            _state.eigvecs = after.eigvecs
        end

        # to prevent spurious bifurcation detection, update the following numbers carefully
        # since the current state is before the bifurcation point, we just save
        # the current state of n_unstable and n_imag
        _state.n_unstable = (after.n_unstable[1], state.n_unstable[1])
        _state.n_imag = (after.n_imag[1], state.n_imag[1])
        interval = (getp(state), getp(after))
    end
    # update the predictor before leaving
    update_predictor!(_state, iter)
    verbose && println("────> Leaving [Loc-Bif]")
    return status, getinterval(interval...)
end
