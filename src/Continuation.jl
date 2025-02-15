import Base: iterate
####################################################################################################
# Iterator interface
"""
$(TYPEDEF)

Define a continuation iterator. This allows for example to do:

```
iter = ContIterable(prob, alg, opts; kwargs...)
for state in iter
    println("Continuation step = ", state.step)
end
```

More information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/iterator/)

# Useful functions
- `setparam(iter, p::Real)` set parameter with lens `iter.prob.lens` to `p`
- `is_event_active(iter)` whether the event detection is active
- `compute_eigenelements(iter)` whether to compute eigen elements
- `save_eigenvectors(iter)` whether to save eigen vectors
- `getparams(iter)` get full list of continuation parameters
- `isindomain(iter, p)` whether `p` in is domain [p_min, p_max]. (See [`ContinuationPar`](@ref))
- `is_on_boundary(iter, p)` whether `p` in is {p_min, p_max}
"""
Base.@kwdef struct ContIterable{Tkind <: AbstractContinuationKind, Tprob, Talg, T, S, E, TnormC, Tfinalisesolution, TcallbackN, Tevent} <: AbstractContinuationIterable{Tkind}
    kind::Tkind
    prob::Tprob
    alg::Talg
    contparams::ContinuationPar{T, S, E}
    plot::Bool = false
    event::Tevent = nothing
    normC::TnormC
    finalise_solution::Tfinalisesolution
    callback_newton::TcallbackN
    verbosity::UInt8 = 2
    filename::String
end

# constructor
function ContIterable(prob::AbstractBifurcationProblem,
                    alg::AbstractContinuationAlgorithm,
                    contparams::ContinuationPar{T, S, E};
                    kind = EquilibriumCont(),
                    filename = "branch-" * string(Dates.now()),
                    plot = false,
                    normC = norm,
                    finalise_solution = finalise_default,
                    callback_newton = cb_default,
                    event = nothing,
                    verbosity = 0, 
                    kwargs...) where {T <: Real, S, E}
    if ~isempty(kwargs)
        @warn "Unrecognized keyword arguments found. Future versions will error."
        print("Unrecognized keyword arguments: ")
        unrecognized = keys(kwargs)
        printstyled(unrecognized; bold = true, color = :red)
        print("\n\n")
        throw("Stopping continuation.")
    end

    return ContIterable(;kind,
                prob,
                alg,
                contparams,
                plot,
                normC,
                finalise_solution,
                callback_newton,
                event,
                verbosity = convert(UInt8, verbosity),
                filename)
end

Base.eltype(it::ContIterable{Tkind, Tprob, Talg, T}) where {Tkind, Tprob, Talg, T} = T
setparam(it::ContIterable{Tkind, Tprob, Talg, T}, p0::T) where {Tkind, Tprob, Talg, T} = setparam(it.prob, p0)

# getters
@inline getlens(it::ContIterable) = getlens(it.prob)
@inline getalg(it::ContIterable) = it.alg
@inline getprob(it::ContIterable) = it.prob
@inline callback(it::ContIterable) = it.callback_newton
record_from_solution(it::ContIterable) = record_from_solution(it.prob)
plot_solution(it::ContIterable) = plot_solution(it.prob)

get_lens_symbol(it::ContIterable) = get_lens_symbol(getlens(it))

# get the linear solver for Continuation
getlinsolver(iter::ContIterable) = getlinsolver(iter.alg)

@inline is_event_active(it::ContIterable) = !isnothing(it.event) && it.contparams.detect_event > 0
@inline compute_eigenelements(it::ContIterable) = compute_eigenelements(it.contparams) || (is_event_active(it) && compute_eigenelements(it.event))
@inline save_eigenvectors(it::ContIterable) = save_eigenvectors(it.contparams)

@inline getcontparams(it::ContIterable) = it.contparams
Base.length(it::ContIterable) = it.contparams.max_steps
@inline isindomain(it::ContIterable, p) = it.contparams.p_min < p < it.contparams.p_max
@inline is_on_boundary(it::ContIterable, p) = (it.contparams.p_min == p) || (p == it.contparams.p_max)
# clamp p value
clamp_predp(p::Number, it::AbstractContinuationIterable) = clamp(p, it.contparams.p_min, it.contparams.p_max)
####################################################################################################
"""
$(TYPEDEF)

Mutable structure containing the state of the continuation procedure. The fields are meant to change during the continuation procedure. 

!!! danger
    If you mutate these fields yourself, you can break the continuation procedure. Use the methods below to access the fields knowing that they do not yield copies.

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
- `copyto!(dest, state)`  copy `state` into `dest`
- `getsolution(state)` returns the current solution (x, p)
- `gettangent(state)` return the tangent at the current solution
- `getpredictor(state)` return the predictor at the current solution
- `getx(state)` returns the x component of the current solution
- `getp(state)` returns the p component of the current solution
- `get_previous_solution(state)` returns the previous solution (x, p)
- `getpreviousx(state)` returns the x component of the previous solution
- `getpreviousp(state)` returns the p component of the previous solution
- `is_stable(state)` whether the current state is stable
"""
Base.@kwdef mutable struct ContState{Tv, T, Teigvals, Teigvec, Tcb} <: AbstractContinuationState{Tv}
    z_pred::Tv                               # predictor
    τ::Tv                                    # tangent to the curve
    z::Tv                                    # current solution
    z_old::Tv                                # previous solution

    converged::Bool                          # boolean for newton correction
    itnewton::Int64 = 0                      # number of newton iteration (in corrector)
    itlinear::Int64 = 0                      # number of linear iteration (in newton corrector)

    step::Int64 = 0                          # current continuation step
    ds::T                                    # step size

    stopcontinuation::Bool = false           # boolean to stop continuation
    stepsizecontrol::Bool = true             # perform step size adaptation

    # the following values encode the current, previous number of unstable (resp. imaginary) eigen values
    # it is initialized as -1 when unknown
    n_unstable::Tuple{Int64, Int64}  = (-1, -1)    # (current, previous)
    n_imag::Tuple{Int64, Int64}      = (-1, -1)    # (current, previous)
    convergedEig::Bool               = true

    eigvals::Teigvals = nothing               # current eigenvalues
    eigvecs::Teigvec  = nothing               # current eigenvectors

    eventValue::Tcb = nothing                 # store the current event values
    in_bisection::Bool = false                # whether the state is in bisection for locating special points
end

function Base.copy(state::ContState)
    return ContState(
        z_pred = _copy(state.z_pred),
        τ      = _copy(state.τ),
        z      = _copy(state.z),
        z_old  = _copy(state.z_old),
        converged = state.converged,
        itnewton         = state.itnewton,
        itlinear         = state.itlinear,
        step             = state.step,
        ds               = state.ds,
        stopcontinuation = state.stopcontinuation,
        stepsizecontrol  = state.stepsizecontrol,
        n_unstable          = state.n_unstable,
        n_imag              = state.n_imag,
        convergedEig        = state.convergedEig,
        eigvals             = state.eigvals,
        eigvecs             = state.eigvecs, # can be removed? to save memory?
        eventValue          = state.eventValue,
        in_bisection        = state.in_bisection
    )
end

function Base.copyto!(dest::ContState, src::ContState)
        copyto!(dest.z_pred , src.z_pred)
        copyto!(dest.τ , src.τ)
        copyto!(dest.z , src.z)
        copyto!(dest.z_old , src.z_old)
        dest.converged        = src.converged
        dest.itnewton         = src.itnewton
        dest.itlinear         = src.itlinear
        dest.step             = src.step
        dest.ds               = src.ds
        dest.stopcontinuation = src.stopcontinuation
        dest.stepsizecontrol  = src.stepsizecontrol
        dest.n_unstable       = src.n_unstable
        dest.n_imag           = src.n_imag
        dest.convergedEig     = src.convergedEig
        dest.eigvals          = src.eigvals
        dest.eigvecs          = src.eigvecs # can be removed? to save memory?
        dest.eventValue       = src.eventValue
        dest.in_bisection     = src.in_bisection
    return dest
end

# getters
@inline converged(::Nothing) = false
@inline converged(state::AbstractContinuationState) = state.converged
@inline gettangent(state::AbstractContinuationState)    = state.τ
@inline getsolution(state::AbstractContinuationState)   = state.z
@inline getpredictor(state::AbstractContinuationState)  = state.z_pred
@inline getx(state::AbstractContinuationState)          = state.z.u
@inline getp(state::AbstractContinuationState)          = state.z.p
@inline get_previous_solution(state::AbstractContinuationState) = state.z_old
@inline getpreviousx(state::AbstractContinuationState) = state.z_old.u
@inline getpreviousp(state::AbstractContinuationState) = state.z_old.p
@inline is_stable(state::AbstractContinuationState)    = state.n_unstable[1] == 0
@inline stepsizecontrol(state::AbstractContinuationState) = state.stepsizecontrol
@inline in_bisection(state::AbstractContinuationState)    = state.in_bisection
@inline in_bisection(::Nothing) = false
####################################################################################################
# condition for halting the continuation procedure (i.e. when returning false)
@inline done(it::ContIterable, state::ContState) =
            (state.step <= it.contparams.max_steps) &&
            (isindomain(it, getp(state)) || state.step == 0) &&
            (state.stopcontinuation == false)

function get_state_summary(it, state::ContState{Tv, T, Teigvals}) where {Tv, T, Teigvals}
    x = getx(state)
    p = getp(state)
    pt = record_from_solution(it)(x, p; iter = it, state)
    stable = Teigvals!=Nothing ? is_stable(state) : nothing
    return mergefromuser(pt, 
                        (param = p,
                        itnewton = state.itnewton,
                        itlinear = state.itlinear,
                        ds = state.ds,
                        n_unstable = state.n_unstable[1],
                        n_imag = state.n_imag[1],
                        stable = stable,
                        step = state.step))
end

function update_stability!(state::ContState, n_unstable::Int, n_imag::Int, converged::Bool)
    state.n_unstable = (n_unstable, state.n_unstable[1])
    state.n_imag = (n_imag, state.n_imag[1])
    state.convergedEig = converged
end

function save!(br::ContResult, 
                it::AbstractContinuationIterable, 
                state::AbstractContinuationState)
    @debug "Saving data on the branch" state.step
    # update branch field
    push!(br.branch, get_state_summary(it, state))
    # save solution
    if it.contparams.save_sol_every_step > 0 && 
        (mod_counter(state.step, it.contparams.save_sol_every_step) || 
        ~done(it, state))
        push!(br.sol, (x = save_solution(it.prob, _copy(getx(state)), setparam(it.prob, getp(state))),
                       p = getp(state), 
                       step = state.step))
    end
    # save eigen elements
    if compute_eigenelements(it)
        if mod(state.step, it.contparams.save_eig_every_step) == 0
            push!(br.eig, (eigenvals = state.eigvals, 
                            eigenvecs = state.eigvecs, 
                            converged = state.convergedEig, 
                            step = state.step))
        end
    end
end

function plot_branch_cont(contres::ContResult, 
                            state::AbstractContinuationState, 
                            iter::ContIterable)
    if iter.plot && mod(state.step, getcontparams(iter).plot_every_step) == 0
        return plot_branch_cont(contres, 
                                state, 
                                iter, 
                                plot_solution(iter))
    end
end

function ContResult(it::AbstractContinuationIterable, 
                    state::AbstractContinuationState)
    x0 = _copy(getx(state))
    p0 = getp(state)
    pt = record_from_solution(it)(x0, p0; iter = it, state)
    return _contresult(it, state,
                        pt,
                        get_state_summary(it, state), 
                        save_solution(it.prob, x0, setparam(it.prob, p0)), 
                        getcontparams(it))
end

# function to update the state according to the event
function update_event!(it::ContIterable, state::ContState)
    # if the event is not active, we return false (not detected)
    if (is_event_active(it) == false); return false; end
    outcb = it.event(it, state)
    state.eventValue = (outcb, state.eventValue[1])
    # update the number of positive values
    return is_event_crossed(it.event, it, state)
end
####################################################################################################
# Continuation Iterator based on the Julia interface:
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration
# In a nutshell, one needs to provide
# - iterate(iter)
# - iterate(iter, state)
# function called at the beginning of the continuation
# used to determine first point on branch and tangent at this point
function Base.iterate(it::ContIterable; _verbosity = it.verbosity)
    # the keyword argument is to overwrite verbosity behaviour, like when locating bifurcations
    verbose = min(it.verbosity, _verbosity) > 0
    prob = it.prob
    p₀ = getparam(prob)

    verbose && printstyled("━"^54*"\n"*"─"^18*" ", typeof(getalg(it)).name.name, " "*"─"^18*"\n\n", bold = true, color = :red)

    # newton parameters
    (;p_min, p_max, max_steps, newton_options, η, ds) = it.contparams

    if !(p_min <= p₀ <= p_max)
        @error "Initial continuation parameter $(String(get_lens_symbol(getlens(prob)))) = $p₀ must be within bounds [p_min, p_max] = [$p_min, $p_max]"
        throw("Stopping continuation.")
    end

    # apply Newton algorithm to initial guess
    verbose && printstyled("━"^18*"  INITIAL GUESS   "*"━"^18, bold = true, color = :magenta)

    # we pass additional kwargs to newton so that it is sent to the newton callback
    sol₀ = solve(prob, Newton(), newton_options; 
                    normN = it.normC,
                    callback = callback(it),
                    iterationC = 0,
                    p = p₀)
    if  ~converged(sol₀)
        printstyled("\nNewton failed to converge for the initial guess on the branch. Residuals:\n", color=:red)
        display(sol₀.residuals)
        throw("Stopping continuation.")
    end
    verbose && (print("\n──▶ convergence of initial guess = ");printstyled("OK\n\n", color=:green))
    verbose && println("──▶ parameter = ", p₀, ", initial step")
    verbose && printstyled("\n"*"━"^18*" INITIAL TANGENT  "*"━"^18, bold = true, color = :magenta)

    sol₁ = solve(re_make(prob; params = setparam(it, p₀ + ds / η), u0 = sol₀.u),
                            Newton(),
                            newton_options; 
                            normN = it.normC,
                            callback = callback(it),
                            iterationC = 0,
                            p = p₀ + ds / η)
    @assert converged(sol₁) "Newton failed to converge. Required for the computation of the initial tangent."
    verbose && (print("\n──▶ convergence of the initial guess = ");printstyled("OK\n\n", color=:green))
    verbose && println("──▶ parameter = ", p₀ + ds/η, ", initial step (bis)")
    return iterate_from_two_points(it, sol₀.u, p₀, 
                                       sol₁.u, p₀ + ds / η; 
                                       _verbosity)
end

# same as previous function but when two (initial guesses) points are provided
function iterate_from_two_points(it::ContIterable, 
                                    u₀, p₀::T, 
                                    u₁, p₁::T; 
                                    _verbosity = it.verbosity) where T
    ds = it.contparams.ds
    z = BorderedArray(_copy(u₁), p₁)
    # compute eigenvalues to get the type. Necessary to give a ContResult
    eigvals, eigvecs = (nothing, nothing)
    cveig::Bool = true
    if compute_eigenelements(it)
        eigvals, eigvecs, cveig, = compute_eigenvalues(it, (z = z,), u₀, getparams(it.prob), it.contparams.nev)
    end

    # compute event value and store it into state
    cbval = is_event_active(it) ? initialize(it.event, T) : nothing
    state = ContState(z_pred = BorderedArray(_copy(u₀), p₀),
                        τ = BorderedArray(0*u₁, zero(p₁)),
                        z = z,
                        z_old = BorderedArray(_copy(u₀), p₀),
                        converged = true,
                        ds = it.contparams.ds,
                        eigvals = eigvals,
                        eigvecs = eigvecs,
                        eventValue = (cbval, cbval))

    # compute the state for the continuation algorithm
    # at this stage, the tangent is set up
    initialize!(state, it)

    # update stability
    if compute_eigenelements(it)
        (;n_unstable, n_imag) = is_stable(getcontparams(it), eigvals)
        update_stability!(state, n_unstable, n_imag, cveig)
    end

    # we update the event function result
    update_event!(it, state)
    return state, state
end

function Base.iterate(it::ContIterable, state::ContState; _verbosity = it.verbosity)
    if !done(it, state) return nothing end
    # the next line is to overwrite verbosity behaviour, for example when locating bifurcations
    verbosity = min(it.verbosity, _verbosity)
    verbose = verbosity > 0; verbose1 = verbosity > 1

    (;step, ds) = state

    if verbose
        printstyled("━"^55*"\nContinuation step $step \n", bold = true);
        @printf("Step size = %2.4e\n", ds); print("Parameter ", get_lens_symbol(it))
        @printf(" = %2.4e ⟶  %2.4e [guess]\n", getp(state), clamp_predp(state.z_pred.p, it))
    end

    # for PALC, z_pred contains the previous solution
    corrector!(state, it; iterationC = step, z0 = state.z)

    if converged(state)
        if verbose
            verbose1 && printstyled("──▶ Step Converged in $(state.itnewton) Nonlinear Iteration(s)\n", color = :green)
            print("Parameter ", get_lens_symbol(it))
            @printf(" = %2.4e ⟶  %2.4e\n", state.z_old.p, getp(state))
        end

        if compute_eigenelements(it)
            # this computes eigen-elements, store them in state and update the stability indices in state
            it_eigen = compute_eigenvalues!(it, state)
            verbose1 && printstyled(color=:green,"──▶ Computed ", length(state.eigvals), " eigenvalues in ", it_eigen, " iterations, #unstable = ", state.n_unstable[1], "\n")
        end
        state.step += 1
    else
        verbose && printstyled("Newton correction failed\n", color = :red)
    end

    # step size control, updates the parameter ds stored in state
    step_size_control!(state, it)

    # predictor: state.z_pred. The following method only mutates z_pred and τ
    getpredictor!(state, it)

    return state, state
end

function continuation!(it::ContIterable, state::ContState, contRes::ContResult)
    contparams = getcontparams(it)
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
        if converged(state) && (state.step <= it.contparams.max_steps) && (state.step > 0)
            # Detection of fold points based on parameter monotony, mutates contRes.specialpoint
            # if we detect bifurcations based on eigenvalues, we disable fold detection to avoid duplicates
            if contparams.detect_fold && contparams.detect_bifurcation < 2
                foldetected = locate_fold!(contRes, it, state)
                if foldetected && contparams.detect_loop
                    state.stopcontinuation |= detect_loop(contRes, nothing; verbose = verbose1)
                end
            end

            if contparams.detect_bifurcation > 1 && detect_bifurcation(state)
                status::Symbol = :guess
                _T = eltype(it)
                interval::Tuple{_T, _T} = getinterval(getpreviousp(state), getp(state))
                # if the detected bifurcation point involves a parameter values with is on
                # the boundary of the parameter domain, we disable bisection because it would
                # lead to infinite looping. Indeed, clamping messes up the `ds`
                if contparams.detect_bifurcation > 2 && ~is_on_boundary(it, getp(state))
                    verbose1 && printstyled(color = :red, "──▶ Bifurcation detected before p = ", getp(state), "\n")
                    # locate bifurcations with bisection, mutates state so that it stays very close the bifurcation point. It also updates the eigenelements at the current state. The call returns :guess or :converged
                    status, interval = locate_bifurcation!(it, state, it.verbosity > 2)
                end
                # we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
                if detect_bifurcation(state)
                    _, bif_pt = get_bifurcation_type(it, state, status, interval)
                    if bif_pt.type != :none; push!(contRes.specialpoint, bif_pt); end
                    # detect loop in the branch
                    contparams.detect_loop && (state.stopcontinuation |= detect_loop(contRes, bif_pt))
                end
            end

            if is_event_active(it)
                # check if an event occurred between the last 2 continuation steps
                event_detected = update_event!(it, state)
                verbose1 && printstyled(color = :blue, "──▶ Event values: ", state.eventValue[2], "\n"*" "^14*"──▶ ", state.eventValue[1],"\n")
                event_detected && (verbose && printstyled(color=:red, "──▶ Event detected before p = ", getp(state), "\n"))
                # save the event if detected and / or use bisection to locate it precisely
                if event_detected
                    _T = eltype(it); status = :guess; interval_event = (_T(0),_T(0))
                    if contparams.detect_event > 1
                        status, interval_event = locate_event!(it.event, it, state, it.verbosity > 2)
                    end
                    success, event_pt = get_event_type(it.event, it, state, it.verbosity, status, interval_event)
                    state.stopcontinuation |= ~success
                    event_pt = finalise_event!(event_pt, it.event, it, state, success)
                    if event_pt.type != :none
                        push!(contRes.specialpoint, event_pt)
                    end
                    # detect loop in the branch
                    contparams.detect_loop && (state.stopcontinuation |= detect_loop(contRes, event_pt))
                end
            end

            # save solution to file
            contparams.save_to_file && save_to_file(it, getx(state), getp(state), state.step, contRes)

            # call user saved finalise_solution function. If returns false, stop continuation
            # we put a OR to stop continuation if the stop was required before
            state.stopcontinuation |= ~it.finalise_solution(getsolution(state), state.τ, state.step, contRes; state = state, iter = it)

            # save current state in the branch
            save!(contRes, it, state)

            # plot current state
            plot_branch_cont(contRes, state, it)
        end
        ########################################################################################
        # body
        next = iterate(it, state)
    end

    plot_branch_cont(contRes, state, it)

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
- `contparams::ContinuationPar` parameters for continuation. See [`ContinuationPar`](@ref)

# Optional Arguments:
- `plot = false` whether to plot the solution/branch/spectrum while computing the branch
- `bothside = true` compute the branches on the two sides of the initial parameter value `p0`, merge them and return it.
- `normC = norm` norm used in the nonlinear solves
- `filename` to save the computed branch during continuation. The identifier .jld2 will be appended to this filename. This requires `using JLD2`.
- `callback_newton` callback for newton iterations. See docs of [`newton`](@ref). For example, it can be used to change the preconditioners.
- `finalise_solution = (z, tau, step, contResult; kwargs...) -> true` Function called at the end of each continuation step. Can be used to alter the continuation procedure (stop it by returning `false`), save personal data, plot... The notations are `z = BorderedArray(x, p)` where `x` (resp. `p`) is the current solution (resp. parameter value), `tau::BorderedArray` is the tangent at `z`, `step::Int` is the index of the current continuation step and `contResult` is the current branch. For advanced use:
    - the state `state::ContState` of the continuation iterator is passed in `kwargs`. This can be used for testing whether this is called from bisection for locating bifurcation points / events: `in_bisection(state)` for example. This allows to escape some personal code in this case.
    Note that you can have a better control over the continuation procedure by using an iterator, see [Iterator Interface](@ref).
    - the iterator `iter::ContIterable` of the continuation is passed in `kwargs`.
- `verbosity::Int = 0` controls the amount of information printed during the continuation process. Must belong to `{0,1,2,3}`. In case `contparams.newton_options.verbose = false`, the following is valid (otherwise the newton iterations are shown). Each case prints more information than the previous one:
    - case 0: print nothing
    - case 1: print basic information about the continuation: used predictor, step size and parameter values
    - case 2: print newton iterations number, stability of solution, detected bifurcations / events
    - case 3: print information during bisection to locate bifurcations / events
- `linear_algo` set the linear solver for the continuation algorithm `alg.` For example, `PALC` needs a linear solver for an enlarged problem (size `n+1` instead of `n`) and one thus needs to tune the one passed in `contparams.newton_options.linsolver`. This is a convenient argument to thus change the `alg` linear solver and is used mostly internally. The proper way is to pass directly to `alg` the correct linear solver.
- `kind::AbstractContinuationKind` [Internal] flag to describe continuation kind (equilibrium, codim 2, ...). Default = `EquilibriumCont()`

# Output:
- `contres::ContResult` composite type which contains the computed branch. See [`ContResult`](@ref) for more information.

!!! tip "Continuing the branch in the opposite direction"
    Just change the sign of `ds` in `ContinuationPar`.

!!! tip "Debug mode"
    Use debug mode to access more irformation about the progression of the continuation run, like iterative solvers convergence, problem update, ...
"""
function continuation(prob::AbstractBifurcationProblem,
                      alg::AbstractContinuationAlgorithm,
                      contparams::ContinuationPar;
                      linear_algo = nothing,
                      bothside::Bool = false,
                      kwargs...)
    # init the continuation parameters
    contparams = init(contparams, prob, alg)

    # update the parameters of alg
    # in the case of PALC, it creates a bordered linear solver based on the newton linear solver provided by the user
    alg = update(alg, contparams, linear_algo)

    # perform continuation
    itfwd = ContIterable(prob, alg, contparams; kwargs...)
    if bothside
        # we deepcopy the iterator here because some problems
        # are changed inplace like in Min. Aug. problems or 
        # periodic orbits computation
        itbwd = deepcopy(itfwd)
        @reset itbwd.contparams.ds = -contparams.ds

        resfwd = continuation(itfwd)
        resbwd = continuation(itbwd)
        contresult = _merge(resfwd, resbwd)

        # we have to update the branch if saved on a file
        itfwd.contparams.save_to_file && save_to_file(itfwd, contresult)
        return contresult
    else
        contresult = continuation(itfwd)
        # we have to update the branch if saved on a file,
        # basically this removes "branchfw" or "branchbw" in file and append "branch"
        itfwd.contparams.save_to_file && save_to_file(itfwd, contresult)
        return contresult
    end
end
