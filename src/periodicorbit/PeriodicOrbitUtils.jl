abstract type PeriodicOrbitAlgorithm end

"""
$(SIGNATURES)

This function returns several useful quantities regarding a Hopf bifurcation point. More precisely, it returns:
- the parameter value at which a Hopf bifurcation occurs
- the period of the bifurcated periodic orbit
- a guess for the bifurcated periodic orbit
- the equilibrium at the Hopf bifurcation point
- the eigenvector at the Hopf bifurcation point.

The arguments are
- `br`: the continuation branch which lists the Hopf bifurcation points
- `ind_hopf`: index of the bifurcation branch, as in `br.specialpoint`
- `eigsolver`: the eigen solver used to find the eigenvectors
- `M` number of time slices in the periodic orbit guess
- `amplitude`: amplitude of the periodic orbit guess
"""
function guess_from_hopf(br, ind_hopf, eigsolver::AbstractEigenSolver, M::Int, amplitude; phase = 0)
    hopfpoint = hopf_point(br, ind_hopf)
    specialpoint = br.specialpoint[ind_hopf]

    # parameter value at the Hopf point
    p_hopf = hopfpoint.p[1]

    # frequency at the Hopf point
    ωH  = hopfpoint.p[end] |> abs

    # vec_hopf is the eigenvector for the eigenvalues iω
    vec_hopf = geteigenvector(eigsolver, br.eig[specialpoint.idx][2], specialpoint.ind_ev-1)
    vec_hopf ./= norm(vec_hopf)

    orbitguess = [real.(hopfpoint.u .+ amplitude .* vec_hopf .* exp(-2pi * complex(0, 1) .* (ii/(M-1) - phase))) for ii in 0:M-1]

    return p_hopf, 2π/ωH, orbitguess, hopfpoint, vec_hopf
end
####################################################################################################
@inline update!(::WrapPOColl, args...; k...) = update_default(args...; k...)
@inline update!(::WrapPOSh, args...; k...) = update_default(args...; k...)
@inline update!(::WrapPOTrap, args...; k...) = update_default(args...; k...)
####################################################################################################
"""
($SIGNATURES)

Update the continuation parameters according to a problem. This can be useful for branching from PD points where the linear solvers have to be updated, e.g. the number of unknowns is roughly doubled.
"""
function _update_cont_params(contParams::ContinuationPar, pb::AbstractShootingProblem, orbitguess)
    if contParams.newton_options.linsolver isa GMRESIterativeSolvers
        @reset contParams.newton_options.linsolver.N = length(orbitguess)
    elseif contParams.newton_options.linsolver isa FloquetWrapperLS
        if contParams.newton_options.linsolver.solver isa GMRESIterativeSolvers
            @reset contParams.newton_options.linsolver.solver.N = length(orbitguess)
        end
    end
    return contParams
end

function _update_cont_params(contParams::ContinuationPar, coll::PeriodicOrbitOCollProblem, orbitguess)
    if contParams.newton_options.linsolver isa BifurcationKit.FloquetWrapperLS{<:COPLS}
        @reset contParams.newton_options.linsolver.solver = COPLS(COPCACHE(coll, Val(0)))
    end
    return contParams
end

function _update_cont_params(contParams::ContinuationPar, pb::AbstractPOFDProblem, orbitguess)
    return contParams
end
####################################################################################################
function modify_po_finalise(prob, kwargs, updateSectionEveryStep)
    return Finaliser(prob, get(kwargs, :finalise_solution, nothing), updateSectionEveryStep)
end

function (finalizer::Finaliser{ <: AbstractPeriodicOrbitProblem})(z, tau, step, contResult; kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    state = get(kF, :state, nothing)
    success = converged(state)
    bisection = in_bisection(state)
    if success && mod_counter(step, updateSectionEveryStep) == 1 && bisection == false
        @debug "[Periodic orbit] update section"
        # Trapezoid and Shooting need the parameters for section update:
        updatesection!(finalizer.prob, z.u, setparam(contResult, z.p))
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(z, tau, step, contResult; prob = finalizer.prob, kF...)
    end
end

# version specific to collocation. Handle mesh adaptation
function (finalizer::Finaliser{ <: Union{ <: PeriodicOrbitOCollProblem,
                                <: WrapPOColl}})(z, tau, step, contResult; kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    coll = finalizer.prob
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    state = get(kF, :state, nothing)
    success = converged(state)
    bisection = in_bisection(state)
    is_mesh_updated = false

    # mesh adaptation
    if success &&
            coll.meshadapt && 
            bisection == false && 
            mod_counter(step, updateSectionEveryStep) == 1 &&
            step > 2
        @debug "[Collocation] update mesh"
        is_mesh_updated = true
        oldsol = _copy(z) # avoid possible overwrite in compute_error!
        oldmesh = get_times(coll) .* getperiod(coll, oldsol.u, nothing)
        adapt = compute_error!(coll, oldsol.u;
                    verbosity = coll.verbose_mesh_adapt,
                    K = coll.K,
                    par = setparam(contResult, z.p)
                    )
        if ~adapt.success # stop continuation if mesh adaptation fails
            return false
        end
    end

    if success && mod_counter(step, updateSectionEveryStep) == 1 && bisection == false
        @debug "[collocation] update section"
        updatesection!(coll, z.u, setparam(contResult, z.p))
    end
    if is_mesh_updated
        # we recompute the tangent predictor
        it = get(kF, :iter, nothing)
        # @debug "[collocation] update predictor"
        getpredictor!(state, it)
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(z, tau, step, contResult; prob = coll, kF...)
    end
end

function modify_po_record(probPO, pars, lens; kwargs...)
    if ~isnothing(get(kwargs, :record_from_solution, nothing))
        _recordsol0 = get(kwargs, :record_from_solution, nothing)
        @assert ~isnothing(_recordsol0) "Please open an issue on the website."
        if probPO isa AbstractShootingProblem
            return _recordsol = (x, p; k...) -> _recordsol0(x, (prob = probPO, p = set(pars, lens, p)); k...)
        else
            return _recordsol = (x, p; k...) -> _recordsol0(x, (prob = probPO, p = p); k...)
        end
    else
        if probPO isa AbstractPODiffProblem
            # FAIRE FONCTION NE PAS FAIRE ANONYMOUS
            return _recordsol = (x, p; k...) -> begin
                period = getperiod(probPO, x, set(pars, lens, p))
                sol = get_periodic_orbit(probPO, x, set(pars, lens, p))
                _min, _max = @views extrema(sol[1,:])
                min = @views minimum(sol[1,:])
                return (max = _max, min = _min, amplitude = _max - _min, period = period)
            end
        else
            return _recordsol = (x, p; k...) -> (period = getperiod(probPO, x, set(pars, lens, p)),)
        end
    end
end
####################################################################################################
function modify_po_plot(::Union{BK_NoPlot, BK_Plots}, probPO, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (x, p; k...) -> _plotsol(x, (prob = probPO, p = set(pars, lens, p)); k...)
end

function modify_po_plot(::BK_Makie, probPO, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (ax, x, p; k...) -> _plotsol(ax, x, (prob = probPO, p = set(pars, lens, p)); k...)
end

modify_po_plot(probPO, pars, lens; kwargs...) = modify_po_plot(get_plot_backend(), probPO, pars, lens; kwargs...)