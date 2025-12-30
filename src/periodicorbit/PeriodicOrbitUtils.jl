"""
($SIGNATURES)

Update the continuation parameters according to a problem. This can be useful for branching from PD points where the linear solvers have to be updated, e.g. the number of unknowns is roughly doubled.
"""
function _update_cont_params(contParams::ContinuationPar, pb::AbstractShootingProblem, orbitguess)
    if contParams.newton_options.linsolver isa GMRESIterativeSolvers
        @reset contParams.newton_options.linsolver.N = length(orbitguess)
    end
    return contParams
end

function _update_cont_params(cont_params::ContinuationPar, coll::PeriodicOrbitOCollProblem, orbitguess)
    if cont_params.newton_options.linsolver isa COPLS
        @reset cont_params.newton_options.linsolver = COPLS(COPCACHE(coll, Val(0)))
    end
    return cont_params
end

@inline _update_cont_params(cont_params::ContinuationPar, pb::AbstractPOFDProblem, orbitguess) = cont_params
####################################################################################################
# @inline user_passed_pofunction(probwrap.recordFromSolution) = user_passed_pofunction(probwrap.recordFromSolution)
@inline user_passed_pofunction(rf::RecordForPeriodicOrbits) = user_passed_function(rf.user_record_from_solution)

function __user_record_solution_periodic_orbit(pbwrap, ::UserPassedFunction, iter, state)
    p = set(getparams(pbwrap), getlens(pbwrap), getp(state))
    return pbwrap.recordFromSolution(getx(state), (prob = pbwrap.prob, p = p); iter, state)
end

function __user_record_solution_periodic_orbit(pbwrap, ::NoUserPassedFunction, iter, state)
    return (period = getperiod(pbwrap, getx(state), set(getparams(pbwrap), getlens(pbwrap), getp(state))),)
end

function __user_record_solution_periodic_orbit(pbwrap::AbstractWrapperFDProblem, ::UserPassedFunction, iter, state)
    return pbwrap.recordFromSolution(getx(state), (prob = pbwrap.prob, p = getp(state)); iter, state)
end

function __user_record_solution_periodic_orbit(pbwrap::AbstractWrapperFDProblem, ::NoUserPassedFunction, iter::ContIterable{Tkind}, state) where {Tkind}
    prob_po = pbwrap.prob
    x = getx(state)
    p = getp(state)
    period = getperiod(prob_po, x, nothing)
    return (;period)
    sol = get_periodic_orbit(prob_po, x, nothing)
    _min, _max = @views extrema(sol[1, :])
    return (;max = _max, min = _min, amplitude = _max - _min, period)
end

function record_from_solution(iter::ContIterable{PeriodicOrbitCont, <: AbstractWrapperPOProblem},
                              state::AbstractContinuationState)
    probwrap = getprob(iter)
    __user_record_solution_periodic_orbit(probwrap, user_passed_pofunction(probwrap.recordFromSolution), iter, state)
end

function record_from_solution(iter::ContIterable{FoldPeriodicOrbitCont, <: FoldMAProblem},
                              state::AbstractContinuationState)
    probma = getprob(iter) # TODO Make small function for this and merge with the one in MinAugFold.jl
    𝐅 = get_formulation(probma)
    probwrap = 𝐅.prob_vf
    lens1, lens2 = get_lenses(probma)
    lenses = get_lens_symbol(lens1, lens2)
    u = getx(state)
    p = getp(state)

    return (; zip(lenses, (getp(u, 𝐅), p))..., 
                    BT = 𝐅.BT, 
                    CP = 𝐅.CP, 
                    ZH = 𝐅.ZH,
                    _namedrecordfromsol(__user_record_solution_periodic_orbit(probwrap, user_passed_pofunction(probwrap.recordFromSolution), iter, state))...
                    ) 
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