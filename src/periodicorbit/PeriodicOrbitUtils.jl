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