function modify_po_plot(::Union{BK_NoPlot, BK_Plots}, probPO::Union{PDMAProblem, NSMAProblem}, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (x, p; k...) -> _plotsol(getvec(x, probPO.prob), (prob = probPO, p = p); k...)
end

function modify_po_plot(::BK_Makie, probPO::Union{PDMAProblem, NSMAProblem}, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (ax, x, p; k...) -> _plotsol(ax, getvec(x, probPO.prob), (prob = probPO, p = p); k...)
end
####################################################################################################
function (finalizer::Finaliser{<: AbstractMABifurcationProblem})(z, tau, step, contResult; bisection = false, kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    state = get(kF, :state, nothing)
    success = converged(state)
    bisection = in_bisection(state)
    if success && mod_counter(step, updateSectionEveryStep) == 1 && bisection == false
        # we get the MA problem
        wrap_ma = finalizer.prob
        𝐏𝐛 = wrap_ma.prob
        prob_sh = 𝐏𝐛.prob_vf.prob
        # we get the state vector at bifurcation point
        x = getvec(z.u, 𝐏𝐛)
        # we get the parameters at the bifurcation point
        lenses = get_lenses(wrap_ma)
        p1, = getp(z.u, 𝐏𝐛)   # first parameter, ca bug pour Folds si p1,_ = getp(...)
        p2 = z.p              # second parameter
        pars = _set(getparams(prob_sh), lenses, (p1, p2))
        @debug "[Periodic orbit] update section"
        updatesection!(prob_sh, x, pars)
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(z, tau, step, contResult; prob = finalizer.prob, kF...)
    end
end

function (finalizer::Finaliser{<: AbstractMABifurcationProblem{ <: AbstractProblemMinimallyAugmented{ <: WrapPOColl}}})(Z, tau, step, contResult; kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    𝐏𝐛 = finalizer.prob.prob
    coll = 𝐏𝐛.prob_vf.prob
     # we get the state vector at bifurcation point
     x = getvec(Z.u, 𝐏𝐛)
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    state = get(kF, :state, nothing)
    success = converged(state)
    bisection = in_bisection(state)
    # mesh adaptation
    if success && coll.meshadapt && bisection == false
        @debug "[Collocation] update mesh"
        oldsol = _copy(x) # avoid possible overwrite in compute_error!
        oldmesh = get_times(coll) .* getperiod(coll, oldsol, nothing)
        adapt = compute_error!(coll, oldsol;
                    verbosity = coll.verbose_mesh_adapt,
                    K = coll.K
                    )
        if ~adapt.success # stop continuation if mesh adaptation fails
            return false
        end
    end
    if success && mod_counter(step, updateSectionEveryStep) == 1 && bisection == false
        @debug "[collocation] update section"
        updatesection!(coll, x, nothing) # collocation does not need the parameter for updatesection!
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(Z, tau, step, contResult; prob = coll, kF...)
    end
end