function compute_eigenvalues(eig::FoldEig, iter::ContIterable{FoldPeriodicOrbitCont}, state, u0, par, nev = iter.contparams.nev; k...)
    z = state.z
    x = getvec(z.u) # fold point
    p1 = getp(z.u)  # first parameter
    p2 = z.p        # second parameter

    probma = getprob(iter)
    lens1, lens2 = get_lenses(probma)
    newpar = set(getparams(probma), lens1, p1)
    newpar = set(newpar, lens2, p2)
    compute_eigenvalues(eig.eigsolver, iter, state, x, newpar, nev; k...)
end
####################################################################################################
function modify_po_plot(::Union{BK_NoPlot, BK_Plots}, probPO::Union{PDMAProblem, NSMAProblem, FoldMAProblem}, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (x, p; k...) -> _plotsol(getvec(x, probPO.prob), (prob = probPO, p = p); k...)
end

function modify_po_plot(::BK_Makie, probPO::Union{PDMAProblem, NSMAProblem, FoldMAProblem}, pars, lens; kwargs...)
    _plotsol = get(kwargs, :plot_solution, nothing)
    _plotsol2 = isnothing(_plotsol) ? plot_default : (ax, x, p; k...) -> _plotsol(ax, getvec(x, probPO.prob), (prob = probPO, p = p); k...)
end
####################################################################################################
## TODO MERGE WITH UPDATE!(COLLOCATION)

__get_discretization(pb::AbstractWrapperPOProblem) = get_discretization(pb)
__get_discretization(pb::AbstractMinimallyAugmentedFormulation) = __get_discretization(pb.prob_vf)
__get_discretization(pb::AbstractPeriodicOrbitProblem) = pb

function __update_codim1_po!(𝐏𝐛, iter, state)
    # we get the AbstractPeriodicOrbitProblem
    disc_po = __get_discretization(𝐏𝐛)
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    success = converged(state)
    bisection = in_bisection(state)
    if success && mod_counter(step, disc_po.update_section_every_step) == 1 && bisection == false
        # state vector at bifurcation point
        z = getsolution(state)
        x = getvec(z.u, 𝐏𝐛)
        # parameters at the bifurcation point
        lenses = get_lenses(getprob(iter))
        p1, = getp(z.u, 𝐏𝐛)   # first parameter, TODO it errors for Folds if p1, _ = getp(...)
        p2 = z.p              # second parameter
        pars = _set(getparams(disc_po), lenses, (p1, p2))
        @debug "[Periodic orbit] update section"
        updatesection!(disc_po, x, pars)
    end
    return true
end

function update!(𝐏𝐛::AbstractMinimallyAugmentedFormulation, 
                 iter::ContIterable{ <: TwoParamPeriodicOrbitCont},
                 state)
    return __update_codim1_po!(𝐏𝐛, iter, state)
end

function update!(𝐏𝐛::FoldMinimallyAugmentedFormulation, 
                 iter::ContIterable{ <: TwoParamPeriodicOrbitCont},
                 state)
    return __update_codim1_po!(𝐏𝐛, iter, state)
end

## TODO MERGE WITH UPDATE!(COLLOCATION)
function update!(𝐏𝐛::AbstractMinimallyAugmentedFormulation{ <: WrapPOColl},
                iter::ContIterable{ <: TwoParamPeriodicOrbitCont},
                state)
    coll = 𝐏𝐛.prob_vf.prob
    # state vector at bifurcation point
    Z = getsolution(state)
    x = getvec(Z.u, 𝐏𝐛)
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
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
    if success && mod_counter(step, coll.update_section_every_step) == 1 && bisection == false
        @debug "[collocation] update section"
        updatesection!(coll, x, nothing) # collocation does not need the parameter for updatesection!
    end
    return true
end