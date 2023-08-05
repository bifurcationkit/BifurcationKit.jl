_perturbSolution(x, p, id) = x
_acceptSolution(x, p) = true
_updateDeflationOp(defOp, x, p) = push!(defOp, x)

"""
$(TYPEDEF)

Structure which holds the parameters specific to Deflated continuation.

# Fields

$(TYPEDFIELDS)

"""
@with_kw_noshow struct DefCont{Tdo, Talg, Tps, Tas, Tud, Tk} <: AbstractContinuationAlgorithm
    "Deflation operator, `::DeflationOperator`"
    deflationOperator::Tdo = nothing
    "Used as a predictor, `::AbstractContinuationAlgorithm`. For example `PALC()`, `Natural()`,..."
    alg::Talg = PALC()
    "maximum number of (active) branches to be computed"
    maxBranches::Int = 100
    "whether to seek new (deflated) solution at every step"
    seekEveryStep::Int = 1
    "maximum number of deflated Newton iterations"
    maxIterDefOp::Int = 5
    "perturb function"
    perturbSolution::Tps = _perturbSolution
    "accept (solution) function"
    acceptSolution::Tas = _acceptSolution
    "function to update the deflation operator, ie pushing new solutions"
    updateDeflationOp::Tud = _updateDeflationOp
    "jacobian for deflated newton. Can be `DeflatedProblemCustomLS()`, or `Val(:autodiff)`, `Val(:fullIterative)`"
    jacobian::Tk = DeflatedProblemCustomLS()
end

# iterable which contains the options associated with Deflated Continuation
@with_kw struct DefContIterable{Tit, Talg <: DefCont}
    it::Tit                        # replicate continuation iterator
    alg::Talg
end

"""

$(TYPEDEF)

Structure holding the result from deflated continuation.

$(TYPEDFIELDS)
"""
struct DCResult{Tprob, Tbr, Tit, Tsol, Talg} <: AbstractBranchResult
    "Bifurcation problem"
    prob::Tprob
    "Branches of solution"
    branches::Tbr
    "Continuation iterator"
    iter::Tit
    "Solutions"
    sol::Tsol
    "Algorithm"
    alg::Talg
end
Base.lastindex(br::DCResult) = lastindex(br.branches)
Base.getindex(br::DCResult, k::Int) = getindex(br.branches, k)
Base.length(br::DCResult) = length(br.branches)

# state specific to Deflated Continuation, it is updated during the continuation process
mutable struct DCState{T, Tstate}
    tmp::T
    state::Tstate
    isactive::Bool
    DCState(sol::T) where T = new{T, Nothing}(_copy(sol), nothing, true)
    DCState(sol::T, state::ContState, active = true) where {T} = new{T, typeof(state)}(_copy(sol), state, active)
end
# whether the branch is active
isActive(dc::DCState) = dc.isactive
# getters
getx(dc::DCState) = getx(dc.state)
getp(dc::DCState) = getp(dc.state)

function updatebranch!(dcIter::DefContIterable,
                        dcstate::DCState,
                        contResult::ContResult,
                        defOp::DeflationOperator;
                        current_param,
                        step)
    isActive(dcstate) == false &&  return false, 0
    state = dcstate.state         # continuation state
    it = dcIter.it                 # continuation iterator
    alg = dcIter.alg
    @unpack step, ds = state
    @unpack verbosity = it
    state.z_pred.p = current_param

    getPredictor!(state, it)
    pbnew = reMake(it.prob; u0 = _copy(getx(state)), params = setParam(it, current_param))
    sol1 = newton(pbnew, defOp, it.contParams.newtonOptions, alg.jacobian;
                    normN = it.normC,
                    callback = it.callbackN,
                    iterationC = step,
                    z0 = state.z)

    if converged(sol1)
        # record previous parameter (cheap) and update current solution
        copyto!(state.z.u, sol1.u); state.z.p = current_param
        state.z_old.p = current_param

        # get tangent, it only mutates tau
        getPredictor!(state, it)

        # call user function to deal with DeflationOperator, allows to tackle symmetries
        alg.updateDeflationOp(defOp, sol1.u, current_param)

        # compute stability and bifurcation points
        computeEigenElements(it.contParams) && computeEigenvalues!(it, state)

        # verbose stability
        (it.verbosity > 0) && printstyled(color=:green,"├─ Computed ", length(state.eigvals), " eigenvalues, #unstable = ", state.n_unstable[1], "\n")

        if it.contParams.detectBifurcation > 1 && detectBifucation(state)
            # we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
            if detectBifucation(state)
                _, bifpt = getBifurcationType(it, state, :guess, getinterval(current_param, current_param-ds))
                if bifpt.type != :none; push!(contResult.specialpoint, bifpt); end
            end
        end
        state.step += 1
        save!(contResult, it, state)
    else
        dcstate.isactive = false
        # save the last solution
        push!(contResult.sol, (x = _copy(getx(state)), p = getp(state), step = state.step))
    end
    return converged(sol1), sol1.itnewton
end

# this is a function barrier to make Deflated continuation type stable
# it returns the set of states and the ContResult
function getStatesContResults(iter::DefContIterable, roots::Vector{Tvec}) where Tvec
    @assert length(roots) > 0 "You must provide roots in the deflation operators. These roots are used as initial conditions for the deflated continuation process."
    contIt = iter.it
    copyto!(contIt.prob.u0, roots[1])
    states = [DCState(rt, iterate(contIt)[1]) for rt in roots]
    # allocate branches to hold the result
    branches = [ContResult(contIt, st.state) for st in states]
    return states, branches
end

# plotting functions
function plotDContBranch(branches, nbrs::Int, nactive::Int, nstep::Int)
    plot(branches..., label = "", title  = "$nbrs branches, actives = $(nactive), step = $nstep")
    for br in branches
        length(br) > 1 && plot!([br.branch[end-1:end].param], [getproperty(br.branch,1)[end-1:end]], label = "", arrow = true, color = :red)
    end
    scatter!([br.branch[1].param for br in branches], [br.branch[1][1] for br in branches], marker = :cross, color=:green, label = "") |> display
end
plotAllDCBranch(branches) = display(plot(branches..., label = ""))


"""
$(SIGNATURES)

This function computes the set of curves of solutions `γ(s) = (x(s), p(s))` to the equation `F(x,p) = 0` based on the algorithm of **deflated continuation** as described in Farrell, Patrick E., Casper H. L. Beentjes, and Ásgeir Birkisson. “The Computation of Disconnected Bifurcation Diagrams.” ArXiv:1603.00809 [Math], March 2, 2016. http://arxiv.org/abs/1603.00809.

Depending on the options in `contParams`, it can locate the bifurcation points on each branch. Note that you can specify different predictors using `alg`.

# Arguments:
- `prob::AbstractBifurcationProblem` bifurcation problem
- `alg::DefCont`, deflated continuation algorithm, see [`DefCont`](@ref)
- `contParams` parameters for continuation. See [`ContinuationPar`](@ref) for more information about the options

# Optional Arguments:
- `plot = false` whether to plot the solution while computing,
- `callbackN` callback for newton iterations. see docs for `newton`. Can be used to change preconditioners or affect the newton iterations. In the deflation part of the algorithm, when seeking for new branches, the callback is passed the keyword argument `fromDeflatedNewton = true` to tell the user can it is not in the continuation part (regular newton) of the algorithm,
- `verbosity::Int` controls the amount of information printed during the continuation process. Must belong to `{0,⋯,5}`,
- `normN = norm` norm used in the Newton solves,
- `dotPALC = (x, y) -> dot(x, y) / length(x)`, dot product used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see online docs on [PALC](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/)). This argument can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...),

# Outputs:
- `contres::DCResult` composite type which contains the computed branches. See [`ContResult`](@ref) for more information,
"""
function continuation(prob::AbstractBifurcationProblem,
             algdc::DefCont,
            contParams::ContinuationPar;
            verbosity::Int = 2,
            plot::Bool = true,
            linearAlgo = BorderingBLS(contParams.newtonOptions.linsolver),
            dotPALC = DotTheta(),
            callbackN = cbDefault,
            filename = "branch-" * string(Dates.now()),
            normN = norm,
            kwcont...)

    algdc = @set algdc.maxIterDefOp = algdc.maxIterDefOp * contParams.newtonOptions.maxIter
    # allow to remove the corner case and associated specific return variables, type stable
    defOp = algdc.deflationOperator
    @assert length(defOp) > 0 "You must provide at least one guess"

    # we make a copy of the deflation operator
    deflationOp = copy(defOp)

    verbosity > 0 && printstyled(color=:magenta, "━"^31*"\n")
    verbosity > 0 && printstyled(color=:magenta, "──▶ There are $(length(deflationOp)) branche(s)\n")

    # underlying continuation iterator
    # we "hack" the saveSolEveryStep option because we always want to record the first point on each branch
    contIt = ContIterable(prob, algdc.alg, ContinuationPar(contParams, saveSolEveryStep = contParams.saveSolEveryStep == 0 ? Int(1e14) : contParams.saveSolEveryStep);
                    plot = plot,
                    normC = normN,
                    callbackN = callbackN,
                    verbosity = verbosity-2,
                    kwcont...)

    iter = DefContIterable(contIt, algdc)

    return deflatedContinuation(iter, deflationOp, contParams, verbosity, plot)
end

function deflatedContinuation(dcIter::DefContIterable,
                            deflationOp::DeflationOperator,
                            contParams,
                            verbosity,
                            plot)

    dcstates, branches = getStatesContResults(dcIter, deflationOp.roots)

    contIt = dcIter.it
    alg = dcIter.alg
    par = getParams(contIt.prob)
    lens = getLens(contIt)
    current_param = get(par, lens)

    # extract the newton options
    optnewton = contParams.newtonOptions

    # function to get new solutions based on Deflated Newton
    function getNewSolution(_st::DCState, _p::Real, _idb)
        u0 = _copy(getx(_st)) # maybe we can remove this copy?
        prob_df = reMake(contIt.prob;
                            u0 = alg.perturbSolution(u0, _p, _idb),
                            params = set(par, lens, _p))
        soln = newton(prob_df, deflationOp,
                setproperties(optnewton; maxIter = alg.maxIterDefOp);
                normN = contIt.normC,
                callback = contIt.callbackN,
                fromDeflatedNewton = true)
        # we confirm that the residual for the non deflated problem is small
        # this should be the case unless the user pass "bad" options
        @set! soln.converged = soln.converged && contIt.normC(residual(contIt.prob, soln.u, prob_df.params)) < optnewton.tol
        if minimum(contIt.normC(soln.u - rt) for rt in deflationOp.roots) < optnewton.tol
            @set! soln.converged = false
        end
        return soln
    end

    nstep = 0
    while ((contParams.pMin < current_param < contParams.pMax) || nstep == 0) &&
                 nstep < contParams.maxSteps
        # we update the parameter value
        current_param = clampPredp(current_param + contParams.ds, contIt)

        verbosity > 0 && println("├"*"──"^31)
        nactive = mapreduce(isActive, +, dcstates)
        verbosity > 0 && println("├─ step = $nstep has $(nactive)/$(length(branches)) active branche(s), p = $current_param")

        # we empty the set of known solutions
        empty!(deflationOp)

        # update the known branches
        for (idb, dcstate) in enumerate(dcstates)
            # this computes the solution for the new parameter value current_param
            # it also updates deflationOp
            # if the branch is inactive, it returns
            flag, itnewton = updatebranch!(dcIter, dcstate, branches[idb], deflationOp;
                    current_param = current_param,
                    step = nstep)
            (verbosity>=2 && isActive(dcstate)) && println("├─── Continuation of branch $idb in $itnewton Iterations")
            (verbosity>=1 && ~flag && isActive(dcstate)) && printstyled(color=:red, "──▶ Fold for branch $idb) ?\n")
        end

        verbosity>1 && printstyled(color = :magenta,"├─ looking for new branches\n")
        # number of branches
        nbrs = length(dcstates)
        # number of active branches
        nactive = mapreduce(isActive, +, dcstates)
        if plot && mod(nstep, contParams.plotEveryStep) == 0
            plotDContBranch(branches, nbrs, nactive, nstep)
        end

        # only look for new branches if the number of active branches is too small
        if mod(nstep, alg.seekEveryStep) == 0 && nactive < alg.maxBranches
            n_active = 0
            # we restrict to 1:nbrs because we don't want to update the newly found branches
            for (idb, dcstate) in enumerate(dcstates[1:nbrs])
                if isActive(dcstate) && (n_active < alg.maxBranches)
                    n_active += 1
                    _success = true
                    verbosity >= 2 && println("├───▶ Deflating branch $idb")
                    while _success
                        sol1 = getNewSolution(dcstate, current_param, idb)
                        _success = converged(sol1)
                        if _success && contIt.normC(sol1.u - getx(dcstate)) < optnewton.tol
                            @error "Same solution found for identical parameter value!!"
                            _success = false
                        end
                        if _success && dcIter.alg.acceptSolution(sol1.u, current_param)
                            verbosity>=1 && printstyled(color=:green, "├───▶ new solution for branch $idb \n")
                            push!(deflationOp.roots, sol1.u)

                            # create a new iterator and iterate it once to set up the ContState
                            contitnew = @set contIt.prob = reMake(contIt.prob, u0 = sol1.u, params = sol1.prob.prob.params)
                            push!(dcstates, DCState(sol1.u, iterate(contitnew)[1], n_active+1<alg.maxBranches))

                            push!(branches, ContResult(contitnew, dcstates[end].state))
                        end
                    end
                end
            end
        end
        nstep += 1
    end
    plot && plotAllDCBranch(branches)
    return DCResult(contIt.prob, branches, contIt, [getx(c.state) for c in dcstates if isActive(c)], dcIter.alg)
end

# function mergeBranches!(brs::Vector{T}, iter::ContIterable; plot = false, iterbrsmax = 2) where {T <: ContResult}
#     @warn "il faut length(br) > 1"
#     # get the continuation parameters
#     @unpack pMin, pMax, ds = brs[1].contparams
#     # we update the parameter span with provided initial conditions
#     pMin = ds > 0 ? brs[1].branch[1].param : pMin
#     pMax = ds < 0 ? brs[1].branch[1].param : pMax
#     @show pMin pMax
#
#     function _setnewsol(_x, _p0, _pMin, _pMax, _sign = -1)
#         @show iter.contParams.maxSteps
#         _iter2 = @set iter.par = setParam(iter, _p0)
#         _iter2 = @set _iter2.contParams = setproperties(_iter2.contParams; ds = abs(iter.contParams.ds) * _sign, pMin = _pMin, pMax = _pMax)
#         copyto!(_iter2.x0, _x)
#         return _iter2
#     end
#
#     # test if the branch is done
#     doneBranch(_br::ContResult) = _br.branch[1].param in [pMin, pMax] && _br.branch[end].param in [pMin, pMax]
#     doneBranch(_brs::Vector{ <: ContResult}) = mapreduce(doneBranch, +, _brs)
#
#     # test if (x,p) belongs to a branch
#     function isin(_br::ContResult, x, p)
#         tol = _br.contparams.newtonOptions.tol
#         for (_id, _sol) in pairs(_br.sol)
#             if max(iter.normC(_sol.x - x), abs(_sol.p - p)) < tol
#                 return true, _id
#             end
#         end
#          return false, 0
#     end
#
#     # function to find the branch to which (x, p) belongs
#     # return found?, which?, end?
#     function intersectBranch(_brs::Vector{ <: ContResult}, x, p, idavoid = 0)
#         for (_id, _br) in pairs(_brs)
#             _res = isin(_br, x, p)
#             if (_res[1] && _id != idavoid)
#                 return (_res[1], _id, _res[2])  # return if branch id found
#             end
#         end
#         return false, 0, 0
#     end
#
#     function getParamValues(_brs::Vector{ <: ContResult})
#         ps = [(p = sol.p, id = ii) for (ii, br) in pairs(brs) if length(br) >1 for sol in br.sol[[1,end]]]
#         sort!(ps, by = x -> x.p, rev = true )
#         return ps
#     end
#     # @assert doneBranch(brs[1]) "error"
#
#     printstyled(color=:blue,"━"^50*"\n")
#     iterbrs = 0
#
#     while doneBranch(brs) < length(brs) && iterbrs < iterbrsmax
#         printstyled(color=:green, "━"^50*" - $iterbrs, #br = $(length(brs)) \n")
#         for (id, br) in pairs(brs)
#             print("──▶ branch $id, # = ", length(br), ", isdone = ")
#             printstyled("$(doneBranch(br)) \n", color = (doneBranch(br) ? :green : :red))
#             if ~doneBranch(br) && length(br) > 1 # if there is a single point, discard
#                 printstyled(color=:magenta,"──▶ branch $id not finished\n")
#                 # we found an un-finished branch
#                 iterbr = 0
#                 # we use as starting point the one
#                 startingpoint = ((br.branch[1].param > br.branch[2].param) & (br.branch[1].param < pMax)) ? (br.sol[1]..., first = true) : (br.sol[end]..., first = false)
#                 @show startingpoint.p
#                 if length(br) > 1
#                     # δp is where we want to go:
#                     δp = startingpoint.first ? br.branch[1].param - br.branch[2].param : br.branch[end].param - br.branch[end-1].param
#                 else
#                     @error "branch not large enough"
#                     δp = -abs(ds)
#                 end
#                 δp = δp == 0 ? -abs(ds) : δp
#                 _ps = getParamValues(brs)
#                 display(_ps)
#                 if δp > 0
#                     @assert 1==1 "searchsorted?"
#                     idp = findfirst(x -> x.p > startingpoint.p, _ps)
#                     @show  idp
#                     pspan = (startingpoint.p, ~isnothing(idp) ? _ps[idp].p : pMax)
#                     printstyled(color=:blue, "──▶ to the right\n")
#                 else
#                     @assert 1==1 "searchsorted?"
#                     idp = findfirst(x -> x.p < startingpoint.p, _ps)
#                     pspan = (~isnothing(idp) ? _ps[idp].p : pMin, startingpoint.p)
#                     printstyled(color=:green, "──▶ to the left\n")
#                 end
#
#                 # we define the parameter span
#                 iter2 = _setnewsol(startingpoint.x, startingpoint.p, pspan..., sign(δp))
#                 while ~doneBranch(brs[id]) && iterbr < 1
#                     brright, = continuation(iter2)
#                     brs[id] = _merge(br, brright)
#                     println("──▶ branch from ",brs[id].sol[1].p," to ", brright.sol[end].p)
#                     # return brright
#
#                     res = intersectBranch(brs, brright.sol[end].x, brright.sol[end].p,id)
#                     if res[1]  #we found a curve to merge with
#                             printstyled(color=:red, "──▶ Merge branch with ", res[2],"\n")
#                         brs[id] = _merge(brs[res[2]], brs[id])
#                             printstyled(color=:green,"\n──▶ the branch is complete = ", doneBranch(brs[id]),"\n")
#                         # we remove the other branch that we merged
#                         deleteat!(brs, res[2])
#                             println("──▶ #br = $(length(brs))\n")
#                     end
#                     iterbr += 1
#                 end
#
#                 break
#             end
#         end
#         iterbrs += 1
#     end
#     iterbrs < iterbrsmax && printstyled("──▶ Merge computed!!!", color=:green)
# end
