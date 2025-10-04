abstract type AbstractPredictor end
abstract type AbstractTangentComputation end

initialize!(state::AbstractContinuationState,
                iter::AbstractContinuationIterable) = initialize!(state, iter, getalg(iter))

# compute the predictor given the tangent already contained in state
getpredictor!(state::AbstractContinuationState,
                iter::AbstractContinuationIterable) = getpredictor!(state, iter, getalg(iter))

# update the predictor given the tangent already contained in state
# this is used in Bifurcation / Event detection using bisection
# basically a dispatch to next method
update_predictor!(state::AbstractContinuationState,
                iter::AbstractContinuationIterable) = update_predictor!(state, iter, getalg(iter))

# default method
update_predictor!(state::AbstractContinuationState,
                iter::AbstractContinuationIterable,
                alg::AbstractContinuationAlgorithm) = getpredictor!(state, iter, alg)

corrector!(state::AbstractContinuationState,
            iter::AbstractContinuationIterable; kwargs...) = corrector!(state, iter, getalg(iter); kwargs...)

step_size_control!(state::AbstractContinuationState,
            iter::AbstractContinuationIterable; kwargs...) = step_size_control!(state, iter, getalg(iter); kwargs...)

# used to reset the predictor when locating bifurcations
Base.empty!(alg::AbstractContinuationAlgorithm) = alg
Base.empty!(alg::AbstractTangentComputation) = alg
# Base.copy(::AbstractContinuationAlgorithm) = throw("Not defined. Please define a copy method for your continuation algorithm")

# name to be print in show(::AbstractBranch)
_shortname(alg::AbstractContinuationAlgorithm) = typeof(alg).name.name

# we need to be able to reset / empty the predictors when locating bifurcation points and when doing automatic branch switching
function Base.empty(alg::AbstractContinuationAlgorithm)
    alg2 = deepcopy(alg)
    empty!(alg2)
    alg2
end

# this is called during initialization of the continuation method. Can be used to adjust the algo.
update(alg::AbstractContinuationAlgorithm, ::ContinuationPar, _) = alg

# helper functions to update ::ContState when calling the corrector
function _update_field_but_not_sol!(state::AbstractContinuationState,
                                    sol::NonLinearSolution)
    state.converged = sol.converged
    state.itnewton  = sol.itnewton
    state.itlinear  = sol.itlineartot
    # record previous solution
    if converged(sol)
        copyto!(state.z_old, state.z)
    end
end
####################################################################################################
function step_size_control!(state::AbstractContinuationState,
                            iter::AbstractContinuationIterable,
                            ::AbstractContinuationAlgorithm)
    if ~state.stopcontinuation && stepsizecontrol(state)
        _step_size_control!(state, getcontparams(iter), iter.verbosity)
    end
end

function _step_size_control!(state, contparams::ContinuationPar, verbosity)
    ds = state.ds
    if converged(state) == false
        if  abs(ds) <= contparams.dsmin
            @error "Failure to converge with given tolerance = $(contparams.newton_options.tol).\nStep = $(state.step)\nYou can decrease the tolerance or pass a different norm using the argument `normC`.\nWe reached the smallest value [dsmin] valid for ds, namely $(contparams.dsmin).\nStopping continuation at continuation step $(state.step)."
            # we stop the continuation
            state.stopcontinuation = true
            return
        end
        dsnew = sign(ds) * max(abs(ds) / 2, contparams.dsmin);
        (verbosity > 0) && printstyled("Halving ds to $(dsnew)\n", color = :red)

    else
        # control to have the same number of Newton iterations
        Nmax = contparams.newton_options.max_iterations
        factor = (Nmax - state.itnewton) / Nmax
        dsnew = ds * (1 + contparams.a * factor^2)
    end

    dsnew = clamp_ds(dsnew, contparams)

    # we do not stop the continuation
    state.ds = dsnew
    state.stopcontinuation = false
    return
end
