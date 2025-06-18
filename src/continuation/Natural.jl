struct ConstantPredictor <: AbstractTangentComputation end
"""
    Natural continuation algorithm. The predictor is the constant predictor and the parameter is incremented by `ContinuationPar().ds` at each continuation step.
"""
struct Natural <: AbstractContinuationAlgorithm
    bothside::Bool
end
Natural() = Natural(false)
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(::Natural, ::Bool) = nothing

function initialize!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg::Natural, nrm = false)
    # we want to start at (u0, p0), not at (u1, p1)
    copyto!(state.z, state.z_old)
    getpredictor!(state, iter, alg)
end

# this function mutates the predictor located in z_pred
function getpredictor!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg::Natural, nrm = false)
    copyto!(state.z_pred, state.z)
    state.z_pred.p += state.ds
end

function corrector!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    alg::Natural; kwargs...)
    sol = _newton(it.prob,
                state.z_pred.u,
                setparam(it, clamp_predp(state.z_pred.p, it)), 
                it.contparams.newton_options; 
                normN = it.normC, 
                callback = it.callback_newton, 
                kwargs...)

    # update fields
    _update_field_but_not_sol!(state, sol)

    # update solution
    if converged(sol)
        copyto!(state.z.u, sol.u)
        state.z.p = state.z_pred.p
    end

    return true
end
