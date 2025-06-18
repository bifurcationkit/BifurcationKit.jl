"""
$(TYPEDEF)

Continuation algorithm which switches automatically between Natural continuation and PALC (or other if specified) depending on the stiffness of the branch being continued.

$(TYPEDFIELDS)

"""
struct AutoSwitch{Talg, T} <: AbstractContinuationAlgorithm
    "Continuation algorithm to switch to when Natural is discarded. Typically `PALC()`"
    alg::Talg
    "tolerance for switching to PALC(), default value = 0.5"
    tol_param::T
end

function AutoSwitch(;alg = PALC(), tol_param = 0.5)
    return AutoSwitch(alg, tol_param)
end

Base.empty!(alg::AutoSwitch) = empty!(alg.alg)
getθ(alg::AutoSwitch) = getθ(alg.alg)
getdot(alg::AutoSwitch) = getdot(alg.alg)
getlinsolver(alg::AutoSwitch) = getlinsolver(alg.alg)
internal_adaptation!(alg::AutoSwitch, onoroff::Bool) = internal_adaptation!(alg.alg, onoroff)

function update(alg::AutoSwitch, contParams::ContinuationPar, linear_algo) 
    alg2 = update(alg.alg, contParams, linear_algo)
    @set alg.alg = alg2
end

function getpredictor!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg_switch::AutoSwitch,
                        nrm = false)
    alg = alg_switch.alg
    # we compute the tangent
    # if the state has not converged, we dot not update the tangent
    # state.z has been updated only if converged(state) == true
    if converged(state)
        @debug "Update tangent AutoSwitch"
        gettangent!(state, iter, alg.tangent, getdot(alg))
    end
    # then update the predictor state.z_pred
    addtangent!(state, nrm)
end

function initialize!(state::AbstractContinuationState,
                    iter::AbstractContinuationIterable,
                    alg::AutoSwitch,
                    nrm = false)
    initialize!(state, iter, alg.alg, nrm)
end

function corrector!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    alg::AutoSwitch; 
                    kwargs...)
    τ = state.τ
    λ = τ.p
    θ = getθ(it)
    dotθ = getdot(alg.alg)
    @debug "" (1-θ)*abs(λ) dotθ(τ, θ)
    if (1-θ)*abs(λ) > alg.tol_param
        @debug "NATURAL" λ
        corrector!(state, it, Natural(); kwargs...)
    else
        @debug "PALC" λ
        corrector!(state, it, alg.alg; kwargs...)
    end
    return true
end