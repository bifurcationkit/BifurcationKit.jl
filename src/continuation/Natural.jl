struct ConstantPredictor <: AbstractTangentComputation end
"""
	Natural continuation algorithm.
"""
struct Natural <: AbstractContinuationAlgorithm
	bothside::Bool
end
Natural() = Natural(false)
# important for bisection algorithm, switch on / off internal adaptive behavior
internalAdaptation!(::Natural, ::Bool) = nothing

function initialize!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::Natural, nrm = false)
	return nothing
end

# this function mutates the predictor located in z_pred
function getPredictor!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::Natural, nrm = false)
	copyto!(state.z_pred, state.z)
	state.z_pred.p += state.ds
end

function corrector!(state::AbstractContinuationState,
					it::AbstractContinuationIterable,
					alg::Natural; kwargs...)
	sol = _newton(it.prob, state.z_pred.u, setParam(it, clampPredp(state.z_pred.p, it)), it.contParams.newtonOptions; normN = it.normC, callback = it.callbackN, kwargs...)

	# update solution
	copyto!(state.z.u, sol.u)
	state.z.p = state.z_pred.p

	# update fields
	_updatefieldButNotSol!(state, sol)
end
