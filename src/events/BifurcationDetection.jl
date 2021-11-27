"""
	`SaveAtEvent(positions::Tuple)`

This event implements the detection of when the parameter values, used during continuation, equals one of the values in `positions`. This state is then saved in the branch.

For example, you can use it like `continuation(args...; event = SaveAtEvent((1., 2., -3.)))`
"""
SaveAtEvent(positions::Tuple) = ContinuousEvent(length(positions), (it, state) -> map(x -> x - getp(state), positions), ntuple(x->"save-$x",length(positions)))
####################################################################################################
# detection of Fold bifurcation, should be based on BorderedPred
"""
	`FoldDetectEvent`

This event implements the detection of Fold points based on the p-component of the tangent vector to the continuation curve. It is designed to work with the predictor `BorderedPred()` that you pass to `continuation` with the keyword argument `tangentAlgo`.
"""
FoldDetectEvent = ContinuousEvent(1, (it, state) -> state.tau.p, ("fold",))
####################################################################################################
# detection of codim 1 bifurcation
struct BifEvent{Tcb} <: AbstractDiscreteEvent
	nb::Int
	condition::Tcb
end

computeEigenElements(::BifEvent) = true
@inline length(eve::BifEvent) = eve.nb
@inline hasCustomLabels(::BifEvent) = true

function detectBifurcationEVE(iter, state)
	# Note that the computation of eigen-elements should have occured before events are called
	# state should be thus up to date at this stage
	@assert state.n_unstable[1] >=0 "Issue with `detectBifurcationEVE`. Please open an issue on https://github.com/rveltz/BifurcationKit.jl/issues."
	# put the max because n_unstable is initialized at -1 at the beginning of the continuation
	return max(0, state.n_unstable[1])
end

"""
	`BifDetectEvent`

This event implements the detection of bifurcations points along a continuation curve. The detection is based on monitoring the number of unstable eigenvalues. More details are given at [Detection of bifurcation points](@ref).
"""
BifDetectEvent = BifEvent(1, detectBifurcationEVE)

function getEventType(event::BifEvent, iter::AbstractContinuationIterable, state, verbosity, status::Symbol, interval::Tuple{T, T}, ind = :; typeE = :user) where T
	return getBifurcationType(iter.contParams, state, iter.normC, iter.recordFromSolution, verbosity, status, interval)
end
