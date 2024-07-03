"""
    `SaveAtEvent(positions::Tuple)`

This event implements the detection of when the parameter values, used during continuation, equals one of the values in `positions`. This state is then saved in the branch.

For example, you can use it like `continuation(args...; event = SaveAtEvent((1., 2., -3.)))`
"""
function SaveAtEvent(positions::Tuple) 
    labels = length(positions) == 1 ? ("save",) : ntuple(x -> "save-$x", length(positions))
    ContinuousEvent(length(positions), (it, state) -> map(x -> x - getp(state), positions), labels)
end
####################################################################################################
# detection of Fold bifurcation, should be based on Bordered
"""
    `FoldDetectEvent`

This event implements the detection of Fold points based on the p-component of the tangent vector to the continuation curve. It is designed to work with `PALC(tangent = Bordered())` as continuation algorithm. To use it, pass `event = FoldDetectEvent` to `continuation`.
"""
FoldDetectEvent = ContinuousEvent(1, (it, state) -> state.τ.p, ("fold",))
####################################################################################################
# detection of codim 1 bifurcation
struct BifEvent{Tcb} <: AbstractDiscreteEvent
    nb::Int
    condition::Tcb
end

compute_eigenelements(::BifEvent) = true
@inline length(eve::BifEvent) = eve.nb
@inline has_custom_labels(::BifEvent) = true

function detect_bifurcation_event(iter, state)
    # Note that the computation of eigen-elements should have occurred before events are called
    # state should be thus up to date at this stage
    @assert state.n_unstable[1] >=0 "Issue with `detect_bifurcation_event`. Please open an issue on https://github.com/rveltz/BifurcationKit.jl/issues."
    # put the max because n_unstable is initialized at -1 at the beginning of the continuation
    return convert_to_tuple_eve(max(0, state.n_unstable[1]))
end

"""
    `BifDetectEvent`

This event implements the detection of bifurcations points along a continuation curve. The detection is based on monitoring the number of unstable eigenvalues. More details are given at [Detection of bifurcation points of Equilibria](@ref).
"""
BifDetectEvent = BifEvent(1, detect_bifurcation_event)

function get_event_type(event::BifEvent,
                        iter::AbstractContinuationIterable, 
                        state, 
                        verbosity, 
                        status::Symbol, 
                        interval::Tuple{T, T}, 
                        ind = :; typeE = :user) where T
    return get_bifurcation_type(iter, state, status, interval)
end
