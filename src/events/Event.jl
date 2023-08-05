abstract type AbstractEvent end
abstract type AbstractContinuousEvent <: AbstractEvent end
abstract type AbstractDiscreteEvent <: AbstractEvent end

# evaluate the functional whose events are sought
(eve::AbstractEvent)(iter, state) = eve.condition(iter, state)

# initialize function, must return the same type as eve(iter, state)
initialize(eve::AbstractEvent, T) = throw("Initialization method not implemented for event ", eve)

# whether the event requires computing eigen-elements
@inline computeEigenElements(::AbstractEvent) = false

length(::AbstractEvent) = throw("length not implemented")

# default label used to record event in ContResult
labels(::AbstractEvent, ind) = "user"

# whether the user provided its own labels
hasCustomLabels(::AbstractEvent) = false

# general condition for detecting a continuous event.
function testEve(eve::AbstractContinuousEvent, x, y)
    ϵ = eve.tol
    return (x * y < 0) || (abs(x) <= ϵ) || (abs(y) <= ϵ)
end

# is x actually an event
function isOnEvent(eve::AbstractContinuousEvent, eventValue) 
    for u in eventValue
        if  abs(u) <= eve.tol
            return true
        end
    end
    return false
end

# Basically, we want to detect if some component of `eve(fct(iter, state))` is below ϵ
# the ind is used to specify which part of the event is tested
function isEventCrossed(eve::AbstractContinuousEvent, iter, state, ind = :)
    if state.eventValue[1] isa Real
        return testEve(eve, state.eventValue[1], state.eventValue[2])
    else
        for u in zip(state.eventValue[1][ind], state.eventValue[2][ind])
            if testEve(eve, u[1], u[2])
                return true
            end
        end
        return false
    end
end

# general condition for detecting a discrete event
testEve(eve::AbstractDiscreteEvent, x, y) = x != y
isOnEvent(::AbstractDiscreteEvent, x) = false

function isEventCrossed(eve::AbstractDiscreteEvent, iter, state, ind = :)
    if state.eventValue[1] isa Integer
        return testEve(eve, state.eventValue[1], state.eventValue[2])
    else
        for u in zip(state.eventValue[1][ind], state.eventValue[2][ind])
            if testEve(eve, u[1], u[2])
                return true
            end
        end
        return false
    end
end
####################################################################################################
# for AbstractContinuousEvent and AbstractDiscreteEvent
# return type when calling eve.fct(iter, state)
initialize(eve::AbstractContinuousEvent, T) = eve.nb == 1 ? T(1) : ntuple(x -> T(1), eve.nb)
initialize(eve::AbstractDiscreteEvent, T) = eve.nb == 1 ? Int64(1) : ntuple(x -> Int64(1), eve.nb)
####################################################################################################
"""
$(TYPEDEF)

Structure to pass a ContinuousEvent function to the continuation algorithm.
A continuous call back returns a **tuple/scalar** value and we seek its zeros.

$(TYPEDFIELDS)
"""
struct ContinuousEvent{Tcb, Tl, T} <: AbstractContinuousEvent
    "number of events, ie the length of the result returned by the callback function"
    nb::Int64

    ", ` (iter, state) -> NTuple{nb, T}` callback function which, at each continuation state, returns a tuple. For example, to detect crossing 1.0 and -2.0, you can pass `(iter, state) -> (getp(state)+2, getx(state)[1]-1)),`. Note that the type `T` should match the one of the parameter specified by the `::Lens` in `continuation`."
    condition::Tcb

    "whether the event requires to compute eigen elements"
    computeEigenElements::Bool

    "Labels used to display information. For example `labels[1]` is used to qualify an event of the type `(0,1.3213,3.434)`. You can use `labels = (\"hopf\",)` or `labels = (\"hopf\", \"fold\")`. You must have `labels::Union{Nothing, NTuple{N, String}}`."
    labels::Tl

    "Tolerance on event value to declare it as true event."
    tol::T
end

ContinuousEvent(nb::Int, fct, labels::Union{Nothing, NTuple{N, String}} = nothing) where N = (@assert nb > 0 "You need to return at least one callback"; ContinuousEvent(nb, fct, false, labels, 0))
@inline computeEigenElements(eve::ContinuousEvent) = eve.computeEigenElements
@inline length(eve::ContinuousEvent) = eve.nb
@inline hasCustomLabels(eve::ContinuousEvent{Tcb, Tl}) where {Tcb, Tl} = ~(Tl == Nothing)
####################################################################################################
"""
$(TYPEDEF)

Structure to pass a DiscreteEvent function to the continuation algorithm.
A discrete call back returns a discrete value and we seek when it changes.

$(TYPEDFIELDS)
"""
struct DiscreteEvent{Tcb, Tl} <: AbstractDiscreteEvent
    "number of events, ie the length of the result returned by the callback function"
    nb::Int64

    "= ` (iter, state) -> NTuple{nb, Int64}` callback function which at each continuation state, returns a tuple. For example, to detect a value change."
    condition::Tcb

    "whether the event requires to compute eigen elements"
    computeEigenElements::Bool

    "Labels used to display information. For example `labels[1]` is used to qualify an event occurring in the first component. You can use `labels = (\"hopf\",)` or `labels = (\"hopf\", \"fold\")`. You must have `labels::Union{Nothing, NTuple{N, String}}`."
    labels::Tl
end
DiscreteEvent(nb::Int, fct, labels::Union{Nothing, NTuple{N, String}} = nothing) where N = (@assert nb > 0 "You need to return at least one callback"; DiscreteEvent(nb, fct, false, labels))
@inline computeEigenElements(eve::DiscreteEvent) = eve.computeEigenElements
@inline length(eve::DiscreteEvent) = eve.nb
@inline hasCustomLabels(eve::DiscreteEvent{Tcb, Tl}) where {Tcb, Tl} = ~(Tl == Nothing)

function labels(eve::Union{ContinuousEvent{Tcb, Nothing}, DiscreteEvent{Tcb, Nothing}}, ind) where Tcb
    return "userC" * mapreduce(x -> "-$x", *, ind)
end

function labels(eve::Union{ContinuousEvent{Tcb, Tl}, DiscreteEvent{Tcb, Tl}}, ind) where {Tcb, Tl}
    if isempty(ind)
        return "user"
    end
    return mapreduce(x -> eve.labels[x], *, ind)
end
####################################################################################################
"""
$(TYPEDEF)

Structure to pass a PairOfEvents function to the continuation algorithm. It is composed of a pair ContinuousEvent / DiscreteEvent. A `PairOfEvents`
is constructed by passing to the constructor a `ContinuousEvent` and a `DiscreteEvent`:

    PairOfEvents(contEvent, discreteEvent)

## Fields
$(TYPEDFIELDS)
"""
struct PairOfEvents{Tc <: AbstractContinuousEvent, Td <: AbstractDiscreteEvent}  <: AbstractEvent
    "Continuous event"
    eventC::Tc

    "Discrete event"
    eventD::Td
end

@inline computeEigenElements(eve::PairOfEvents) = computeEigenElements(eve.eventC) || computeEigenElements(eve.eventD)
@inline length(event::PairOfEvents) = length(event.eventC) + length(event.eventD)
# is x actually an event, we just need to test the continuous part
isOnEvent(eve::PairOfEvents, x) = isOnEvent(eve.eventC, x[1:length(eve.eventC)])

function (eve::PairOfEvents)(iter, state)
    outc = eve.eventC(iter, state)
    outd = eve.eventD(iter, state)
    return outc..., outd...
end

initialize(eve::PairOfEvents, T) = initialize(eve.eventC, T)..., initialize(eve.eventD, T)...
function isEventCrossed(eve::PairOfEvents, iter, state, ind = :)
    nc = length(eve.eventC)
    n = length(eve)
    resC = isEventCrossed(eve.eventC, iter, state, 1:nc)
    resD = isEventCrossed(eve.eventD, iter, state, nc+1:n)
    return resC || resD
end
####################################################################################################
"""
$(TYPEDEF)

Multiple events can be chained together to form a `SetOfEvents`. A `SetOfEvents`
is constructed by passing to the constructor `ContinuousEvent`, `DiscreteEvent` or other `SetOfEvents` instances:

    SetOfEvents(cb1, cb2, cb3)

# Example

     BifurcationKit.SetOfEvents(BK.FoldDetectCB, BK.BifDetectCB)

You can pass as many events as you like.

$(TYPEDFIELDS)
"""
struct SetOfEvents{Tc <: Tuple, Td <: Tuple}  <: AbstractEvent
    "Continuous event"
    eventC::Tc

    "Discrete event"
    eventD::Td
end

SetOfEvents(callback::AbstractDiscreteEvent) = SetOfEvents((),(callback,))
SetOfEvents(callback::AbstractContinuousEvent) = SetOfEvents((callback,),())
SetOfEvents() = SetOfEvents((),())
SetOfEvents(cb::Nothing) = SetOfEvents()

# For Varargs, use recursion to make it type-stable
SetOfEvents(events::Union{AbstractEvent, Nothing}...) = SetOfEvents(split_events((), (), events...)...)

"""
    split_events(cs, ds, args...)
Split comma separated callbacks into sets of continuous and discrete callbacks. Inspired by DiffEqBase.
"""
@inline split_events(cs, ds) = cs, ds
@inline split_events(cs, ds, c::Nothing, args...) = split_events(cs, ds, args...)
@inline split_events(cs, ds, c::AbstractContinuousEvent, args...) = split_events((cs..., c), ds, args...)
@inline split_events(cs, ds, d::AbstractDiscreteEvent, args...) = split_events(cs, (ds..., d), args...)
@inline function split_events(cs, ds, d::SetOfEvents, args...)
  split_events((cs...,d.eventC...), (ds..., d.eventD...), args...)
end

@inline computeEigenElements(eve::SetOfEvents) = mapreduce(computeEigenElements, |, eve.eventC) || mapreduce(computeEigenElements, |, eve.eventD)

function (eve::SetOfEvents)(iter, state)
    outc = map(x -> x(iter, state), eve.eventC)
    outd = map(x -> x(iter, state), eve.eventD)
    return (outc..., outd...)
end

initialize(eve::SetOfEvents, T) = map(x->initialize(x,T),eve.eventC)..., map(x->initialize(x,T),eve.eventD)...

# is x actually an event, we just need to test the continuous events
function isOnEvent(eves::SetOfEvents, eValues)
    out = false
    for (index, eve) in pairs(eves.eventC)
        out = out | isOnEvent(eve, eValues[index])
    end
    return out
end  

function isEventCrossed(event::SetOfEvents, iter, state)
    res = false
    nC = length(event.eventC)
    nD = length(event.eventD)
    nCb = nC+nD
    for (i, eve) in enumerate(event.eventC)
        res = res | isEventCrossed(eve, iter, state, i)
    end
    for (i, eve) in enumerate(event.eventD)
        res = res | isEventCrossed(eve, iter, state, nC + i)
    end
    return  res
end
