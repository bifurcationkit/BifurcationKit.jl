abstract type AbstractEvent end
abstract type AbstractContinuousEvent <: AbstractEvent end
abstract type AbstractDiscreteEvent <: AbstractEvent end

# evaluate the functional whose events are seek
(eve::AbstractEvent)(iter, state) = eve.condition(iter, state)

# initialize function, must return the same type as eve(iter, state)
initialize(eve::AbstractEvent, T) = throw("Initialization method not implemented for event ", eve)

# whether the event requires computing eigen-elements
@inline computeEigenElements(::AbstractEvent) = false

# general condition for detecting a continuous event.
# Basically, we want to detect if some component of `abs.(fct(iter, state))` is below ϵ
isEvent(::AbstractEvent, iter, state) = !isnothing(findfirst(x -> abs(x) < iter.contParams.tolBisectionEvent, state.eventValue[2]))

# this function is called to determine if callbaclVals is an event
test(::AbstractEvent, callbaclVals, precision) = false
####################################################################################################
