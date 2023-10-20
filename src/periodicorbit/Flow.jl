abstract type AbstractFlow end

# The vector field `F(x, p)` associated to a Cauchy problem ẋ = F(x, p). Used for the differential of the shooting problem. The vector field is used like `vf(flow, x, p)`` and must return `F(x, p)``
function vf(::AbstractFlow, x, par; k...) end

# return a real number (like 1e-8) used to compute derivative w.r.t. the parameter by finite differences. This is used for example in PALC, Moore-Penrose, etc.
function getdelta(::AbstractFlow) end

# these functions are used in the Standard Shooting method
# the function implements the flow (or semigroup) `(x, p, t) -> flow(x, p, t)` associated to an autonomous Cauchy problem. Only the last time point must be returned in the form Named Tuple `(u = ..., t = t)`. In the case of Poincaré Shooting, one must be able to call the flow like `evolve(fl, x, par, Inf)`.
function evolve(::AbstractFlow, x, par, δt; k...) end

# The differential `dflow` of the flow *w.r.t.* `x`, `(x, p, dx, t) -> dflow(x, p, dx, t)`. One important thing is that we require `dflow(x, p, dx, t)` to return a Named Tuple: `(t = t, u = flow(x, p, t), du = dflow(x, p, dx, t))`, the last component being the value of the derivative of the flow.
function jvp(::AbstractFlow, x, par, dx, δt; k...) end

# The adjoint differential `vjpflow` of the flow *w.r.t.* `x`, `(x, p, dx, t) -> vjpflow(x, p, dx, t)`. One important thing is that we require `vjpflow(x, p, dx, t)` to return a Named Tuple: `(t = t, u = flow(x, p, t), du = vjpflow(x, p, dx, t))`, the last component being the value of the derivative of the flow.
function vjp(::AbstractFlow, x, par, dx, δt; k...) end

# [Optional] The function implements the flow (or semigroup) associated to an autonomous Cauchy problem `(x, p, t) -> flow(x, p, t)`. The whole solution on the time interval [0,t] must be returned. It is not strictly necessary to provide this, it is mainly used for plotting on the user side. In the case of Poincaré Shooting, one must be able to call the flow like `evolve(fl, Val(:Full), x, par, Inf)`.
function evolve(::AbstractFlow, ::Val{:Full}, x, par, δt; k...) end

# [Optional / Internal] Serial version of the flow. Used for Matrix based jacobian (Shooting and Poincaré Shooting) and diffPoincareMap. Must return a Named Tuple `(u = ..., t = t)`
function evolve(fl::AbstractFlow, ::Val{:SerialTimeSol}, x, par, δt; k...) end

# [Optional] Flow which returns the tuple `(t, u(t))`. Optional, mainly used for plotting on the user side.
function evolve(::AbstractFlow, ::Val{:TimeSol}, x, par, δt = Inf; k...) end

# [Optional] Serial version of `dflow`. Used internally for parallel multiple shooting. Returns a named Tuple `(u = ..., du = ..., t = t)`
function evolve(::AbstractFlow, ::Val{:SerialdFlow}, x, par, dx, tΣ; kwargs...) end

####################################################################################################
# Structures related to computing ODE/PDE Flows
"""
$(TYPEDEF)
$(TYPEDFIELDS)

# Simplified constructor(s)
We provide a simple constructor where you only pass the vector field `F`, the flow `ϕ` and its differential `dϕ`:

    fl = Flow(F, ϕ, dϕ)

# Simplified constructors for DifferentialEquations.jl

These are some simple constructors for which you only have to pass a `prob::ODEProblem` or `prob::EnsembleProblem` (for parallel computation) from `DifferentialEquations.jl` and an ODE time stepper like `Tsit5()`. Hence, you can do for example

    fl = Flow(prob, Tsit5(); kwargs...)

where `kwargs` is passed to `SciMLBase::solve`. If your vector field depends on parameters `p`, you can define a `Flow` using

    fl = Flow(prob, Tsit5(); kwargs...)

Finally, you can pass two `ODEProblem` where the second one is used to compute the variational equation:

    fl = Flow(prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)

"""
@with_kw struct Flow{TF, Tf, Tts, Tff, Td, Tad, Tse, Tprob, TprobMono, Tfs, Tcb, Tδ} <: AbstractFlow
    "The vector field `(x, p) -> F(x, p)` associated to a Cauchy problem. Used for the differential of the shooting problem."
    F::TF = nothing

    "The flow (or semigroup) `(x, p, t) -> flow(x, p, t)` associated to the Cauchy problem. Only the last time point must be returned in the form (u = ...)"
    flow::Tf = nothing

    "Flow which returns the tuple (t, u(t)). Optional, mainly used for plotting on the user side."
    flowTimeSol::Tts = nothing

    "[Optional] The flow (or semigroup) associated to the Cauchy problem `(x, p, t) -> flow(x, p, t)`. The whole solution on the time interval [0,t] must be returned. It is not strictly necessary to provide this, it is mainly used for plotting on the user side. Please use `nothing` as default."
    flowFull::Tff = nothing

    "The differential `dflow` of the flow *w.r.t.* `x`, `(x, p, dx, t) -> dflow(x, p, dx, t)`. One important thing is that we require `dflow(x, dx, t)` to return a Named Tuple: `(t = t, u = flow(x, p, t), du = dflow(x, p, dx, t))`, the last component being the value of the derivative of the flow."
    jvp::Td = nothing

    "The adjoint differential `vjpflow` of the flow *w.r.t.* `x`, `(x, p, dx, t) -> vjpflow(x, p, dx, t)`. One important thing is that we require `vjpflow(x, p, dx, t)` to return a Named Tuple: `(t = t, u = flow(x, p, t), du = vjpflow(x, p, dx, t))`, the last component being the value of the derivative of the flow."
    vjp::Tad = nothing

    "[Optional] Serial version of dflow. Used internally when using parallel multiple shooting. Please use `nothing` as default."
    jvpSerial::Tse = nothing

    "[Internal] store the ODEProblem associated to the flow of the Cauchy problem"
    prob::Tprob = nothing

    "[Internal] store the ODEProblem associated to the flow of the variational problem"
    probMono::TprobMono = nothing

    "[Internal] Serial version of the flow"
    flowSerial::Tfs = nothing

    "[Internal] Store possible callback"
    callback::Tcb = nothing

    "[Internal]"
    delta::Tδ = 1e-8
end

# constructors
Flow(F, fl, df = nothing; k...) = Flow(;F = F, flow = fl, jvp = df, jvpSerial = df, k...)

vf(fl::Flow, x, p) = fl.F(x, p)
getdelta(fl::Flow) = fl.delta

evolve(fl::Flow, x, p, t; k...)                          = fl.flow(x, p, t; k...)
jvp(fl::Flow, x, p, dx, t; k...)                         = fl.jvp(x, p, dx, t; k...)
evolve(fl::Flow, ::Val{:Full}, x, p, t; k...)            = fl.flowFull(x, p, t; k...)

# for Poincaré Shooting
evolve(fl::Flow, ::Val{:SerialTimeSol}, x, p, t; k...)   = fl.flowSerial(x, p, t; k...)
evolve(fl::Flow, ::Val{:SerialdFlow}, x, p, dx, t; k...) = fl.jvpSerial(x, p, dx, t; k...)
