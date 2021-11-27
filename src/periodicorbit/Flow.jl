using DiffEqBase: remake, solve, ODEProblem, EnsembleProblem, EnsembleThreads, DAEProblem, isinplace
abstract type AbstractFlow end

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

where `kwargs` is passed to `DiffEqBase::solve`. If your vector field depends on parameters `p`, you can define a `Flow` using

	fl = Flow(prob, Tsit5(); kwargs...)

Finally, you can pass two `ODEProblem` where the second one is used to compute the variational equation:

	fl = Flow(prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)

"""
@with_kw struct Flow{TF, Tf, Tts, Tff, Td, Tse, Tprob, TprobMono, Tfs, Tcb} <: AbstractFlow
	"The vector field `(x, p) -> F(x, p)` associated to a Cauchy problem. Used for the differential of the shooting problem."
	F::TF = nothing

	"The flow (or semigroup) `(x, p, t) -> flow(x, p, t)` associated to the Cauchy problem. Only the last time point must be returned in the form (u = ...)"
	flow::Tf = nothing

	"Flow which returns the tuple (t, u(t)). Optional, mainly used for plotting on the user side."
	flowTimeSol::Tts = nothing

	"[Optional] The flow (or semigroup) associated to the Cauchy problem `(x, p, t) -> flow(x, p, t)`. The whole solution on the time interval [0,t] must be returned. It is not strictly necessary to provide this, it is mainly used for plotting on the user side. Please use `nothing` as default."
	flowFull::Tff = nothing

	"The differential `dflow` of the flow *w.r.t.* `x`, `(x, p, dx, t) -> dflow(x, p, dx, t)`. One important thing is that we require `dflow(x, dx, t)` to return a Named Tuple: `(t = t, u = flow(x, p, t), du = dflow(x, p, dx, t))`, the last component being the value of the derivative of the flow."
	dflow::Td = nothing

	"[Optional] Serial version of dflow. Used internally when using parallel multiple shooting. Please use `nothing` as default."
	dfSerial::Tse = nothing

	"[Internal] store the ODEProblem associated to the flow of the Cauchy problem"
	prob::Tprob = nothing

	"[Internal] store the ODEProblem associated to the flow of the variational problem"
	probMono::TprobMono = nothing

	"[Internal] Serial version of the flow"
	flowSerial::Tfs = nothing

	"[Internal] Store possible callback"
	callback::Tcb = nothing
end

# constructors
Flow(F, fl, df = nothing) = Flow(F = F, flow = fl, dflow = df, dfSerial = df)

# callable struct
(fl::Flow)(x, p, t; k...)     			  			= fl.flow(x, p, t; k...)
(fl::Flow)(x, p, dx, t; k...) 	  					= fl.dflow(x, p, dx, t; k...)
(fl::Flow)(::Val{:Full}, x, p, t; k...) 	  		= fl.flowFull(x, p, t; k...)
(fl::Flow)(::Val{:TimeSol}, x, p, t; k...)  		= fl.flowTimeSol(x, p, t; k...)
(fl::Flow)(::Val{:SerialTimeSol}, x, p, t; k...)   	= fl.flowSerial(x, p, t; k...)
(fl::Flow)(::Val{:SerialdFlow}, x, p, dx, t; k...)  = fl.dfSerial(x, p, dx, t; k...)

function getVectorField(prob::Union{ODEProblem, DAEProblem})
	if isinplace(prob)
		return (x, p) -> (out = similar(x); prob.f(out, x, p, prob.tspan[1]); return out)
	else
		return (x, p) -> prob.f(x, p, prob.tspan[1])
	end
end
getVectorField(pb::EnsembleProblem) = getVectorField(pb.prob)
