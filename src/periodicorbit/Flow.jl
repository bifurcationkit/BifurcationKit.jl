using DiffEqBase

####################################################################################################
# this function takes into accound a parameter passed to the vector field
# Putting the options `save_start=false` seems to bug with Sundials
function flowTimeSol(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)
	return sol.t[end], sol[end]
end

flow(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = flowTimeSol(x, p, tspan, pb; alg = alg, kwargs...)[2]
flow(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...) =  flow(x, nothing, tspan, pb; alg = alg, kwargs...)
################
# function used to compute the derivative of the flow, so pb encodes the variational equation
function dflow(x::AbstractVector, dx, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	n = length(x)
	_prob = DiffEqBase.remake(pb; u0 = vcat(x, dx), tspan = tspan, p = p)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)[end]
	return sol[1:n], sol[n+1:end]
end

dflow(x, dx, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = dflow(x, dx, nothing, tspan, pb; alg = alg, kwargs...)
################
# this function takes into accound a parameter passed to the vector field
function dflow_fd(x, dx, p, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...)
	sol1 = flow(x .+ δ .* dx, p, tspan, pb; alg = alg, kwargs...)
	sol2 = flow(x 			, p, tspan, pb; alg = alg, kwargs...)
	return sol2, (sol1 .- sol2) ./ δ
end
dflow_fd(x, dx, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...) = dflow_fd(x, dx, nothing, tspan, pb; alg = alg, δ = δ, kwargs...)
################
# this gives access to the full solution, convenient for Poincaré shooting
# this function takes into accound a parameter passed to the vector field and returns the full solution from the ODE solver. This is useful in Poincare Shooting to extract the period.
function flowFull(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	sol = DiffEqBase.solve(_prob, alg; kwargs...)
end
flowFull(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = flowFull(x, nothing, tspan, pb; alg = alg, kwargs...)
####################################################################################################
# Structures related to computing ODE/PDE Flows
"""
	fl = Flow(F, flow, dflow)
This composite type encapsulates:
- the vector field `x -> F(x)` associated to a Cauchy problem,
- the flow (or semigroup) associated to the Cauchy problem `(x, t) -> flow(x, t)`. Only the last time point must be returned.
- the flow (or semigroup) associated to the Cauchy problem `(x, t) -> flow(x, t)`. The whole solution on the time interval (0,t) must be returned. This is not strictly necessary to provide this.
- the differential `dflow` of the flow w.r.t. `x`, `(x, dx, t) -> dflow(x, dx, t)`. One important thing is that we require `dflow(x, dx, t)` to return 2 vectors: the first is `flow(x, t)` and the second is the value of the derivative of the flow as the second vector.

There are some simple constructors for which you only have to pass a `prob::ODEProblem` from `DifferentialEquations.jl` and an ODE time stepper like `Tsit5()`. Hence, you can do for example

	fl = Flow(F, prob, Tsit5(); kwargs...)

If your vector field depends on parameters `p`, you can define a `Flow` using

	fl = Flow(F, p, prob, Tsit5(); kwargs...)

Finally, you can pass two `ODEProblem` where the second one is used to compute the variational equation:

	fl = Flow(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)

"""
struct Flow{TF, Tf, Tts, Tff, Td}
	F::TF				# vector field F(x)
	flow::Tf			# flow(x,t)
	flowTimeSol::Tts	# flow(x,t)
	flowFull::Tff		# flow(x,t) returns full solution
	dflow::Td			# dflow(x,dx,t) returns (flow(x,t), dflow(x,t)⋅dx)

	function Flow(F::TF, fl::Tf, fts::Tts, flf::Tff, df::Td) where {TF, Tf, Tts, Tff, Td}
		new{TF, Tf, Tts, Tff, Td}(F, fl, fts, flf, df)
	end

	function Flow(F::TF, fl::Tf, df::Td) where {TF, Tf, Td}
		new{TF, Tf, Nothing, Nothing, Td}(F, fl, nothing, nothing, df)
	end
end

(fl::Flow)(x, tspan)     			  = fl.flow(x, tspan)
(fl::Flow)(x, dx, tspan; kw2...) 	  = fl.dflow(x, dx, tspan; kw2...)
(fl::Flow)(::Val{:Full}, x, tspan) 	  = fl.flowFull(x, tspan)
(fl::Flow)(::Val{:TimeSol}, x, tspan) = fl.flowTimeSol(x, tspan)

"""
Creates a Flow variable based on a `prob::ODEProblem` and ODE solver `alg`. The vector field `F` has to be passed, this will be resolved in the future as it can be recovered from `prob`. Also, the derivative of the flow is estimated with finite differences.
"""
# this constructor takes into accound a parameter passed to the vector field
function Flow(F, p, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) ->			 	flow(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) ->		 flowTimeSol(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) -> 		 	flowFull(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		# we remove the callback in order to use this for the Jacobian in Poincare Shooting
		(x, dx, t; kw2...) -> dflow_fd(x, dx, p, (zero(t), t), prob; alg = alg, kwargs..., kw2...)
		)
end

function Flow(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)
	return Flow(F,
		(x, t) -> 			flow(x, p, (zero(t), t), prob1, alg = alg1; kwargs...),
		(x, t) ->	 flowTimeSol(x, p, (zero(t), t), prob1; alg = alg1, kwargs...),
		(x, t) -> 		flowFull(x, p, (zero(t), t), prob1, alg = alg1; kwargs...),
		(x, dx, t; kw2...) ->  dflow(x, dx, p, (zero(t), t), prob2; alg = alg2, kwargs..., kw2...)
		)
end
